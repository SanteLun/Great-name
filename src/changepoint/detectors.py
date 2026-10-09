# -*- coding: utf-8 -*-
"""Детекторы точек структурных изменений (шоков).

Все детекторы принимают одномерный ряд y (как правило, ОСТАТКИ прогнозной модели,
стандартизированные) и возвращают индексы точек изменений.
- cusum: классический CUSUM (онлайн)
- peltt / binseg: офлайн-методы из пакета ruptures
- bocpd: Байесовский онлайн-детектор (Adams & MacKay, 2007) — работает онлайн,
  точка за точкой, что критично для задачи «раннего» обнаружения шоков
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("sber_index.changepoint")

try:
    import ruptures as rpt
except Exception as exc:  # noqa: BLE001
    logger.warning("ruptures не доступен: %s", exc)
    rpt = None


# ---------------------------------------------------------------- CUSUM
def detect_cusum(y: np.ndarray, threshold_factor: float = 4.0,
                 drift: float = 0.0, cooldown: int = 3) -> np.ndarray:
    """Классический двусторонний CUSUM с кулдауном между срабатываниями."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)

    # Вход — стандартизированные остатки; фиксированный порог не смотрит в будущее.
    thresh = threshold_factor

    s_pos, s_neg = 0.0, 0.0
    cps = []
    last_cp = -10**9
    for t in range(1, len(y)):
        d = y[t]
        s_pos = max(0.0, s_pos + d - drift)
        s_neg = max(0.0, s_neg - d - drift)
        if (s_pos > thresh or s_neg > thresh) and (t - last_cp >= cooldown):
            cps.append(t)
            s_pos = s_neg = 0.0
            last_cp = t
    return np.array(cps, dtype=int)


# ---------------------------------------------------------------- ruptures
def _fit_ruptures(y: np.ndarray, algo: str, model: str, pen: float) -> np.ndarray:
    if rpt is None:
        raise ImportError("Пакет ruptures не установлен")
    signal = np.asarray(y, dtype=float).reshape(-1, 1)
    if algo == "peltt":
        det = rpt.Pelt(model=model).fit(signal)
        bkps = det.predict(pen=pen)
    elif algo == "binseg":
        det = rpt.Binseg(model=model).fit(signal)
        bkps = det.predict(pen=pen)
    else:
        raise ValueError(f"Неизвестный алгоритм: {algo}")
    # ruptures возвращает границы сегментов (включая len); точки изменений = bkps[:-1]
    bkps = np.asarray(bkps, dtype=int)
    return bkps[bkps < len(y)]


def detect_peltt(y: np.ndarray, model: str = "rbf", pen: float = 10.0) -> np.ndarray:
    return _fit_ruptures(y, "peltt", model, pen)


def detect_binseg(y: np.ndarray, model: str = "rbf", pen: float = 15.0) -> np.ndarray:
    return _fit_ruptures(y, "binseg", model, pen)


# ---------------------------------------------------------------- BOCPD
class BOCPD:
    """Байесовский онлайн-детектор точек изменений (Adams & MacKay, 2007).

    Наблюдения моделируются N(mu, sigma2) с ИЗВЕСТНОЙ дисперсией sigma2 и
    неизвестным средним mu (сопряжённый Normal-Normal априор). Для наших задач
    ряд — стандартизованные остатки прогнозной модели, поэтому sigma2 = 1 —
    корректное допущение, а детектор становится существенно чувствительнее
    к сдвигам уровня, чем вариант с неизвестной дисперсией.
    """

    def __init__(self, hazard: float = 0.05, mu0: float = 0.0, kappa0: float = 1.0,
                 sigma2: float = 1.0, threshold_prob: float = 0.5):
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.sigma2 = sigma2
        self.threshold_prob = threshold_prob
        if not 0 < hazard < 1 or kappa0 <= 0 or sigma2 <= 0:
            raise ValueError("0 < hazard < 1; kappa0/sigma2 > 0")
        self.reset()

    def reset(self) -> None:
        self.run_lengths = [0]
        self.R = np.array([1.0])          # P(r_t | y_1..t)
        self.cps = []
        self._observed = []

    def _predictive_log_prob(self, x: float, mu: float, kappa: float) -> float:
        """log p(x) для N(mu, sigma2*(1 + 1/kappa))."""
        var = self.sigma2 * (1.0 + 1.0 / kappa)
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var

    def update(self, x: float) -> float:
        """Обрабатывает новое наблюдение; возвращает P(r_t=0 | данные)."""
        obs = self._observed
        T = len(self.run_lengths)
        log_predictive = np.empty(T)
        for i, rl in enumerate(self.run_lengths):
            seg = obs[len(obs) - rl:] if rl > 0 else np.array([])
            n = float(len(seg))
            s_mean = float(np.mean(seg)) if n else 0.0
            kappa = self.kappa0 + n
            mu = (self.kappa0 * self.mu0 + n * s_mean) / kappa
            log_predictive[i] = self._predictive_log_prob(x, mu, kappa)

        from scipy.special import logsumexp
        with np.errstate(divide="ignore"):
            log_joint = np.log(self.R) + log_predictive
        log_growth = log_joint + np.log1p(-self.hazard)
        log_change = logsumexp(log_joint) + np.log(self.hazard)
        log_posterior = np.r_[log_change, log_growth]
        new_R = np.exp(log_posterior - logsumexp(log_posterior))
        self.R = new_R
        self.run_lengths = [0] + [r + 1 for r in self.run_lengths]
        self._observed.append(float(x))
        return float(new_R[0])

    def process(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Онлайн-прогон ряда.

        Сигнал смены: P(смена произошла за последние K шагов) > threshold_prob,
        либо MAP run length = 0. Кулдаун предотвращает серии срабатываний после
        одного шока. Возвращает (сигналы смен, вероятности r=0).
        """
        self.reset()
        signals = []
        probs = []
        warmup = 4
        short_run = 3
        cooldown = 4
        last_signal = -10**9
        for t, x in enumerate(np.asarray(y, dtype=float)):
            self.update(x)
            recent = float(self.R[: short_run + 1].sum())
            probs.append(recent)
            fresh = int(np.argmax(self.R)) == 0
            ready = (t - last_signal) >= cooldown
            if t >= warmup and ready and (fresh or recent > self.threshold_prob):
                signals.append(t)
                last_signal = t
                # перезапускаем детектор для чистоты следующего CP
                self.R = np.array([1.0])
                self.run_lengths = [0]
                self._observed = []
        return np.array(signals, dtype=int), np.array(probs, dtype=float)


def detect_bocpd(y: np.ndarray, hazard: float = 0.05, prior_scale: float = 1.0,
                 threshold_prob: float = 0.5) -> np.ndarray:
    det = BOCPD(hazard=hazard, mu0=0.0, kappa0=prior_scale,
                sigma2=1.0, threshold_prob=threshold_prob)
    signals, _ = det.process(y)
    return signals


# ---------------------------------------------------------------- обёртка
def run_detector(name: str, y: np.ndarray, cfg: dict) -> np.ndarray:
    """Запускает детектор по имени с параметрами из конфига changepoint.yaml."""
    c = cfg["methods"].get(name, {})
    y = np.asarray(y, dtype=float)
    if name == "cusum":
        return detect_cusum(y, c.get("threshold_factor", 3.0), c.get("drift", 0.0),
                            c.get("cooldown", 3))
    if name == "peltt":
        return detect_peltt(y, c.get("model", "rbf"), c.get("pen", 10.0))
    if name == "binseg":
        return detect_binseg(y, c.get("model", "rbf"), c.get("pen", 15.0))
    if name == "bocpd":
        return detect_bocpd(y, c.get("hazard", 0.02), c.get("prior_scale", 3.0),
                            c.get("threshold_prob", 0.3))
    raise ValueError(f"Неизвестный детектор: {name}")
