# -*- coding: utf-8 -*-
"""Синтетический бенчмарк детекторов структурных изменений.

Поскольку в данных СберИндекса нет разметки шоков, оцениваем детекторы на
рядах с инжектированными шоками известной формы (сдвиг уровня, изменение тренда,
рост волатильности, разовый всплеск). Метрики: precision / recall / F1 и
средняя задержка обнаружения (detection delay) — насколько позже (или раньше)
детектор нашёл точку изменения.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.changepoint.detectors import run_detector

logger = logging.getLogger("sber_index.changepoint")


def generate_series(length: int = 60, seed: int = 42, shocks: list[dict] | None = None) -> tuple[np.ndarray, list[int]]:
    """Генерирует ряд с одним случайным шоком (или без шока) в середине."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, length).astype(float)
    cps = []
    if rng.random() < 0.7:  # 70% серий содержат шок
        if length < 16:
            raise ValueError("benchmark length must be >= 16")
        cp = int(rng.integers(max(4, length // 3), length - 4))
        spec = (shocks or [{"type": "level", "magnitude": [0.5, 2.0]},
                          {"type": "trend", "magnitude": [0.1, 0.5]},
                          {"type": "variance", "magnitude": [1.5, 3.0]},
                          {"type": "spike", "magnitude": [2.0, 4.0]}])[int(rng.integers(len(shocks) if shocks else 4))]
        kind = spec["type"]
        mag = float(rng.uniform(*spec["magnitude"]))
        if kind == "level":
            y[cp:] += mag
        elif kind == "trend":
            ramp = mag * np.arange(length - cp)
            y[cp:] += ramp
        elif kind == "variance":
            y[cp:] = y[cp:] * mag
        else:  # spike
            y[cp] += mag
        cps.append(cp)
    return y, cps


def match_predictions(pred_cps: np.ndarray, true_cps: list[int],
                      tol: int = 3, allow_early: bool = True) -> tuple[int, int, int, list[int]]:
    """Сопоставление предсказанных точек с истинными в окне tol.

    Возвращает (tp, fp, fn, delays) — delays для верно найденных точек.
    """
    preds = sorted(int(p) for p in pred_cps)
    trues = sorted(true_cps)
    matched_true = set()
    matched_pred = set()
    delays = []
    for j, p in enumerate(preds):
        best = None
        best_d = np.inf
        for i, c in enumerate(trues):
            if i in matched_true:
                continue
            d = abs(p - c)
            if d <= tol and d < best_d and (allow_early or p >= c):
                best_d = d
                best = i
        if best is not None:
            matched_true.add(best)
            matched_pred.add(j)
            delays.append(p - trues[best])
    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(trues) - tp
    return tp, fp, fn, delays


def evaluate_series(y: np.ndarray, true_cps: list[int], detector: str,
                    cfg: dict, tol: int = 3) -> dict:
    pred = run_detector(detector, y, cfg)
    tp, fp, fn, delays = match_predictions(pred, true_cps, tol=tol,
                                          allow_early=detector not in ("cusum", "bocpd"))
    n_true = max(len(true_cps), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / n_true
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    delay = float(np.mean(delays)) if delays else float("nan")
    online = detector in ("cusum", "bocpd")
    causal_delays = [d for d in delays if d >= 0] if online else []
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "delay": delay, "n_pred": len(pred),
            "online": online, "detection_delay": float(np.mean(causal_delays)) if causal_delays else np.nan}


def run_benchmark(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прогоняет все включённые детекторы на синтетическом корпусе."""
    b = cfg["benchmark"]
    n_series = b.get("n_series", 500)
    length = b.get("length", 60)
    seed = b.get("seed", 42)
    tol = b.get("match_tolerance", 3)
    methods = [m for m, c in cfg["methods"].items() if c.get("enable", False)]

    rows = []
    for i in range(n_series):
        y, cps = generate_series(length, seed=seed + i, shocks=b.get("shocks"))
        for det in methods:
            res = evaluate_series(y, cps, det, cfg, tol=tol)
            res.update({"series": i, "detector": det, "n_true": len(cps)})
            rows.append(res)

    df = pd.DataFrame(rows)
    summary = (df.groupby("detector")[["precision", "recall", "f1"]]
               .mean().reset_index())
    summary["avg_delay"] = df.groupby("detector")["delay"].mean().values
    summary["detection_delay"] = df.groupby("detector")["detection_delay"].mean().values
    totals = df.groupby("detector")[["tp", "fp", "fn"]].sum()
    summary["micro_precision"] = (totals.tp / (totals.tp + totals.fp).clip(lower=1)).to_numpy()
    summary["micro_recall"] = (totals.tp / (totals.tp + totals.fn).clip(lower=1)).to_numpy()
    summary["micro_f1"] = (2 * totals.tp / (2 * totals.tp + totals.fp + totals.fn).clip(lower=1)).to_numpy()
    clean = df[df.n_true == 0].groupby("detector")
    summary["false_alarm_series_rate"] = summary.detector.map(clean.n_pred.apply(lambda s: (s > 0).mean()))
    summary["false_alarms_per_100_months"] = summary.detector.map(clean.n_pred.mean() * 100 / length)
    rng = np.random.default_rng(seed + 100000)
    for detector, group in df.groupby("detector"):
        counts = group[["tp", "fp", "fn"]].to_numpy()
        sampled = counts[rng.integers(len(counts), size=(1000, len(counts)))].sum(axis=1)
        f1 = 2 * sampled[:, 0] / np.maximum(2 * sampled[:, 0] + sampled[:, 1] + sampled[:, 2], 1)
        summary.loc[summary.detector == detector, "f1_ci_low"] = np.quantile(f1, .025)
        summary.loc[summary.detector == detector, "f1_ci_high"] = np.quantile(f1, .975)
    logger.info("Бенчмарк шоков завершён: %d серий, методы %s",
                n_series, methods)
    return df, summary
