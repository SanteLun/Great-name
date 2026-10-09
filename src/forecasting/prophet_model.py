# -*- coding: utf-8 -*-
"""Бейзлайн Prophet (модель Сбера) с прогнозом на h месяцев по каждой серии."""
from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.config import PROJECT_ROOT

# CmdStan создаёт JSON и вывод Stan во временном каталоге. В ограниченном
# Windows-окружении стандартный TEMP может оказаться недоступен для файлов,
# хотя проверка os.access проходит: tempfile тогда перебирает имена бесконечно.
# Все промежуточные файлы этого вычислительного процесса размещаем в проекте.
STAN_TMP = PROJECT_ROOT / "outputs" / "tmp"
STAN_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(STAN_TMP)
from prophet import Prophet

logger = logging.getLogger("sber_index.forecasting")


def _fit_predict_series(df: pd.DataFrame, horizon: int,
                        params: dict, tid: int, cat: str) -> list[dict]:
    """Обучает Prophet на одной серии и прогнозирует horizon месяцев."""
    # Проверка одним открытием: Windows tempfile иначе может миллионы раз
    # повторять PermissionError при несовместимых правах песочницы и mkdtemp.
    from cmdstanpy import _TMPDIR
    probe = Path(_TMPDIR) / f"write-check-{uuid.uuid4().hex}.json"
    try:
        with probe.open("x", encoding="utf-8") as stream:
            stream.write("{}")
        probe.unlink()
    except PermissionError as exc:
        raise PermissionError("CmdStan TEMP недоступен: запустите вычисление с правами "
                              "на временный каталог (в Codex — разрешите прогон вне песочницы)") from exc
    observed_df = df[df["observed"]] if "observed" in df else df
    hist = observed_df[["date", "consumption"]].copy()
    hist["ds"] = hist["date"].dt.to_timestamp()
    hist = hist.rename(columns={"consumption": "y"})
    hist = hist[["ds", "y"]].dropna()
    if params.get("use_log", True):
        hist["y"] = np.expm1(hist["y"])

    m = Prophet(
        yearly_seasonality=params.get("yearly_seasonality", 4),
        weekly_seasonality=params.get("weekly_seasonality", False),
        daily_seasonality=params.get("daily_seasonality", False),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 5.0),
        interval_width=params.get("interval_width", 0.9),
        growth="linear",
    )
    seed = params.get("seed", 42)
    m.fit(hist, seed=seed)
    gap = df.date.max().ordinal - pd.Period(hist.ds.max(), freq="M").ordinal
    future = m.make_future_dataframe(periods=horizon + gap, freq="MS")
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        fc = m.predict(future)
    finally:
        np.random.set_state(rng_state)
    # берём только прогнозные даты (после конца истории)
    last_ds = hist.ds.max()
    fc = fc[fc.ds > last_ds]
    rows = []
    for _, r in fc.iterrows():
        predicted, lower, upper = [max(float(r[c]), 0.0)
                                   for c in ["yhat", "yhat_lower", "yhat_upper"]]
        if params.get("use_log", True):
            predicted, lower, upper = np.log1p([predicted, lower, upper])
        rows.append({
            "territory_id": tid,
            "category": cat,
            "date": pd.Period(r.ds, freq="M"),
            "predicted": float(predicted),
            "lower": float(lower),
            "upper": float(upper),
        })
    return rows


def predict_prophet(features_df: pd.DataFrame, cutoff: pd.Period,
                    horizon: int, params: dict,
                    subsample: int = 300) -> pd.DataFrame:
    """Prophet-прогноз всех серий из точки cutoff.

    features_df: панель с колонками date, territory_id, category, consumption.
    subsample: число серий (случайная выборка) для экономии времени; 0 = все.

    Известное ограничение: при истории короче min_train_factor*horizon месяцев
    Prophet нестабилен (взрывной линейный тренд на длинных горизонтах) —
    в этом случае модель пропускается (см. configs/forecasting.yaml).
    """
    hist = features_df[features_df.date <= cutoff]
    hist_months = int(hist.date.nunique())
    factor = params.get("min_train_factor", 2.0)
    if hist_months < horizon * factor:
        logger.warning("Prophet пропущен: история %d мес < %d*%.1f (горизонт %d)",
                       hist_months, horizon, factor, horizon)
        return pd.DataFrame(columns=["territory_id", "category", "date",
                                     "predicted", "lower", "upper"])
    keys = (hist.groupby(["territory_id", "category"])
            .size().sort_values(ascending=False).index.to_frame(index=False))
    if subsample and subsample > 0 and len(keys) > subsample:
        keys = keys.sample(subsample, random_state=42)
    logger.info("Prophet: обучение на %d сериях, cutoff=%s, h=%d",
                len(keys), cutoff, horizon)

    all_rows = []
    for tid, cat in keys.itertuples(index=False):
        sub = hist[(hist.territory_id == tid) & (hist.category == cat)]
        if sub.consumption.notna().sum() < 6:
            continue
        try:
            all_rows.extend(_fit_predict_series(sub, horizon, params, tid, cat))
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prophet сбой серии (%s,%s): %s", tid, cat, exc)
            continue

    res = pd.DataFrame(all_rows)
    logger.info("Prophet h=%d cutoff=%s: прогнозов %d", horizon, cutoff, len(res))
    return res
