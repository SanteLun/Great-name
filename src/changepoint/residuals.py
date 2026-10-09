# -*- coding: utf-8 -*-
"""Остатки прогнозной модели как объект анализа для детекторов шоков.

Логика решения: шок = систематическое отклонение факта от того, что «должно было
быть» по модели. Поэтому детекторы запускаются на стандартизированных остатках
(факт - прогноз), а не на сыром ряде: так убираются тренд и сезонность.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("sber_index.changepoint")


def standardized_residuals(actual: np.ndarray, predicted: np.ndarray,
                           std_window: int = 6) -> pd.Series:
    """Стандартизированные остатки r = (actual - predicted) / std(rolling).

    std_window — окно скользящего стандартного отклонения остатков.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    resid = a - p
    s = pd.Series(resid)
    history = s.shift(1)
    roll_std = history.rolling(std_window, min_periods=3).std()
    center = history.rolling(std_window, min_periods=3).mean()
    z = (s - center) / roll_std.clip(lower=1e-6)
    return z


def residuals_from_forecast(forecast_df: pd.DataFrame, features_df: pd.DataFrame,
                            std_window: int = 6) -> pd.DataFrame:
    """Остатки прогнозов по всем сериям из таблицы прогноза.

    forecast_df: territory_id, category, date, predicted (в исходном масштабе).
    features_df: панель с фактическими значениями (consumption в исходном масштабе).
    """
    fact = features_df[["date", "territory_id", "category", "consumption"]].copy()
    fc = forecast_df.copy()
    if str(fc["date"].dtype) not in ("period[M]", "<Period[M]>"):
        try:
            fc["date"] = pd.PeriodIndex(fc["date"], freq="M")
        except Exception:  # noqa: BLE001
            fc["date"] = pd.to_datetime(fc["date"]).dt.to_period("M")
    if str(fact["date"].dtype) != str(fc["date"].dtype):
        try:
            fact["date"] = pd.PeriodIndex(fact["date"], freq="M")
        except Exception:  # noqa: BLE001
            pass
    merged = fc.merge(fact, on=["date", "territory_id", "category"],
                      how="inner")
    if merged.empty:
        return merged
    merged["residual"] = merged["consumption"] - merged["predicted"]
    groups = ["territory_id", "category"] + [c for c in ["model", "horizon"] if c in merged]
    merged = merged.sort_values(groups + ["date"]).reset_index(drop=True)
    if merged.duplicated(groups + ["date"]).any():
        raise ValueError("Повторные месяцы остатков: нужны прогнозы точного горизонта с cutoff")
    merged["zscore"] = np.nan
    for _, group in merged.groupby(groups, observed=True):
        monthly = group.set_index("date").residual
        months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
        dense = monthly.reindex(months)
        scores = standardized_residuals(dense.to_numpy(), np.zeros(len(dense)), std_window)
        scores.index = months
        merged.loc[group.index, "zscore"] = scores.reindex(group.date).to_numpy()
    return merged
