# -*- coding: utf-8 -*-
"""Наивные бейзлайны: persistence и seasonal naive.

Единый интерфейс для walk-forward валидации:
    predict(features_df, cutoff, horizon) -> DataFrame(territory_id, category, date, predicted)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("sber_index.forecasting")


def predict_naive(features_df: pd.DataFrame, cutoff: pd.Period,
                  horizon: int, mode: str = "seasonal",
                  value_col: str = "consumption") -> pd.DataFrame:
    """Наивный прогноз из точки cutoff на horizon месяцев вперёд.

    mode='last': прогноз = последнее известное значение серии.
    mode='seasonal': прогноз на месяц t = значение того же месяца год назад (t-12).

    features_df: полная панель с колонками date, territory_id, category, value_col.
    Возвращает: колонки territory_id, category, date, predicted.
    """
    if mode not in ("last", "seasonal") or horizon < 1:
        raise ValueError("Positive horizon and mode last/seasonal required")
    hist = features_df[features_df.date <= cutoff]
    if hist.empty:
        return pd.DataFrame(columns=["territory_id", "category", "date", "predicted"])

    keys = ["territory_id", "category"]
    monthly_keys = keys + ["date"]
    if hist.duplicated(monthly_keys).any():
        raise ValueError("Duplicate monthly series keys")
    hist = hist.sort_values(monthly_keys)
    last = hist.groupby(keys, observed=True).tail(1)[keys + [value_col]].rename(columns={value_col: "last_value"})
    previous = hist.groupby(keys, observed=True).nth(-13)[keys + [value_col]].rename(columns={value_col: "fallback_value"})
    last = last.merge(previous, on=keys, how="left", validate="one_to_one")
    seasonal = hist[monthly_keys + [value_col]].rename(columns={value_col: "seasonal_value"}).copy()
    seasonal["date"] += 12
    parts = []
    for t in range(1, horizon + 1):
        part = last.assign(date=cutoff + t)
        if mode == "seasonal":
            part = part.merge(seasonal, on=monthly_keys, how="left", indicator=True, validate="one_to_one")
            backup = part.fallback_value.combine_first(part.last_value)
            part["predicted"] = part.seasonal_value.where(part._merge == "both", backup)
        else:
            part["predicted"] = part.last_value
        parts.append(part[monthly_keys + ["predicted"]])
    res = pd.concat(parts, ignore_index=True).sort_values(monthly_keys).reset_index(drop=True)
    logger.info("naive(%s) h=%d cutoff=%s: %d прогнозов", mode, horizon, cutoff, len(res))
    return res
