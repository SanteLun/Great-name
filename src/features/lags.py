# -*- coding: utf-8 -*-
"""Лаговые признаки: лаги, скользящие статистики, разности."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _calendar_lag(panel, months):
    if months < 1:
        raise ValueError("Lag must be positive")
    keys = ["territory_id", "category", "date"]
    if panel.duplicated(keys).any():
        raise ValueError("Duplicate monthly series keys")
    dates = pd.PeriodIndex(panel.date, freq="M")
    lookup = panel.assign(date=dates).set_index(keys).consumption
    previous = pd.MultiIndex.from_arrays([panel.territory_id, panel.category, dates - months], names=keys)
    return pd.Series(lookup.reindex(previous).to_numpy(), index=panel.index)


def build_lags(panel: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Сдвиговые признаки value_{t-k} внутри (territory_id, category)."""
    out = panel.copy()
    out = out.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)
    for k in lags:
        out[f"lag_{k}"] = _calendar_lag(out, k)
    return out


def build_rolling(panel: pd.DataFrame, windows: list[int],
                  features: list[str]) -> pd.DataFrame:
    """Скользящие статистики на окнах (только прошлое: shift(1) перед агрегацией)."""
    out = panel.copy()
    out = out.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)
    if out.duplicated(["territory_id", "category", "date"]).any():
        raise ValueError("Duplicate monthly series keys")
    unknown = set(features) - {"mean", "std", "min", "max", "trend"}
    if unknown:
        raise ValueError(f"Unknown rolling statistics: {sorted(unknown)}")
    for w in windows:
        if w < 1:
            raise ValueError("Rolling window must be positive")
        for feature in features:
            out[f"roll_{w}_{feature}"] = np.nan
    # Reindex only inside each series: an absent month must occupy its calendar
    # position, rather than pulling an older observation into the window.
    for _, group in out.groupby(["territory_id", "category"], observed=True):
        dates = pd.PeriodIndex(group.date, freq="M")
        values = pd.Series(group.consumption.to_numpy(), index=dates)
        values = values.reindex(pd.period_range(dates.min(), dates.max(), freq="M"))
        for w in windows:
            rolling = values.shift(1).rolling(w, min_periods=max(1, w // 2))
            for feature in features:
                if feature == "trend":
                    def slope(x):
                        valid = np.isfinite(x)
                        return float(np.polyfit(np.arange(len(x))[valid], x[valid], 1)[0]) if valid.sum() >= 2 else np.nan
                    statistic = rolling.apply(slope, raw=True)
                else:
                    statistic = getattr(rolling, feature)()
                out.loc[group.index, f"roll_{w}_{feature}"] = statistic.reindex(dates).to_numpy()
    return out


def build_diffs(panel: pd.DataFrame, orders: list[int]) -> pd.DataFrame:
    """Разности порядка orders (месячные и годовые)."""
    out = panel.copy()
    for o in orders:
        previous = _calendar_lag(out, o)
        out[f"diff_{o}"] = out.consumption - previous
        out[f"diff_{o}_rel"] = (out.consumption / previous.replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
    return out


def build_all(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Собирает все лаговые признаки по конфигу features.yaml."""
    out = panel.copy()
    if cfg.get("lags"):
        out = build_lags(out, cfg["lags"])
    if cfg.get("rolling", {}).get("windows"):
        out = build_rolling(out, cfg["rolling"]["windows"],
                            cfg["rolling"].get("features", ["mean"]))
    if cfg.get("diffs", {}).get("enable"):
        out = build_diffs(out, cfg["diffs"].get("orders", [1]))
    return out
