# -*- coding: utf-8 -*-
"""Календарные признаки (сезонность, праздники, рабочие дни)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные признаки по колонке date (Period 'M')."""
    out = panel.copy()
    dt = out["date"].dt.to_timestamp()
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["year"] = dt.dt.year
    out["day_of_month"] = dt.dt.day
    out["is_eoy"] = (out["month"] == 12).astype(int)

    # число дней в месяце
    out["days_in_month"] = dt.dt.days_in_month
    # индикатор января (сезонный минимум расходов после праздников)
    out["is_jan"] = (out["month"] == 1).astype(int)
    # месячный тренд-счётчик внутри года
    out["month_cycle"] = dt.dt.month / 12.0
    return out


def russian_working_days(panel: pd.DataFrame,
                         holidays: list[str] | None = None) -> pd.DataFrame:
    """Число рабочих и праздничных дней в месяце (РФ, фиксированные праздники).
    Приближённый календарь: переносы выходных и рабочие субботы не учитываются.
    """
    out = panel.copy()
    fixed_holidays = holidays or [
        "01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08",
        "02-23", "03-08", "05-01", "05-09", "06-12", "11-04",
    ]
    holiday_set = set(fixed_holidays)

    def _counts(d: pd.Period) -> tuple[int, int]:
        work, hol = 0, 0
        for ts in pd.date_range(d.start_time, d.end_time, freq="D"):
            wd = ts.weekday()  # 0=Mon .. 6=Sun
            key = ts.strftime("%m-%d")
            is_weekend = wd >= 5
            is_holiday = key in holiday_set
            if is_holiday:
                hol += 1
            if is_weekend or is_holiday:
                continue
            work += 1
        return work, hol

    # считаем по уникальным месяцам, объединяем по строковому ключу YYYY-MM
    uniq = out[["date"]].drop_duplicates()
    uniq["_key"] = uniq["date"].astype(str)
    uniq[["working_days", "holidays"]] = uniq["date"].map(_counts).apply(pd.Series)
    out["_key"] = out["date"].astype(str)
    out = out.drop(columns=["working_days", "holidays"], errors="ignore")
    out = out.merge(uniq[["_key", "working_days", "holidays"]], on="_key", how="left")
    out = out.drop(columns=["_key"])
    return out


def build_calendar(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Собирает календарные признаки согласно конфигу features.yaml."""
    out = panel.copy()
    if cfg.get("month", True):
        out["month"] = out["date"].dt.to_timestamp().dt.month
    if cfg.get("quarter", True):
        out["quarter"] = out["date"].dt.to_timestamp().dt.quarter
    if cfg.get("year", True):
        out["year"] = out["date"].dt.to_timestamp().dt.year
    if cfg.get("day_of_month", True):
        out["day_of_month"] = out["date"].dt.to_timestamp().dt.day
    if cfg.get("is_eoy", True):
        out["is_eoy"] = (out["date"].dt.to_timestamp().dt.month == 12).astype(int)
    if cfg.get("working_days", False) or cfg.get("holidays", False):
        out = russian_working_days(out)
    return out
