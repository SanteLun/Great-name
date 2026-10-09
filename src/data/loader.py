# -*- coding: utf-8 -*-
"""Загрузка и подготовка данных СберИндекса."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("sber_index.data")

DATE_FORMAT = "%Y-%m"


def load_consumption(path: str | Path) -> pd.DataFrame:
    """Загружает consumption.parquet, приводит date к месячному Period."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT).dt.to_period("M")
    df = df.rename(columns={"value": "consumption"})
    logger.info("consumption: %s строк, диапазон %s..%s, МО %d, категорий %d",
                len(df), df.date.min(), df.date.max(), df.territory_id.nunique(),
                df.category.nunique())
    return df


def load_market_access(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("market_access: %d МО", len(df))
    return df


def load_connection(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("connection: %d рёбер, узлов %d, типы %s",
                len(df), len(set(df.territory_id_x) | set(df.territory_id_y)),
                sorted(df.type.unique()))
    return df


def load_all(data_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает все три датасета по конфигу."""
    p = data_cfg["paths"]
    cons = load_consumption(p["consumption"])
    ma = load_market_access(p["market_access"])
    conn = load_connection(p["connection"])
    return cons, ma, conn


def filter_series(cons: pd.DataFrame, categories: list[str] | None = None,
                  territory_ids: list[int] | None = None) -> pd.DataFrame:
    """Фильтрация по категориям и территориям."""
    out = cons.copy()
    if categories is not None:
        out = out[out.category.isin(categories)]
    if territory_ids is not None:
        out = out[out.territory_id.isin(territory_ids)]
    return out


def complete_panel(cons: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Восстанавливает полную сетку (date x territory x category).
    Для отсутствующих пар value = NaN (далее — интерполяция)."""
    dates = pd.period_range(cons.date.min(), cons.date.max(), freq=freq)
    territories = cons.territory_id.unique()
    categories = cons.category.unique()
    index = pd.MultiIndex.from_product([dates, territories, categories],
                                       names=["date", "territory_id", "category"])
    full = cons.set_index(["date", "territory_id", "category"]).reindex(index).reset_index()
    return full


def pivot_wide(panel: pd.DataFrame, category: str) -> pd.DataFrame:
    """Широкий формат: строки = даты, столбцы = territory_id, для одной категории."""
    sub = panel[panel.category == category].copy()
    wide = sub.pivot(index="date", columns="territory_id", values="consumption")
    wide = wide.sort_index()
    return wide


def to_long(wide: pd.DataFrame, category: str, freq: str = "M") -> pd.DataFrame:
    """Обратное преобразование широкого формата в длинный."""
    long = wide.stack().reset_index()
    long.columns = ["date", "territory_id", "consumption"]
    long["category"] = category
    return long.sort_values(["territory_id", "date"]).reset_index(drop=True)