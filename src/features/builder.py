# -*- coding: utf-8 -*-
"""Сборка полного набора признаков для прогнозирования."""
from __future__ import annotations

import logging
import math

import pandas as pd

from src.features.calendar import build_calendar
from src.features.lags import build_all as build_lag_features
from src.features.macro import add_macro_features, load_macro
from src.features.spatial import add_spatial_features

logger = logging.getLogger("sber_index.features")

FEATURE_COLUMNS = [
    "month", "quarter", "year", "day_of_month", "is_eoy", "is_jan",
    "days_in_month", "month_cycle", "working_days", "holidays",
]


def _add_static(panel: pd.DataFrame, market_access: pd.DataFrame | None) -> pd.DataFrame:
    out = panel.copy()
    if market_access is not None and not market_access.empty:
        ma = market_access[["territory_id", "market_access"]].copy()
        out = out.merge(ma, on="territory_id", how="left")
        # логируем (значения 0..1000)
        out["market_access_log"] = out["market_access"].apply(
            lambda v: float("nan") if pd.isna(v) else math.log1p(v)
        )
    return out


def _add_categorical_codes(panel: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    out = panel.copy()
    cfg = cfg or {"territory_id": True, "category": True}
    if cfg.get("territory_id", True):
        out["territory_id_cat"] = out["territory_id"].astype("category")
    if cfg.get("category", True):
        out["category_cat"] = out["category"].astype("category")
    return out


def build_features(panel: pd.DataFrame, conn: pd.DataFrame,
                   market_access: pd.DataFrame | None,
                   feat_cfg: dict) -> pd.DataFrame:
    """Полный конвейер построения признаков.

    panel: длинная панель с колонками date, territory_id, category, consumption
           (значения уже трансформированы log1p).
    """
    logger.info("Старт построения признаков: %d строк", len(panel))
    out = panel.copy().sort_values(["territory_id", "category", "date"])
    out["current_value"] = out["consumption"]

    # 1. Календарь
    out = build_calendar(out, feat_cfg.get("calendar", {}))

    # 2. Лаги и скользящие
    out = build_lag_features(out, feat_cfg)

    # 3. Пространственные признаки (диффузия от соседей)
    if feat_cfg.get("spatial", {}).get("enable", True):
        out = add_spatial_features(out, conn, feat_cfg["spatial"])

    # 3.1 Макроэкономические признаки (официальные публичные данные)
    if feat_cfg.get("macro", {}).get("enable", True):
        macro = load_macro()
        out = add_macro_features(out, macro,
                                 lags=feat_cfg["macro"].get("lags", [1, 3]))

    # 4. Статические признаки
    if feat_cfg.get("spatial", {}).get("use_market_access", True):
        out = _add_static(out, market_access)

    # 5. Категориальные коды для ML-моделей
    if feat_cfg.get("categorical", {}).get("territory_id", True) or \
       feat_cfg.get("categorical", {}).get("category", True):
        out = _add_categorical_codes(out, feat_cfg.get("categorical", {}))

    # порядок колонок: идентификаторы, целевая, признаки
    id_cols = ["date", "territory_id", "category", "consumption"]
    feat_cols = [c for c in out.columns if c not in id_cols]
    out = out[id_cols + sorted(feat_cols)]
    logger.info("Признаки построены: %d строк, %d признаков", len(out), len(feat_cols))
    return out
