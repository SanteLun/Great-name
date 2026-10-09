# -*- coding: utf-8 -*-
"""Обработка пропусков и трансформации целевой переменной."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("sber_index.data")


def impute_time(panel: pd.DataFrame, max_gap_months: int = 3) -> pd.DataFrame:
    """Причинное заполнение: перенос последнего факта максимум на max_gap_months.

    Будущие наблюдения не используются даже при подготовке полной панели.
    """
    out = panel.copy().sort_values(["territory_id", "category", "date"])
    if max_gap_months > 0:
        out["consumption"] = out.groupby(["territory_id", "category"])["consumption"].ffill(
            limit=max_gap_months)
    logger.info("impute_time: осталось NaN: %d", int(out.consumption.isna().sum()))
    return out


def neighbor_distance_matrix(conn: pd.DataFrame, connection_type: str | None = None
                            ) -> pd.DataFrame:
    """Матрица расстояний между МО из графа connection.
    Возвращает DataFrame с индексом territory_id_x и колонками territory_id_y."""
    df = conn.copy()
    if connection_type is not None:
        df = df[df.type == connection_type]
    # используем минимальное расстояние highway/railway между парой
    d = (df.groupby(["territory_id_x", "territory_id_y"], as_index=False)["distance"].min())
    mat = d.pivot(index="territory_id_x", columns="territory_id_y", values="distance")
    # симметризуем: в данных x->y в одну сторону, добавляем обратное направление
    full = mat.combine_first(mat.T)
    return full


def spatial_impute(panel: pd.DataFrame, conn: pd.DataFrame,
                   k: int = 5, connection_type: str = "highway") -> pd.DataFrame:
    """Заполнение оставшихся NaN средним расходом k ближайших соседей (по 1/distance)."""
    out = panel.copy()
    missing = out[out.consumption.isna()]
    if missing.empty:
        return out

    dist = neighbor_distance_matrix(conn, connection_type)

    for _, sub in out.groupby(["date", "category"]):
        available = sub.set_index("territory_id")["consumption"]
        for idx, row in sub[sub.consumption.isna()].iterrows():
            tid = row["territory_id"]
            if tid not in dist.index:
                continue
            distances = dist.loc[tid].drop(index=tid, errors="ignore").dropna()
            distances = distances[distances > 0].nsmallest(k)
            values = available.reindex(distances.index)
            valid = values.notna()
            if valid.any():
                out.loc[idx, "consumption"] = np.average(
                    values[valid], weights=1.0 / distances[valid])
    logger.info("spatial_impute: осталось NaN: %d", int(out.consumption.isna().sum()))
    return out


def impute_missing(panel: pd.DataFrame, conn: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Полный пайплайн заполнения пропусков по конфигу data.yaml."""
    method = cfg["imputation"]["method"]
    out = panel.copy()
    if method in ("linear", "forward", "both"):
        out = impute_time(out, max_gap_months=cfg["imputation"]["max_gap_months"])
    if method in ("neighbors", "both") and out.consumption.isna().any():
        out = spatial_impute(out, conn,
                             k=cfg["imputation"]["neighbor_k"],
                             connection_type="highway")
    if out.consumption.isna().any():
        # крайний случай: заполняем медианой по (дата, категория)
        med = (out.groupby(["date", "category"])["consumption"]
               .transform(lambda s: s.median()))
        out["consumption"] = out["consumption"].fillna(med)
        logger.info("fillna_median: осталось NaN: %d", int(out.consumption.isna().sum()))
    return out


# ---------- Трансформации целевой переменной ----------

def apply_transform(series: pd.Series, use_log: bool = True) -> pd.Series:
    if use_log:
        return np.log1p(series.clip(lower=0))
    return series.copy()


def inverse_transform(series: pd.Series, use_log: bool = True) -> pd.Series:
    if use_log:
        return np.expm1(series)
    return series.copy()
