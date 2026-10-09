# -*- coding: utf-8 -*-
"""Пространственные признаки: диффузия расходов между соседними МО.

Идея: шок в одном муниципалитете распространяется на соседей (транспортные связи,
поездки за покупками, общие экономические условия). Признак «пространственный лаг» —
взвешенная сумма расходов k ближайших соседей по графу connection с весами 1/d^decay,
взятая с задержкой (используются ТОЛЬКО прошлые значения соседей, чтобы не было
утечки будущего при прогнозировании).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.cleaner import neighbor_distance_matrix

logger = logging.getLogger("sber_index.features")


def build_weight_matrix(conn: pd.DataFrame, connection_type: str,
                        k: int = 8, decay: float = 1.0) -> pd.DataFrame:
    """Матрица весов W (территории x территории): 1/distance**decay для k соседей.
    Строки нормированы на единицу. Территории без связей — нулевые строки."""
    dist = neighbor_distance_matrix(conn, connection_type)
    weights = 1.0 / dist.where(dist > 0).pow(decay)
    for tid in weights.index.intersection(weights.columns):
        weights.loc[tid, tid] = np.nan
    # оставляем k ближайших соседей (наибольшие веса)
    for i in weights.index:
        keep = weights.loc[i].nlargest(k).dropna().index
        weights.loc[i, ~weights.columns.isin(keep)] = 0.0
    weights = weights.fillna(0.0)
    row_sums = weights.sum(axis=1).replace(0, 1.0)
    weights = weights.div(row_sums, axis=0)
    return weights


def spatial_lag_values(panel: pd.DataFrame, weights: pd.DataFrame,
                       category: str, lag_months: list[int],
                       value_col: str = "consumption") -> pd.DataFrame:
    """Для одной категории считает пространственные лаги соседей в месяцы t-k.

    Возвращает DataFrame (territory_id, date) c колонками sp_lag_{type}_{k}.
    """
    sub = panel[panel.category == category].copy()
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot(index="date", columns="territory_id", values=value_col)
    wide = wide.reindex(columns=weights.index).sort_index()
    W = weights.to_numpy(dtype=float)          # n x n
    X = wide.to_numpy(dtype=float)             # T x n
    dates = wide.index

    # взвешенная сумма по соседям: sp[t, i] = sum_j W[i, j] * X[t-k, j]  =>  sp = X_k @ W.T
    # вычисляем сразу для всех лагов и формируем длинный формат (territory, date)
    sp_by_lag: dict[int, np.ndarray] = {}
    for k in lag_months:
        X_k = np.full_like(X, np.nan)
        if k <= 0:
            raise ValueError("Пространственный лаг должен быть положительным")
        if k < X.shape[0]:
            X_k[k:, :] = X[:-k, :]
        observed = np.isfinite(X_k)
        numerator = np.where(observed, X_k, 0.0) @ W.T
        denominator = observed.astype(float) @ W.T
        sp_by_lag[k] = np.divide(numerator, denominator,
                                out=np.full_like(numerator, np.nan),
                                where=denominator > 0)

    out_long = []
    for j, t_id in enumerate(weights.index):
        row = pd.DataFrame({"date": dates})
        row["territory_id"] = t_id
        for k in lag_months:
            row[f"sp_lag_{k}"] = sp_by_lag[k][:, j]
        out_long.append(row)
    res = pd.concat(out_long, ignore_index=True)
    return res[["territory_id", "date"] + [f"sp_lag_{k}" for k in lag_months]]


def add_spatial_features(panel: pd.DataFrame, conn: pd.DataFrame,
                         cfg: dict, value_col: str = "consumption") -> pd.DataFrame:
    """Добавляет пространственные признаки в панель для всех категорий."""
    if not cfg.get("enable", True):
        return panel.copy()

    k = cfg.get("neighbor_k", 8)
    decay = cfg.get("decay_exponent", 1.0)
    lag_months = cfg.get("lag_months", [1, 2, 3])
    types = cfg.get("connection_types", ["highway", "railway"])
    result = panel.copy()

    for ctype in types:
        weights = build_weight_matrix(conn, ctype, k=k, decay=decay)
        logger.info("spatial %s: матрица весов %s", ctype, weights.shape)
        per_type = []
        for cat in panel.category.unique():
            sl = spatial_lag_values(result, weights, cat, lag_months, value_col)
            if not sl.empty:
                sl.columns = ["territory_id", "date"] + \
                             [f"sp_lag_{ctype}_{k_}" for k_ in lag_months]
                sl["category"] = cat
                per_type.append(sl)
        if per_type:
            # каждый тип связи мержим отдельно, чтобы не плодить дубли ключей
            feats = pd.concat(per_type, ignore_index=True)
            result = result.merge(feats, on=["territory_id", "date", "category"],
                                  how="left")
    return result
