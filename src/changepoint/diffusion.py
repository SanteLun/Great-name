# -*- coding: utf-8 -*-
"""Пространственная диффузия шоков между МО.

Идея: шок в одном муниципалитете распространяется на соседей по графу
транспортных связей (highway/railway). Если у k ближайших соседей детектор
сработал за последние window месяцев, мы заранее повышаем внимание к данному МО
(пространственный триггер) — это работает как «раннее предупреждение» и снижает
порог срабатывания у потенциально «заражённого» МО.

Функции:
- neighbor_signal_score: число сигналов у соседей по окну месяцев
- spatial_boost: результирующий скор с учётом собственных сигналов детектора
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.cleaner import neighbor_distance_matrix

logger = logging.getLogger("sber_index.changepoint")


def build_neighbor_list(conn: pd.DataFrame, k: int = 8,
                        connection_type: str = "highway") -> dict[int, list[int]]:
    """k ближайших соседей каждого МО по транспортной сети."""
    dist = neighbor_distance_matrix(conn, connection_type)
    out = {}
    for i in dist.index:
        neigh = dist.loc[i].dropna().sort_values().head(k)
        out[int(i)] = [int(j) for j in neigh.index if int(j) != int(i)]
    return out


def neighbor_signal_score(signals: pd.DataFrame, neighbors: dict[int, list[int]],
                          window: int = 3) -> pd.DataFrame:
    """Число сигналов у соседей за последние window месяцев.

    signals: DataFrame(territory_id, date, signal) — 1 если детектор сработал.
    Возвращает DataFrame(territory_id, date, neighbor_shocks).
    """
    if signals.empty:
        return pd.DataFrame(columns=["territory_id", "date", "neighbor_shocks"])

    if "category" in signals:
        parts = []
        for category, group in signals.groupby("category", observed=True):
            part = neighbor_signal_score(group.drop(columns="category"), neighbors, window)
            part["category"] = category
            parts.append(part)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # расширяем до полной сетки месяцев на территорию для корректного окна
    rows = []
    signals = signals.copy()
    signals["date"] = pd.PeriodIndex(signals["date"], freq="M")
    if "signal" in signals:
        signals = signals[signals.signal > 0]
    if signals.empty:
        return pd.DataFrame(columns=["territory_id", "date", "neighbor_shocks"])
    for tid in neighbors:
        tid = int(tid)
        if tid not in neighbors:
            continue
        dates = sorted(signals.date.unique())
        if not dates:
            continue
        month_min = pd.Period(min(dates), freq="M") - window
        month_max = pd.Period(max(dates), freq="M") + 1
        months = list(pd.period_range(month_min, month_max, freq="M"))
        # карта сигналов соседей: {neighbor: set(months)}
        neigh_sig = {}
        for nb in neighbors[tid]:
            nb_rows = signals[signals.territory_id == nb]
            neigh_sig[nb] = set(nb_rows.date)
        for m in months:
            cnt = sum(1 for nb in neighbors[tid]
                      if any(0 < m.ordinal - d.ordinal <= window
                             for d in neigh_sig.get(nb, ())))
            rows.append({"territory_id": tid, "date": m,
                         "neighbor_shocks": cnt})
    return pd.DataFrame(rows)


def spatial_trigger(signals: pd.DataFrame, conn: pd.DataFrame,
                    k: int = 8, window: int = 3,
                    min_neighbors: int = 2) -> pd.DataFrame:
    """Пространственный триггер: срабатывание, если >= min_neighbors соседей
    имели сигнал за последние window месяцев (собственный сигнал не нужен)."""
    neighbors = build_neighbor_list(conn, k=k)
    score = neighbor_signal_score(signals, neighbors, window=window)
    if score.empty:
        return score
    score["spatial_trigger"] = (score["neighbor_shocks"] >= min_neighbors).astype(int)
    return score
