# -*- coding: utf-8 -*-
"""Метрики качества прогноза: MAE (обязательная), RMSE, MAPE, R2."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


METRICS = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "r2": r2,
}


def compute_metrics(y_true, y_pred, names: list[str] | None = None) -> dict[str, float]:
    """Считает все запрошенные метрики; возвращает словарь."""
    names = names or ["mae", "rmse", "mape", "r2"]
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("Actuals and predictions must have equal lengths")
    unknown = set(names) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require finite actuals and predictions")
    if len(y_true) == 0:
        return {n: float("nan") for n in names}
    return {n: METRICS[n](y_true, y_pred) for n in names}


def aggregate_metrics(records: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует метрики из длинного формата: (model, horizon, metric) -> value.
    Возвращает сводную таблицу model x horizon с MAE/R2 и пр."""
    out = records.copy()
    # средние по всем сериям для каждой (model, horizon, metric)
    summary = (out.groupby(["model", "horizon", "metric"])["value"]
               .agg(["mean", "std"]).reset_index())
    return summary
