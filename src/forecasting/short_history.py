"""Regularized local trend and annual Fourier model for short monthly series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_short_history(features_df, cutoff, horizon, params=None):
    params = params or {}
    rows = []
    columns = ["date", "territory_id", "category", "consumption"]
    if "observed" in features_df:
        columns.append("observed")
    history = features_df.loc[features_df.date <= cutoff, columns]
    for (tid, cat), group in history.groupby(
            ["territory_id", "category"], observed=True):
        group = group.sort_values("date")
        if "observed" in group:
            group = group[group.observed]
        group = group.dropna(subset=["consumption"])
        if group.empty:
            continue
        t = np.array([d.ordinal - cutoff.ordinal for d in group.date], dtype=float)
        y = group.consumption.to_numpy(dtype=float)
        def design(times):
            return np.column_stack([np.ones(len(times)), times / 12,
                                    np.sin(2 * np.pi * times / 12),
                                    np.cos(2 * np.pi * times / 12),
                                    np.sin(4 * np.pi * times / 12),
                                    np.cos(4 * np.pi * times / 12)])
        x = design(t)
        penalty = np.diag([0., params.get("trend_penalty", 12.), 2., 2., 4., 4.])
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        future = design(np.array([float(horizon)]))
        damping = params.get("damping", .9)
        future[0, 1] = sum(damping ** k for k in range(1, horizon + 1)) / 12
        rows.append({"territory_id": tid, "category": cat,
                     "date": cutoff + horizon, "predicted": float((future @ beta)[0])})
    return pd.DataFrame(rows)
