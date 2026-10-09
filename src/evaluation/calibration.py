"""Causal relative-error conformal intervals; no temporal coverage guarantee."""
from __future__ import annotations
import numpy as np
import pandas as pd


def prequential_intervals(predictions, alpha=.2, min_pairs=20):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if min_pairs < 1:
        raise ValueError("min_pairs must be positive")
    out = predictions.copy().reset_index(drop=True)
    for column in ["date", "cutoff"]:
        out[column] = pd.PeriodIndex(out[column], freq="M")
    out["calibrated_lower"], out["calibrated_upper"] = np.nan, np.nan
    out["calibration_n"] = 0
    out["interval_nominal_coverage"] = 1 - alpha
    if "actual" not in out:
        out["actual"] = np.nan
    for _, group in out.groupby(["model", "horizon", "category"], observed=True):
        scale = np.maximum(group.predicted.to_numpy(), 1.)
        scores = np.abs(group.actual.to_numpy() - group.predicted.to_numpy()) / scale
        for cutoff in sorted(group.cutoff.unique()):
            known = ((group.date <= cutoff) & (group.cutoff < cutoff)).to_numpy() & np.isfinite(scores)
            current = (group.cutoff == cutoff).to_numpy()
            indices = group.index[current]
            n = int(known.sum())
            out.loc[indices, "calibration_n"] = n
            if n < min_pairs:
                continue
            rank = int(np.ceil((n + 1) * (1 - alpha)))
            # The finite-sample quantile is infinity when the requested rank
            # exceeds n. Do not silently replace it with a narrower interval.
            if rank > n:
                continue
            q = np.sort(scores[known])[rank - 1]
            center = group.predicted.to_numpy()[current]
            margin = q * scale[current]
            out.loc[indices, "calibrated_lower"] = np.maximum(center - margin, 0.)
            out.loc[indices, "calibrated_upper"] = center + margin
    return out


def coverage_summary(calibrated):
    rows = []
    for (model, horizon), group in calibrated.groupby(["model", "horizon"]):
        valid = group.dropna(subset=["actual", "calibrated_lower", "calibrated_upper"])
        if valid.empty:
            continue
        rows.append({"model": model, "horizon": int(horizon), "n_pairs": len(valid),
                     "coverage": ((valid.actual >= valid.calibrated_lower) & (valid.actual <= valid.calibrated_upper)).mean(),
                     "mean_width": (valid.calibrated_upper - valid.calibrated_lower).mean(),
                     "evaluation_fraction": len(valid) / len(group),
                     "nominal_coverage": float(group.interval_nominal_coverage.iloc[0]),
                     "native_coverage_same_pairs": ((valid.actual >= valid.lower) & (valid.actual <= valid.upper)).mean()
                     if "lower" in valid and "upper" in valid and valid.lower.notna().all() and valid.upper.notna().all() else np.nan})
    return pd.DataFrame(rows, columns=["model", "horizon", "n_pairs", "coverage", "mean_width", "evaluation_fraction",
                                       "nominal_coverage", "native_coverage_same_pairs"])
