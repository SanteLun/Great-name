"""Forecast integrity checks and comparisons resistant to coverage differences."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["territory_id", "category", "cutoff", "date", "horizon"]


def validate_forecasts(results):
    required = set(KEYS + ["model", "actual", "predicted"])
    if required - set(results):
        raise ValueError(f"Missing columns: {sorted(required - set(results))}")
    out = results.copy()
    for col in ["date", "cutoff"]:
        out[col] = pd.PeriodIndex(out[col], freq="M")
    if out.empty:
        raise ValueError("No forecast results")
    if out[KEYS + ["model"]].isna().any().any():
        raise ValueError("Missing forecast identifiers")
    if out.duplicated(KEYS + ["model"]).any():
        raise ValueError("Duplicate forecast keys")
    if ((out.horizon < 1) | (out.horizon % 1 != 0)).any():
        raise ValueError("Horizons must be positive integer months")
    if (out.date != out.cutoff + out.horizon.astype(int)).any():
        raise ValueError("Forecast dates do not match exact horizons")
    if not np.isfinite(out[["actual", "predicted"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite actuals or predictions")
    if (out[["actual", "predicted"]] < 0).any().any():
        raise ValueError("Negative consumption")
    facts = out.groupby(KEYS, observed=True).actual.agg(["min", "max"])
    if not np.allclose(facts["min"], facts["max"], rtol=1e-10, atol=1e-8):
        raise ValueError("Models disagree about actual consumption on the same keys")
    return out


def coverage_table(results):
    rows = []
    for h, group in results.groupby("horizon"):
        total = len(group[KEYS].drop_duplicates())
        for model, part in group.groupby("model"):
            errors = part.assign(error=abs(part.actual - part.predicted))
            series_mae = errors.groupby(["territory_id", "category"], observed=True).error.mean()
            fallback = part.get("fallback", pd.Series(index=part.index, dtype=object)).notna()
            rows.append({"model": model, "horizon": int(h), "n_pairs": len(part),
                         "coverage_fraction": len(part) / total,
                         "n_series": len(series_mae), "n_origins": part.cutoff.nunique(),
                         "macro_series_mae": series_mae.mean(), "median_series_mae": series_mae.median(),
                         "fallback_fraction": fallback.mean()})
    return pd.DataFrame(rows)


def pairwise_skill(results, baseline="prophet"):
    rows = []
    for h, group in results.groupby("horizon"):
        reference = group[group.model == baseline]
        if reference.empty:
            continue
        for model, part in group.groupby("model"):
            if model == baseline:
                continue
            paired = part.merge(reference[KEYS + ["predicted"]], on=KEYS,
                                suffixes=("", "_baseline"), validate="one_to_one")
            if paired.empty:
                continue
            error = abs(paired.actual - paired.predicted)
            base_error = abs(paired.actual - paired.predicted_baseline)
            rows.append({"model": model, "baseline": baseline, "horizon": int(h),
                         "n_pairs": len(paired), "baseline_pair_coverage": len(paired) / len(reference),
                         "mae": error.mean(), "baseline_mae_same_pairs": base_error.mean(),
                         "mae_skill_pct": 100 * (1 - error.mean() / base_error.mean()) if base_error.mean() else np.nan,
                         "pair_win_fraction": (error < base_error).mean()})
    return pd.DataFrame(rows, columns=["model", "baseline", "horizon", "n_pairs", "baseline_pair_coverage",
                                       "mae", "baseline_mae_same_pairs", "mae_skill_pct", "pair_win_fraction"])
