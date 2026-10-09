"""Choose models using only outcomes available at each forecast origin."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["territory_id", "category", "cutoff", "date"]


def causal_select(results, min_pairs=20, min_origins=2, warmup_model="pooled_seasonal"):
    """Category/horizon selection on common past pairs; deterministic warmup.

    Inputs and outputs are in original consumption units. A selected model's
    unavailable rows use the warmup model, with explicit attribution.
    """
    if min_pairs < 1 or min_origins < 1:
        raise ValueError("min_pairs and min_origins must be positive")
    frame = results[~results.model.isin(["ensemble", "causal_selector"])].copy()
    for col in ["date", "cutoff"]:
        frame[col] = pd.PeriodIndex(frame[col], freq="M")
    if frame.duplicated(["model", "horizon"] + KEYS).any():
        raise ValueError("Duplicate forecast keys")
    if (frame.date != frame.cutoff + frame.horizon).any():
        raise ValueError("Forecast dates do not match exact horizons")
    output = []
    for (_, _), group in frame.groupby(["horizon", "category"], observed=True):
        for cutoff in sorted(group.cutoff.unique()):
            current = group[group.cutoff == cutoff]
            names = sorted(current.model.unique())
            if warmup_model not in names:
                raise ValueError(f"Warmup model {warmup_model} unavailable at {cutoff}")
            past = group[(group.cutoff < cutoff) & (group.date <= cutoff) & group.model.isin(names)].copy()
            past = past[np.isfinite(past.actual) & np.isfinite(past.predicted)]
            counts = past.groupby(KEYS, observed=True).model.nunique()
            common = counts[counts == len(names)].reset_index()[KEYS]
            past = past.merge(common, on=KEYS, validate="many_to_one")
            n_pairs = len(common)
            n_origins = past.cutoff.nunique()
            winner = warmup_model
            reason = "warmup"
            if n_pairs >= min_pairs and n_origins >= min_origins:
                errors = past.assign(error=abs(past.actual - past.predicted)).groupby("model").error.mean()
                winner = errors.sort_index().sort_values(kind="stable").index[0]
                reason = "past_common_mae"
            chosen = current[current.model == winner].copy()
            chosen = chosen[np.isfinite(chosen.predicted)]
            backup = current[current.model == warmup_model].copy()
            missing = backup.merge(chosen[KEYS], on=KEYS, how="left", indicator=True)
            missing = missing[missing._merge == "left_only"].drop(columns="_merge")
            if not missing.empty:
                missing["selection_reason"] = "selected_model_missing_forecast"
            chosen["selection_reason"] = reason
            chosen = pd.concat([chosen, missing], ignore_index=True)
            chosen["selected_model"] = chosen.model
            chosen["selection_history_pairs"] = n_pairs
            chosen["selection_history_origins"] = n_origins
            chosen["model"] = "causal_selector"
            output.append(chosen)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()
