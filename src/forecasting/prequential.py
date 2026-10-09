"""Ensemble weights use only forecasts whose outcomes are known at the origin."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


def prequential_combine(raw, features, use_log=True, min_pairs=20, shrinkage=20., min_models=2):
    if min_pairs < 1 or shrinkage < 0 or min_models < 1:
        raise ValueError("Positive minimum counts and nonnegative shrinkage required")
    keys = ["territory_id", "category", "cutoff", "date"]
    base = None
    names = []
    for name, frame in raw.items():
        if frame.empty:
            continue
        names.append(name)
        part = frame[keys + ["predicted"]].rename(columns={"predicted": name})
        base = part if base is None else base.merge(part, on=keys, how="outer", validate="one_to_one")
    if base is None or len(names) < min_models:
        return pd.DataFrame()
    fact = features
    if "observed" in fact:
        fact = fact[fact.observed]
    base = base.merge(fact[["territory_id", "category", "date", "consumption"]],
                      on=["territory_id", "category", "date"], how="left", validate="many_to_one")
    values = np.maximum(np.expm1(base[names].to_numpy()), 0) if use_log else base[names].to_numpy()
    actual = np.expm1(base.consumption.to_numpy()) if use_log else base.consumption.to_numpy()
    output = []
    for cutoff in sorted(base.cutoff.unique()):
        current = np.flatnonzero((base.cutoff == cutoff).to_numpy())
        known = ((base.date <= cutoff) & (base.cutoff < cutoff)).to_numpy() & np.isfinite(actual)
        availability, membership = np.unique(np.isfinite(values[current]), axis=0, return_inverse=True)
        for index, available in enumerate(availability):
            candidates = np.flatnonzero(available)
            if len(candidates) < min_models:
                continue
            indices = current[membership == index]
            # A subsampled model must not restrict all other models' coverage.
            # Learn only from common known past pairs for this candidate set.
            past = known & np.isfinite(values[:, candidates]).all(axis=1)
            unique = []
            for j in candidates:
                if not any(np.allclose(values[indices, j], values[indices, k]) for k in unique):
                    unique.append(j)
            weights = np.ones(len(unique)) / len(unique)
            if past.sum() >= min_pairs:
                errors = np.mean(np.abs(values[past][:, unique] - actual[past, None]), axis=0)
                inverse = 1 / np.maximum(errors, 1e-6)
                learned = inverse / inverse.sum()
                strength = past.sum() / (past.sum() + shrinkage)
                weights = strength * learned + (1 - strength) * weights
            prediction = values[indices][:, unique] @ weights
            part = base.iloc[indices][keys].copy()
            part["predicted"] = np.log1p(np.maximum(prediction, 0)) if use_log else prediction
            part["weight_history_pairs"] = int(past.sum())
            part["ensemble_n_models"] = len(candidates)
            part["ensemble_weights"] = json.dumps({names[j]: float(w) for j, w in zip(unique, weights)}, sort_keys=True)
            output.append(part)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()
