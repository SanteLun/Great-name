"""Robust category seasonality pooled across municipalities, shrunk local growth."""
from __future__ import annotations
import numpy as np
import pandas as pd


def predict_pooled_seasonal(features_df, cutoff, horizon, params=None):
    params = params or {}
    if horizon < 1 or params.get("level_window", 3) < 1 or params.get("growth_shrinkage", 6.) < 0:
        raise ValueError("Positive horizon/window and nonnegative shrinkage required")
    hist = features_df[features_df.date <= cutoff].copy()
    if "observed" in hist:
        hist = hist[hist.observed]
    hist = hist.dropna(subset=["consumption"])
    if hist.empty:
        return pd.DataFrame()
    hist["month"] = hist.date.map(lambda d: d.month)
    # Center each series/year so spending levels do not dominate seasonal shape.
    hist["year"] = hist.date.map(lambda d: d.year)
    center = hist.groupby(["territory_id", "category", "year"], observed=True).consumption.transform("mean")
    hist["centered"] = hist.consumption - center
    count = hist.groupby(["territory_id", "category", "year"], observed=True).date.transform("count")
    complete = hist[count >= 10]
    profile = complete.groupby(["category", "month"], observed=True).centered.median()
    keys = ["territory_id", "category"]
    monthly_keys = keys + ["date"]
    if hist.duplicated(monthly_keys).any():
        raise ValueError("Duplicate monthly series keys")
    hist = hist.sort_values(monthly_keys)
    hist = hist.merge(profile.rename("season").reset_index(), on=["category", "month"], how="left", validate="many_to_one")
    hist["deseasonalized"] = hist.consumption - hist.season.fillna(0.)
    # Vectorized joins preserve calendar gaps and avoid thousands of tiny frames.
    lookup = hist[monthly_keys + ["consumption"]].copy()
    lookup["date"] += 12
    lookup = lookup.rename(columns={"consumption": "previous_year"})
    hist = hist.merge(lookup, on=monthly_keys, how="left", validate="one_to_one")
    hist["yoy"] = hist.consumption - hist.previous_year
    growth = hist.groupby(keys, observed=True).yoy.agg(["median", "count"])
    denominator = growth["count"] + params.get("growth_shrinkage", 6.)
    growth["growth"] = (growth["median"].fillna(0.) * growth["count"] / denominator.replace(0, np.nan)).fillna(0.).clip(-.5, .5)
    recent = hist.groupby(keys, observed=True).tail(params.get("level_window", 3))
    level = recent.groupby(keys, observed=True).deseasonalized.median().rename("level")
    out = level.to_frame().join(growth[["growth"]]).reset_index()
    target = cutoff + horizon
    target_profile = profile.xs(target.month, level="month") if target.month in profile.index.get_level_values("month") else pd.Series(dtype=float)
    seasonal = out.category.map(target_profile).astype(float).fillna(0.)
    out["date"] = target
    out["predicted"] = out.level + seasonal + out.growth * horizon / 12
    return out[keys + ["date", "predicted"]]
