"""Optional CPU zero-shot Chronos-Bolt adapter; no silent baseline replacement."""
from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
import pandas as pd


def model_manifest(params=None):
    """Record the locally cached revision and exact checkpoint files."""
    import hashlib
    params = params or {}
    name = params.get("model_name", "amazon/chronos-bolt-tiny")
    cache = Path(__file__).resolve().parents[2] / "outputs/models/hub" / ("models--" + name.replace("/", "--"))
    reference = cache / "refs/main"
    revision = params.get("revision") or (reference.read_text().strip() if reference.exists() else None)
    files = {}
    if revision:
        snapshot = cache / "snapshots" / revision
        for path in sorted(snapshot.glob("*")):
            if path.is_file() and path.suffix in (".json", ".safetensors", ".bin"):
                files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"model_name": name, "revision": revision, "configured_revision": params.get("revision"),
            "cached_checkpoint_sha256": files}


@lru_cache(maxsize=2)
def load_pipeline(model_name, revision=None):
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "outputs/models"))
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(4)
    return BaseChronosPipeline.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32,
                                               **({"revision": revision} if revision else {}))


def predict_chronos(features_df, cutoff, horizon, params=None):
    import torch
    params = params or {}
    pipeline = load_pipeline(params.get("model_name", "amazon/chronos-bolt-tiny"), params.get("revision"))
    series, contexts = [], []
    columns = ["date", "territory_id", "category", "consumption"]
    if "observed" in features_df:
        columns.append("observed")
    hist = features_df.loc[features_df.date <= cutoff, columns]
    for key, group in hist.groupby(["territory_id", "category"], observed=True):
        group = group.sort_values("date").set_index("date")
        months = pd.period_range(group.index.min(), cutoff, freq="M")
        y = group.consumption.reindex(months).to_numpy(dtype=float, copy=True)
        if "observed" in group:
            observed = group.observed.reindex(months, fill_value=False).to_numpy(dtype=bool)
            y[~observed] = np.nan
        if not np.isfinite(y).any():
            continue
        # The pretrained model receives the original nonnegative unit, not log values.
        if params.get("use_log", True):
            y = np.expm1(y)
        contexts.append(torch.tensor(y[-params.get("context_length", 24):], dtype=torch.float32))
        series.append(key)
    rows = []
    batch_size = params.get("batch_size", 64)
    for start in range(0, len(contexts), batch_size):
        with torch.inference_mode():
            forecast = pipeline.predict(contexts[start:start + batch_size], prediction_length=horizon)
        values = forecast.detach().cpu().numpy()
        levels = np.array(pipeline.model.chronos_config.quantiles)
        for key, prediction in zip(series[start:start + batch_size], values):
            out = {"territory_id": key[0], "category": key[1], "date": cutoff + horizon}
            for column, level in [("predicted", .5), ("lower", .1), ("upper", .9)]:
                value = max(float(prediction[np.argmin(abs(levels - level)), horizon - 1]), 0.)
                out[column] = np.log1p(value) if params.get("use_log", True) else value
            rows.append(out)
    return pd.DataFrame(rows)
