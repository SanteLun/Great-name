"""Combine verified independent model runs and recompute a causal ensemble."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.evaluation.audit import validate_forecasts
from src.evaluation.provenance import verify_features, reproducibility_metadata
from src.evaluation.walk_forward import attach_actuals, save_results, summarize
from src.forecasting.prequential import prequential_combine
from src.utils.config import load_config, save_dict_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/full_panel")
    parser.add_argument("--foundation-source")
    parser.add_argument("--prophet-source")
    parser.add_argument("--output-dir", default="outputs/full_combined")
    args = parser.parse_args()
    feature_path = Path("data/processed/features.parquet")
    sources = [(args.source, None)]
    if args.foundation_source:
        sources.append((args.foundation_source, "chronos"))
    if args.prophet_source:
        sources.append((args.prophet_source, "prophet"))
    output = Path(args.output_dir)
    if any(Path(directory).resolve() == output.resolve() for directory, _ in sources):
        parser.error("Output directory must differ from all input runs")
    parts, provenance, primary = [], [], {}
    columns = ["territory_id", "category", "date", "cutoff", "horizon", "model", "actual", "predicted", "lower", "upper", "fallback"]
    for directory, selected_model in sources:
        directory = Path(directory)
        metadata = {}
        for name in ["run_config.yaml", "baseline_run_config.yaml", "evaluation_manifest.yaml"]:
            path = directory / name
            if path.exists():
                current = yaml.safe_load(path.read_text(encoding="utf-8"))
                verify_features(feature_path, current)
                metadata.update(current)
        if not metadata.get("features_sha256"):
            raise ValueError(f"Missing feature fingerprint: {directory}")
        if selected_model is None:
            primary = metadata
        path = directory / "forecast_results.csv"
        frame = pd.read_csv(path, low_memory=False, usecols=lambda column: column in columns)
        frame = frame[frame.model == selected_model] if selected_model else frame[~frame.model.isin(["ensemble", "causal_selector"])]
        if frame.empty:
            raise ValueError(f"No requested predictions in {directory}")
        parts.append(validate_forecasts(frame))
        provenance.append({"directory": str(directory), "model_filter": selected_model,
                           "forecast_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "run_metadata": metadata})
        foundation_manifest = directory / "model_manifest.yaml"
        if foundation_manifest.exists():
            provenance[-1]["model_manifest"] = yaml.safe_load(foundation_manifest.read_text(encoding="utf-8"))
    results = validate_forecasts(pd.concat(parts, ignore_index=True))
    features = pd.read_parquet(feature_path)
    features["date"] = pd.PeriodIndex(features.date, freq="M")
    training_ids = primary.get("territory_ids")
    if training_ids is not None:
        features = features[features.territory_id.isin(training_ids)]
    if not set(results.territory_id).issubset(set(features.territory_id)):
        raise ValueError("Additional model predicts territories outside primary training panel")
    cfg = primary.get("config", load_config("forecasting"))
    use_log = cfg.get("use_log", True)
    ecfg = cfg.get("ensemble", {})
    ensembles = []
    for horizon, group in results.groupby("horizon"):
        raw = {}
        for model, frame in group.groupby("model"):
            frame = frame.copy()
            if use_log:
                frame["predicted"] = np.log1p(frame.predicted)
            raw[model] = frame
        combined = prequential_combine(raw, features, use_log, ecfg.get("min_pairs", 20),
                                       ecfg.get("shrinkage", 20.), ecfg.get("min_models", 2))
        if not combined.empty:
            combined = attach_actuals(combined, features, use_log).dropna(subset=["actual", "predicted"])
            combined["model"], combined["horizon"] = "ensemble", int(horizon)
            ensembles.append(combined)
    results = validate_forecasts(pd.concat([results, *ensembles], ignore_index=True))
    save_results(results, summarize(results), str(output))
    primary = dict(primary)
    primary["parent_forecast_results_sha256"] = primary.get("forecast_results_sha256")
    primary["forecast_results_sha256"] = hashlib.sha256((output / "forecast_results.csv").read_bytes()).hexdigest()
    save_dict_yaml(primary, output / "run_config.yaml")
    save_dict_yaml({"sources": provenance, "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                    "reproducibility": reproducibility_metadata(Path(__file__).resolve().parents[1]),
                    "comparison_note": "Prophet and Chronos fit each series independently; a sampled Prophet baseline does not depend on other training territories.",
                    "ensemble": "Available models only; weights on common past known outcomes; duplicate fallback forecasts get one vote."},
                   output / "evaluation_manifest.yaml")
    for name in ["refreshed_prophet_config.yaml", "refreshed_lightgbm_config.yaml"]:
        (output / name).unlink(missing_ok=True)
    for directory, selected_model in sources:
        import shutil
        for name in ["refreshed_prophet_config.yaml", "refreshed_lightgbm_config.yaml"]:
            if selected_model and name != f"refreshed_{selected_model}_config.yaml":
                continue
            path = Path(directory) / name
            if path.exists():
                shutil.copy2(path, output / name)
    print(summarize(results).to_string(index=False))


if __name__ == "__main__":
    main()
