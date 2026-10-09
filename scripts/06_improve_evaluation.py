"""Reevaluate added models on exactly the saved baseline origins and series."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.evaluation.walk_forward import attach_actuals, save_results, summarize, common_results
from src.forecasting.short_history import predict_short_history
from src.forecasting.prequential import prequential_combine


def paired_bootstrap(results, seed=42, repetitions=1000):
    """Cluster bootstrap: all origins of one municipality/category stay together."""
    rows = []
    rng = np.random.default_rng(seed)
    keys = ["territory_id", "category", "cutoff", "date"]
    for horizon, group in results.groupby("horizon"):
        baseline = group[group.model == "prophet"][keys + ["actual", "predicted"]]
        for model, part in group.groupby("model"):
            if model == "prophet":
                continue
            paired = part.merge(baseline, on=keys, suffixes=("", "_prophet"), validate="one_to_one")
            if paired.empty:
                continue
            paired["difference"] = abs(paired.actual - paired.predicted) - abs(paired.actual_prophet - paired.predicted_prophet)
            clusters = paired.groupby(["territory_id", "category"], observed=True).difference.agg(["sum", "count"])
            draw = rng.integers(len(clusters), size=(repetitions, len(clusters)))
            samples = clusters["sum"].to_numpy()[draw].sum(axis=1) / clusters["count"].to_numpy()[draw].sum(axis=1)
            rows.append({"model": model, "horizon": horizon, "n_series": len(clusters),
                         "n_pairs": len(paired), "mae_difference_vs_prophet": paired.difference.mean(),
                         "ci_low": np.quantile(samples, .025), "ci_high": np.quantile(samples, .975)})
    return pd.DataFrame(rows, columns=["model", "horizon", "n_series", "n_pairs", "mae_difference_vs_prophet", "ci_low", "ci_high"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/forecasting")
    parser.add_argument("--output-dir", default="outputs/improved")
    parser.add_argument("--chronos", action="store_true")
    parser.add_argument("--prophet-source", default=None, help="Replace baseline Prophet with corrected independent rerun")
    parser.add_argument("--lightgbm-source", default=None, help="Replace baseline LightGBM with deterministic rerun")
    args = parser.parse_args()
    source = Path(args.source) / "forecast_results.csv"
    baseline = pd.read_csv(source, low_memory=False)
    if args.prophet_source:
        refreshed = pd.read_csv(Path(args.prophet_source) / "forecast_results.csv")
        baseline = pd.concat([baseline[baseline.model != "prophet"], refreshed[refreshed.model == "prophet"]], ignore_index=True)
    if args.lightgbm_source:
        refreshed = pd.read_csv(Path(args.lightgbm_source) / "forecast_results.csv")
        baseline = pd.concat([baseline[baseline.model != "lightgbm"], refreshed[refreshed.model == "lightgbm"]], ignore_index=True)
    for col in ["date", "cutoff"]:
        baseline[col] = pd.PeriodIndex(baseline[col], freq="M")
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.PeriodIndex(features.date, freq="M")
    import yaml
    from src.utils.config import load_config
    defaults = load_config("forecasting")
    run_cfg = defaults
    configuration = Path(args.source) / "run_config.yaml"
    selected = baseline.territory_id.unique()
    if configuration.exists():
        saved_config = yaml.safe_load(configuration.read_text(encoding="utf-8"))
        run_cfg = saved_config.get("config", defaults)
        from src.evaluation.provenance import training_territory_ids, verify_features
        selected = training_territory_ids(saved_config, baseline)
        verify_features("data/processed/features.parquet", saved_config)
    features = features[features.territory_id.isin(selected)]
    use_log = run_cfg.get("use_log", True)
    def model_params(name):
        return {**defaults.get("models", {}).get(name, {}), **run_cfg.get("models", {}).get(name, {})}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parts = []
    for horizon, group in baseline.groupby("horizon"):
        raw = {}
        for model, part in group.groupby("model"):
            if model in ("ensemble", "causal_selector", "short_history", "pooled_seasonal", "chronos"):
                continue
            part = part.copy()
            if use_log:
                part["predicted"] = np.log1p(part.predicted.clip(lower=0))
            raw[model] = part
        predictions = []
        for cutoff in sorted(group.cutoff.unique()):
            part = predict_short_history(features, cutoff, int(horizon), model_params("short_history"))
            part["cutoff"] = cutoff
            predictions.append(part)
        raw["short_history"] = pd.concat(predictions, ignore_index=True)
        added = ["short_history"]
        from src.forecasting.pooled_seasonal import predict_pooled_seasonal
        predictions = []
        for cutoff in sorted(group.cutoff.unique()):
            part = predict_pooled_seasonal(features, cutoff, int(horizon), model_params("pooled_seasonal"))
            part["cutoff"] = cutoff
            predictions.append(part)
        raw["pooled_seasonal"] = pd.concat(predictions, ignore_index=True)
        added.append("pooled_seasonal")
        if args.chronos:
            from src.forecasting.chronos_model import predict_chronos
            predictions = []
            for cutoff in sorted(group.cutoff.unique()):
                part = predict_chronos(features, cutoff, int(horizon), {**model_params("chronos"), "use_log": use_log})
                part["cutoff"] = cutoff
                predictions.append(part)
            raw["chronos"] = pd.concat(predictions, ignore_index=True)
            added.append("chronos")
        for model in added:
            result = attach_actuals(raw[model], features, use_log).dropna(subset=["actual", "predicted"])
            result["model"], result["horizon"] = model, horizon
            parts.append(result)
        ecfg = run_cfg.get("ensemble", {})
        ensemble = prequential_combine(raw, features, use_log, ecfg.get("min_pairs", 20),
                                      ecfg.get("shrinkage", 20.), ecfg.get("min_models", 2))
        if not ensemble.empty:
            result = attach_actuals(ensemble, features, use_log).dropna(subset=["actual", "predicted"])
            result["model"], result["horizon"] = "ensemble", horizon
            parts.append(result)
        parts.append(group[~group.model.isin(["ensemble", "causal_selector", "short_history", "pooled_seasonal", "chronos"])])
    results = pd.concat(parts, ignore_index=True)
    save_results(results, summarize(results), str(output))
    paired_bootstrap(results).to_csv(output / "paired_bootstrap.csv", index=False)
    category = []
    for (cat, horizon, model), group in results.groupby(["category", "horizon", "model"], observed=True):
        category.append({"category": cat, "horizon": horizon, "model": model,
                         "mae": abs(group.actual - group.predicted).mean(), "n_pairs": len(group)})
    pd.DataFrame(category).to_csv(output / "category_metrics.csv", index=False)
    intervals = []
    if "lower" in results and "upper" in results:
        for (model, horizon), group in results.groupby(["model", "horizon"]):
            valid = group.dropna(subset=["lower", "upper", "actual"])
            if valid.empty:
                continue
            intervals.append({"model": model, "horizon": int(horizon), "n_pairs": len(valid),
                              "coverage": ((valid.actual >= valid.lower) & (valid.actual <= valid.upper)).mean(),
                              "mean_width": (valid.upper - valid.lower).mean()})
    pd.DataFrame(intervals, columns=["model", "horizon", "n_pairs", "coverage", "mean_width"]).to_csv(output / "interval_coverage.csv", index=False)
    import hashlib
    import shutil
    from src.utils.config import save_dict_yaml
    metadata = {"baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "features_sha256": hashlib.sha256(Path("data/processed/features.parquet").read_bytes()).hexdigest(),
                "chronos": args.chronos, "bootstrap_seed": 42, "bootstrap_repetitions": 1000,
                "ensemble": "prequential, known outcomes only, deduplicated, arithmetic rubles"}
    if args.prophet_source:
        refreshed_source = Path(args.prophet_source) / "forecast_results.csv"
        metadata["refreshed_prophet_sha256"] = hashlib.sha256(refreshed_source.read_bytes()).hexdigest()
    if args.lightgbm_source:
        refreshed_source = Path(args.lightgbm_source) / "forecast_results.csv"
        metadata["refreshed_lightgbm_sha256"] = hashlib.sha256(refreshed_source.read_bytes()).hexdigest()
    if args.chronos:
        from src.forecasting.chronos_model import load_pipeline
        chcfg = model_params("chronos")
        pipeline = load_pipeline(chcfg.get("model_name", "amazon/chronos-bolt-tiny"), chcfg.get("revision"))
        from src.forecasting.chronos_model import model_manifest
        metadata["foundation_model"] = model_manifest(chcfg)
    save_dict_yaml(metadata, output / "evaluation_manifest.yaml")
    run_config = Path(args.source) / "run_config.yaml"
    if run_config.exists():
        shutil.copy2(run_config, output / "baseline_run_config.yaml")
    for name, directory in [("prophet", args.prophet_source), ("lightgbm", args.lightgbm_source)]:
        if directory and (Path(directory) / "run_config.yaml").exists():
            shutil.copy2(Path(directory) / "run_config.yaml", output / f"refreshed_{name}_config.yaml")
    print(summarize(common_results(results)).to_string(index=False))


if __name__ == "__main__":
    main()
