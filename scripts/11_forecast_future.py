"""Refit selected individual models and forecast beyond the observed panel."""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.utils.config import load_config, save_dict_yaml, latest_forecast_source
from src.forecasting.lightgbm_model import predict_lightgbm, default_feature_cols


def predict_selected(features, cutoff, horizon, winner, cfg):
    """Use the same model parameters and transformation as the evaluation."""
    params = cfg["models"].get(winner, {})
    if winner == "lightgbm":
        return predict_lightgbm(features, cutoff, horizon,
                               {**params["params"], "quantiles": params.get("quantiles", {})},
                               default_feature_cols(features))
    if winner == "catboost":
        from src.forecasting.catboost_model import predict_catboost
        return predict_catboost(features, cutoff, horizon, params["params"], default_feature_cols(features))
    if winner == "prophet":
        from src.forecasting.prophet_model import predict_prophet
        return predict_prophet(features, cutoff, horizon, {**params, "use_log": cfg.get("use_log", True)}, subsample=0)
    if winner == "chronos":
        from src.forecasting.chronos_model import predict_chronos
        return predict_chronos(features, cutoff, horizon, {**params, "use_log": cfg.get("use_log", True)})
    if winner == "pooled_seasonal":
        from src.forecasting.pooled_seasonal import predict_pooled_seasonal
        return predict_pooled_seasonal(features, cutoff, horizon, params)
    if winner == "short_history":
        from src.forecasting.short_history import predict_short_history
        return predict_short_history(features, cutoff, horizon, params)
    if winner in ("naive_last", "naive_seasonal"):
        from src.forecasting.naive import predict_naive
        return predict_naive(features, cutoff, horizon, mode=winner.removeprefix("naive_"))
    raise ValueError(f"Unsupported deployment model: {winner}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=latest_forecast_source())
    parser.add_argument("--territories", type=int, default=0, help="Zero forecasts all municipalities")
    parser.add_argument("--output-dir", default="outputs/future")
    parser.add_argument("--exclude-models", nargs="*", default=[], help="Explicit exclusions for unavailable runtime dependencies")
    args = parser.parse_args()
    if args.territories < 0:
        parser.error("--territories must be nonnegative")
    cfg = load_config("forecasting")
    import yaml
    run_metadata = {}
    for name in ["run_config.yaml", "baseline_run_config.yaml"]:
        path = Path(args.source) / name
        if path.exists():
            run_metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
            cfg = run_metadata.get("config", cfg)
            break
    cfg.setdefault("use_log", load_config("data")["transform"]["apply"])
    from src.evaluation.provenance import verify_features
    from src.evaluation.provenance import reproducibility_metadata
    verify_features("data/processed/features.parquet", run_metadata)
    evaluation_manifest = Path(args.source) / "evaluation_manifest.yaml"
    if evaluation_manifest.exists():
        verify_features("data/processed/features.parquet", yaml.safe_load(evaluation_manifest.read_text(encoding="utf-8")))
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.PeriodIndex(features.date, freq="M")
    if features.empty:
        raise ValueError("Empty feature panel")
    if args.territories:
        rng = np.random.default_rng(42)
        selected = rng.choice(sorted(features.territory_id.unique()), min(args.territories, features.territory_id.nunique()), replace=False)
        features = features[features.territory_id.isin(selected)]
    cutoff = features.date.max()
    summary = pd.read_csv(Path(args.source) / "forecast_common_summary.csv")
    rows = []
    chosen = {}
    for horizon, group in summary.groupby("horizon"):
        # Deployment chooses an individual model; ensemble weights need their own state.
        candidates = group[~group.model.isin(["ensemble", "causal_selector", *args.exclude_models])].dropna(subset=["mae"])
        if candidates.empty:
            raise ValueError(f"No individual models available for horizon {horizon}")
        winner = candidates.sort_values(["mae", "model"]).iloc[0].model
        h = int(horizon)
        chosen[h] = winner
        model_cfg = copy.deepcopy(cfg)
        refreshed = Path(args.source) / f"refreshed_{winner}_config.yaml"
        if refreshed.exists():
            updated = yaml.safe_load(refreshed.read_text(encoding="utf-8")).get("config", {})
            model_cfg["models"][winner] = updated.get("models", {}).get(winner, model_cfg["models"].get(winner, {}))
        pred = predict_selected(features, cutoff, h, winner, model_cfg)
        if pred.empty:
            pred = pd.DataFrame(columns=["territory_id", "category", "date", "predicted"])
        pred = pred[pred.date == cutoff + h].copy()
        pred = pred[np.isfinite(pred.predicted)]
        pred["model"] = winner
        pred["prediction_model"] = winner
        if "fallback" in pred:
            pred.loc[pred.fallback == "seasonal_no_training_pairs", "prediction_model"] = "naive_seasonal"
        from src.forecasting.naive import predict_naive
        available = features[["territory_id", "category"]].drop_duplicates()
        missing = available.merge(pred[["territory_id", "category"]], on=["territory_id", "category"], how="left", indicator=True)
        missing = missing[missing._merge == "left_only"][["territory_id", "category"]]
        if not missing.empty:
            fallback = predict_naive(features, cutoff, h, mode="seasonal")
            fallback = fallback[fallback.date == cutoff + h].merge(missing, on=["territory_id", "category"])
            fallback["fallback"] = "selected_model_missing_forecast"
            fallback["model"] = winner
            fallback["prediction_model"] = "naive_seasonal"
            pred = pd.concat([pred, fallback], ignore_index=True)
        for column in ["predicted", "lower", "upper"]:
            if column in pred and cfg.get("use_log", True):
                pred[column] = np.maximum(np.expm1(pred[column]), 0)
        pred["model"], pred["horizon"], pred["cutoff"] = winner, h, cutoff
        rows.append(pred)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    future = pd.concat(rows, ignore_index=True)
    if future.duplicated(["territory_id", "category", "horizon", "cutoff", "date"]).any():
        raise ValueError("Duplicate future forecast keys")
    if not np.isfinite(future.predicted).all():
        raise ValueError("Nonfinite future predictions")
    from src.evaluation.calibration import prequential_intervals
    historical = pd.read_csv(Path(args.source) / "forecast_results.csv", low_memory=False)
    combined = prequential_intervals(pd.concat([historical, future], ignore_index=True))
    combined[combined.cutoff == cutoff].drop(columns=["actual"], errors="ignore").to_csv(output / "forecast.csv", index=False)
    save_dict_yaml({"selection_source": args.source, "selected_individual_models": chosen,
                    "reproducibility": reproducibility_metadata(Path(__file__).resolve().parents[1]),
                    "excluded_models": args.exclude_models,
                    "config": cfg, "interval_calibration": {"nominal_coverage": .8, "min_past_pairs": 20,
                                                             "known_outcomes_only": True},
                    "cutoff": str(cutoff), "territories": int(features.territory_id.nunique()),
                    "territory_ids": [int(t) for t in sorted(features.territory_id.unique())],
                    "selection_note": "Retrospective validation winner; selection performance requires independent future evaluation.",
                    "validation_note": "Future targets are unavailable; these rows have no measured forecast quality."},
                   output / "manifest.yaml")


if __name__ == "__main__":
    main()
