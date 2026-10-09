"""Monthly residual monitoring with independently calibrated online detector."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
from src.changepoint.detectors import run_detector
from src.changepoint.residuals import standardized_residuals


def main():
    parser = argparse.ArgumentParser()
    from src.utils.config import latest_forecast_source
    parser.add_argument("--source", default=latest_forecast_source())
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--output-dir", default="outputs/monitoring")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("outputs/calibration/selected_changepoint.yaml").read_text(encoding="utf-8"))
    source = Path(args.source) / "forecast_results.csv"
    from src.evaluation.audit import validate_forecasts
    forecast = validate_forecasts(pd.read_csv(source, low_memory=False))
    forecast = forecast[(forecast.model == args.model) & (forecast.horizon == 1)].copy()
    if forecast.empty:
        raise ValueError(f"No monthly forecasts for model {args.model}")
    forecast["date"] = pd.PeriodIndex(forecast.date, freq="M")
    rows = []
    for (tid, category), group in forecast.groupby(["territory_id", "category"], observed=True):
        group = group.sort_values("date").copy()
        months = pd.period_range(group.date.min(), group.date.max(), freq="M")
        dense = group.set_index("date").reindex(months)
        dense["territory_id"], dense["category"] = tid, category
        dense["residual"] = dense.actual - dense.predicted
        dense["zscore"] = standardized_residuals(dense.actual.to_numpy(), dense.predicted.to_numpy(),
                                                  std_window=cfg["residual_based"].get("std_window", 6)).to_numpy()
        dense["signal"] = 0
        dense["score_status"] = np.where(dense.actual.isna() | dense.predicted.isna(), "missing_fact_or_forecast",
                                          np.where(dense.zscore.isna(), "warmup", "ready"))
        # Missing months break the online state; do not compress calendar gaps.
        valid = dense.zscore.notna()
        blocks = (~valid).cumsum()
        for _, block in dense[valid].groupby(blocks[valid]):
            alarms = run_detector(cfg["residual_based"]["detector"], block.zscore.to_numpy(), cfg)
            dense.loc[block.index[alarms], "signal"] = 1
        rows.append(dense.reset_index(names="date"))
    monitored = pd.concat(rows, ignore_index=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    monitored.to_csv(output / "monthly_residuals.csv", index=False)
    monitored[monitored.signal > 0].to_csv(output / "signals.csv", index=False)
    examples = monitored.dropna(subset=["zscore"]).copy()
    examples["abs_zscore"] = abs(examples.zscore)
    examples.nlargest(20, "abs_zscore").to_csv(output / "review_candidates.csv", index=False)
    import hashlib
    from src.utils.config import save_dict_yaml
    save_dict_yaml({"forecast_source": str(source), "forecast_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "model": args.model, "horizon": 1, "detector_config": cfg,
                    "interpretation": "Residual alerts are detected anomalies, not verified warnings before future shocks."},
                   output / "monitoring_manifest.yaml")
    print(f"Monthly rows: {len(monitored)}; valid scores: {monitored.zscore.notna().sum()}; alarms: {monitored.signal.sum()}")


if __name__ == "__main__":
    main()
