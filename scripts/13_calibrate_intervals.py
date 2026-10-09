"""Evaluate causal interval calibration on saved walk-forward predictions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.evaluation.calibration import prequential_intervals, coverage_summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/final")
    args = parser.parse_args()
    root = Path(args.source)
    calibrated = prequential_intervals(pd.read_csv(root / "forecast_results.csv", low_memory=False))
    calibrated.to_csv(root / "calibrated_forecasts.csv", index=False)
    summary = coverage_summary(calibrated)
    summary.to_csv(root / "calibrated_coverage_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
