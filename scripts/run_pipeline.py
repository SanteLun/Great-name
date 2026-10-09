"""One-command reproducible local workflow; downloads are explicit flags."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--territories", type=int, default=0)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--chronos", action="store_true")
    parser.add_argument("--collect-news", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-prophet", action="store_true", help="Explicitly omit Prophet when CmdStan is unavailable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def run(script, *arguments):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(root / "scripts" / script), *map(str, arguments)], cwd=root, check=True)
    if args.prepare:
        run("01_prepare_data.py")
    run("02_run_forecasting.py", "--territories", args.territories, "--output-dir", "outputs/validated",
        *(["--quick"] if args.quick else []),
        *(["--models", "naive", "lightgbm", "catboost", "pooled_seasonal", "short_history"] if args.skip_prophet else []))
    run("06_improve_evaluation.py", "--source", "outputs/validated", "--output-dir", "outputs/improved",
        *(["--chronos"] if args.chronos else []))
    run("14_audit_solution.py", "--source", "outputs/improved", "--output-dir", "outputs/final")
    run("13_calibrate_intervals.py", "--source", "outputs/final")
    run("07_calibrate_detectors.py")
    run("10_monitor_validated.py", "--source", "outputs/validated")
    run("12_diffusion_benchmark.py")
    if args.collect_news:
        run("08_collect_official_news.py")
    if (root / "data/news/cbr_releases.csv").exists():
        run("09_news_ablation.py", "--source", "outputs/validated", "--dense")
    run("03_run_changepoint.py", "--source", "outputs/validated")
    run("05_make_report.py", "--source", "outputs/final")


if __name__ == "__main__":
    main()
