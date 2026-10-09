"""Synthetic causal diffusion experiment; not evidence of real-world lead time."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.changepoint.detectors import run_detector
from src.changepoint.diffusion import neighbor_signal_score
from src.changepoint.early_warning import evaluate_warnings


def main():
    cfg = load_config("changepoint")
    rng = np.random.default_rng(7042)
    dates = pd.period_range("2020-01", periods=60, freq="M")
    # Six layers, two linked municipalities per layer; propagation takes 4 months.
    neighbors = {i: [j for j in range(12) if abs(i // 2 - j // 2) == 1] for i in range(12)}
    totals = []
    for episode in range(100):
        propagation = rng.random() < .7
        events = []
        series = {}
        for tid in range(12):
            y = rng.normal(size=60)
            if propagation:
                onset = 20 + (tid // 2) * 4
                y[onset:] += 3.
                events.append({"territory_id": tid, "date": dates[onset]})
            series[tid] = y
        truth = pd.DataFrame(events, columns=["territory_id", "date"])
        for detector in ["cusum", "bocpd"]:
            signal_rows = []
            for tid, values in series.items():
                for index in run_detector(detector, values, cfg):
                    signal_rows.append({"territory_id": tid, "date": dates[index], "signal": 1})
            signals = pd.DataFrame(signal_rows, columns=["territory_id", "date", "signal"])
            score = neighbor_signal_score(signals, neighbors, window=1)
            alerts = score[score.neighbor_shocks >= 2] if not score.empty else pd.DataFrame(columns=["territory_id", "date"])
            alerts = alerts[(alerts.date >= dates[0]) & (alerts.date <= dates[-1])]
            result = evaluate_warnings(alerts, truth, max_lead=3)
            result.update(episode=episode, detector=detector, propagation=propagation)
            totals.append(result)
    raw = pd.DataFrame(totals)
    rows = []
    for detector, group in raw.groupby("detector"):
        tp, fp, fn = group[["tp", "fp", "fn"]].sum()
        lead = (group.mean_lead_months.fillna(0) * group.tp).sum() / max(tp, 1)
        rows.append({"detector": detector, "precision": tp / max(tp + fp, 1),
                     "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                     "mean_lead_months": lead, "episodes": len(group),
                     "scenario": "synthetic layered propagation; 3-sigma level shifts"})
    out = Path("outputs/diffusion_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
