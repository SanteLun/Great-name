"""Select online detector parameters on disjoint synthetic calibration seeds."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.changepoint.benchmark import run_benchmark
from src.utils.config import load_config, save_dict_yaml


def main():
    original = load_config("changepoint")
    trials = []
    candidates = []
    for threshold in [4., 6., 8.]:
        for drift in [.25, .5, .75]:
            candidates.append(("cusum", {"threshold_factor": threshold, "drift": drift, "cooldown": 3}))
    for hazard in [.02, .05]:
        for threshold in [.4, .6, .8]:
            candidates.append(("bocpd", {"hazard": hazard, "prior_scale": 1., "threshold_prob": threshold}))
    for detector, parameters in candidates:
        cfg = copy.deepcopy(original)
        cfg["benchmark"].update(seed=10042, n_series=200)
        for name in cfg["methods"]:
            cfg["methods"][name]["enable"] = name == detector
        cfg["methods"][detector].update(parameters)
        _, summary = run_benchmark(cfg)
        trials.append({**summary.iloc[0].to_dict(), "parameters": parameters})
    table = pd.DataFrame(trials)
    # False alarms are operationally important; selection never uses test seed 42.
    eligible = table[table.false_alarm_series_rate <= .15]
    if eligible.empty:
        eligible = table
    winner = eligible.sort_values("micro_f1", ascending=False).iloc[0]
    selected = copy.deepcopy(original)
    for detector in ["cusum", "bocpd"]:
        candidates = eligible[eligible.detector == detector]
        if not candidates.empty:
            best = candidates.sort_values("micro_f1", ascending=False).iloc[0]
            selected["methods"][detector].update(best.parameters)
    selected["residual_based"]["detector"] = winner.detector
    output = Path("outputs/calibration")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "calibration_trials.csv", index=False)
    save_dict_yaml(selected, output / "selected_changepoint.yaml")
    raw, summary = run_benchmark(selected)
    raw.to_csv(output / "heldout_raw.csv", index=False)
    summary.to_csv(output / "heldout_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
