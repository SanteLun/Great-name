"""Event-level evaluation of warnings strictly before future annotated events."""
from __future__ import annotations
import pandas as pd


def evaluate_warnings(warnings, events, max_lead=3):
    if max_lead < 1:
        raise ValueError("max_lead must be positive")
    keys = ["territory_id"] + (["category"] if "category" in events else [])
    required = keys + ["date"]
    for name, frame in [("warnings", warnings), ("events", events)]:
        if set(required) - set(frame):
            raise ValueError(f"{name}: missing event identifiers")
        if frame[required].isna().any().any():
            raise ValueError(f"{name}: missing event identifiers")
    alerts = warnings.copy()
    truth = events.copy()
    alerts["date"] = pd.PeriodIndex(alerts.date, freq="M")
    truth["date"] = pd.PeriodIndex(truth.date, freq="M")
    alerts = alerts.drop_duplicates(required).reset_index(drop=True)
    truth = truth.drop_duplicates(required).reset_index(drop=True)
    matched = set()
    leads = []
    for row in alerts.sort_values("date").itertuples(index=False):
        candidates = truth
        for key in keys:
            candidates = candidates[candidates[key] == getattr(row, key)]
        eligible = [(index, event.date.ordinal - row.date.ordinal)
                    for index, event in candidates.iterrows()
                    if index not in matched and 0 < event.date.ordinal - row.date.ordinal <= max_lead]
        if eligible:
            index, lead = min(eligible, key=lambda pair: pair[1])
            matched.add(index)
            leads.append(lead)
    tp, fp, fn = len(leads), len(alerts) - len(leads), len(truth) - len(leads)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "mean_lead_months": sum(leads) / len(leads) if leads else float("nan")}
