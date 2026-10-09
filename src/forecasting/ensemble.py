# -*- coding: utf-8 -*-
"""Ансамбль прогнозных моделей: inverse-error weighting и простое среднее."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("sber_index.forecasting")


def inverse_error_weights(model_mae: dict[str, float], epsilon: float = 1e-6) -> dict[str, float]:
    """Веса моделей обратно пропорциональны их MAE."""
    inv = {m: 1.0 / max(mae, epsilon) for m, mae in model_mae.items()}
    total = sum(inv.values())
    return {m: w / total for m, w in inv.items()}


def combine(predictions: dict[str, pd.DataFrame], method: str = "inverse_error",
            model_mae: dict[str, float] | None = None,
            min_models: int = 2) -> pd.DataFrame:
    """Объединяет прогнозы нескольких моделей.

    predictions: {model_name: DataFrame(territory_id, category, date, predicted)}.
    method: 'simple_average' | 'inverse_error'.
    Возвращает DataFrame с колонками + predicted (комбинированный).
    """
    valid = {name: df for name, df in predictions.items() if df is not None and not df.empty}
    frames = list(valid.values())
    if len(frames) < min_models:
        raise ValueError(f"Недостаточно моделей для ансамбля: {len(frames)}")

    if method == "inverse_error":
        if model_mae is None:
            raise ValueError("Для inverse_error нужны model_mae")
        weights = inverse_error_weights({n: model_mae[n] for n in valid})
    else:
        weights = {m: 1.0 / len(frames) for m in valid}

    keys = ["territory_id", "category", "date"]
    if all("cutoff" in df for df in frames):
        keys.append("cutoff")
    base = None
    for name, df in valid.items():
        frame = df[keys + ["predicted"]].rename(columns={"predicted": name})
        base = frame if base is None else base.merge(frame, on=keys, how="inner", validate="one_to_one")
    base["predicted"] = sum(base[name] * weight for name, weight in weights.items())
    base = base[keys + ["predicted"]]
    logger.info("Ансамбль (%s): моделей %d, строк %d",
                method, len(valid), len(base))
    return base
