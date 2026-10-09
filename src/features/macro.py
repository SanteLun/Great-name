# -*- coding: utf-8 -*-
"""Проверенная ключевая ставка ЦБ на конец месяца с лагами 1/3.

Данные: configs/macro.csv; происхождение: configs/macro_sources.yaml.
Ошибочные ИПЦ и неподтверждённый валютный курс не используются.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("sber_index.features")

MACRO_PATH = Path("configs/macro.csv")
MACRO_LAGS = [1, 3]  # значения прошлого месяца; текущая публикация ещё не гарантирована


def load_macro(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else MACRO_PATH
    if not path.exists():
        logger.warning("Макро-справочник не найден: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.PeriodIndex(df["date"], freq="M")
    # Only the end-of-month policy rate is verified in the supplied reference.
    # Historical CPI/FX columns in the original CSV are not reliable vintages.
    return df[["date", "key_rate"]]


def add_macro_features(panel: pd.DataFrame, macro: pd.DataFrame,
                       lags: list[int] | None = None) -> pd.DataFrame:
    """Добавляет макро-признаки в панель (merge по месяцу)."""
    if macro.empty:
        return panel
    out = panel.copy()
    lags = lags or MACRO_LAGS
    cols = [c for c in macro.columns if c != "date"]

    m = macro.set_index("date")
    for lag in lags:
        if lag < 1:
            raise ValueError("Macro lag must be positive")
        mm = m.copy()
        mm.index = mm.index + lag
        mm = mm.reset_index()
        rename = {c: f"macro_{c}_lag{lag}" for c in cols}
        mm = mm.rename(columns=rename)[["date"] + list(rename.values())]
        out = out.merge(mm, on="date", how="left")
    logger.info("Макро-признаки добавлены: %d колонок (%s)",
                len(cols) * len(lags), ", ".join(f"{c}_lag{l}" for c in cols[:1] for l in lags))
    return out
