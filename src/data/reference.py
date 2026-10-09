# -*- coding: utf-8 -*-
"""Справочник территорий (названия МО). Загружается при наличии файла."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("sber_index.data")

KNOWN_LOCATIONS = [
    "data/reference/territories.csv",
    "data/reference/territories.parquet",
    "data/hackathonlicence/territories.csv",
    "data/hackathonlicence/territories.parquet",
]


def load_reference(extra_paths: list[str] | None = None) -> pd.DataFrame | None:
    """Ищет справочник (id -> название территории) и возвращает его.

    Формат ожидаемых колонок: territory_id, name (или название).
    Если файл не найден, возвращает None — пайплайн продолжит без названий.
    """
    candidates = KNOWN_LOCATIONS + (extra_paths or [])
    for cand in candidates:
        path = Path(cand)
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            df.columns = [str(c).strip().lower() for c in df.columns]
            rename = {"id": "territory_id", "name": "name", "название": "name"}
            df = df.rename(columns=rename)
            if "territory_id" in df.columns and "name" in df.columns:
                logger.info("Справочник территорий загружен: %s (%d записей)", path, len(df))
                return df
            logger.warning("Справочник %s не содержит колонок territory_id/name", path)
    logger.warning("Справочник территорий не найден; работаем только с territory_id")
    return None