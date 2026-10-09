# -*- coding: utf-8 -*-
"""Загрузка YAML-конфигураций проекта."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def latest_forecast_source():
    """Prefer the most recently saved completed standard forecast run."""
    candidates = []
    for directory in ["outputs/full_audited", "outputs/audited", "outputs/final", "outputs/validated", "outputs/forecasting"]:
        path = PROJECT_ROOT / directory
        results = path / "forecast_results.csv"
        if not results.exists():
            continue
        manifest = path / "run_config.yaml"
        if manifest.exists():
            metadata = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            if metadata.get("status") not in (None, "complete"):
                continue
        candidates.append((results.stat().st_mtime_ns, path))
    return str(max(candidates, key=lambda pair: pair[0])[1]) if candidates else str(PROJECT_ROOT / "outputs/forecasting")


def load_config(name: str) -> dict[str, Any]:
    """Загружает конфиг из configs/<name>.yaml."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_dict_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Сохраняет словарь в YAML (для артефактов)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
