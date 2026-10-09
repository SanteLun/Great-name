# -*- coding: utf-8 -*-
"""Настройка логирования с выводом в консоль и файл."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str = "sber_index", log_dir: str | Path = "outputs/logs") -> logging.Logger:
    logger = logging.getLogger(name)
    # Configure their common parent so sibling data/model/evaluation modules
    # also report progress instead of inheriting Python's WARNING threshold.
    shared = logging.getLogger("sber_index") if name.startswith("sber_index") else logger
    if shared.handlers:
        return logger
    shared.setLevel(logging.INFO)
    shared.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    shared.addHandler(console)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "run.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    shared.addHandler(file_handler)
    return logger
