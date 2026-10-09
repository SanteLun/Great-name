# -*- coding: utf-8 -*-
"""Шаг 1. Подготовка данных и построение признаков.

Загружает три датасета СберИндекса, восстанавливает полную панель
(date x territory x category), заполняет пропуски (линейная интерполяция +
пространственная по соседям), применяет log1p-трансформацию и строит признаки
(календарь, лаги, скользящие статистики, пространственные лаги диффузии,
market_access). Сохраняет результат в data/processed/features.parquet.

Запуск:  python scripts/01_prepare_data.py [--territories N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_all, complete_panel, filter_series
from src.data.cleaner import impute_missing, apply_transform
from src.features.builder import build_features
from src.utils.config import load_config, PROJECT_ROOT
from src.utils.logging import setup_logger

logger = setup_logger("sber_index.01")


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка данных и признаков")
    parser.add_argument("--territories", type=int, default=0,
                        help="Ограничение числа МО для отладки (0 = все)")
    args = parser.parse_args()

    data_cfg = load_config("data")
    feat_cfg = load_config("features")

    t0 = time.time()
    cons, ma, conn = load_all(data_cfg)

    if args.territories:
        terrs = cons.territory_id.unique()[: args.territories]
        cons = filter_series(cons, territory_ids=terrs)
        logger.info("Отладочный режим: %d МО", len(terrs))

    panel = complete_panel(cons)
    panel["observed"] = panel["consumption"].notna()
    logger.info("Полная сетка: %d строк, NaN в цели: %d",
                len(panel), int(panel.consumption.isna().sum()))

    panel = impute_missing(panel, conn, data_cfg)
    panel["consumption"] = apply_transform(panel["consumption"],
                                           use_log=data_cfg["transform"]["apply"])

    features = build_features(panel, conn, ma, feat_cfg)

    out_dir = Path(data_cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Сохранено: %s (%.1f c)", out_path, time.time() - t0)


if __name__ == "__main__":
    main()
