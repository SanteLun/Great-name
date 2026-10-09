# -*- coding: utf-8 -*-
"""Шаг 3. Обнаружение точек структурных изменений (шоков).

1. Синтетический бенчмарк: сравнение CUSUM / PELT / BinSeg / BOCPD на рядах
   с инжектированными шоками (precision/recall/F1, задержка обнаружения).
2. Детекция шоков на остатках прогнозной модели на реальных данных (если
   есть результаты шага 2) с калибровкой на известных макро-событиях ЦБ.

Запуск:  python scripts/03_run_changepoint.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.changepoint.benchmark import run_benchmark
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("sber_index.03")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/forecasting")
    args = parser.parse_args()
    cp_cfg = load_config("changepoint")
    out_dir = Path(cp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Синтетический бенчмарк ---
    t0 = time.time()
    raw, summary = run_benchmark(cp_cfg)
    raw.to_csv(out_dir / "benchmark_raw.csv", index=False)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    logger.info("Бенчмарк: %.1f c", time.time() - t0)
    print("\n=== СИНТЕТИЧЕСКИЙ БЕНЧМАРК ДЕТЕКТОРОВ ШОКОВ ===")
    print(summary.round(4).to_string(index=False))

    # --- 2. Калибровка на реальных событиях 2023-2024 ---
    events = pd.DataFrame(cp_cfg["real_events"])
    print("\n=== ИЗВЕСТНЫЕ МАКРО-СОБЫТИЯ ДЛЯ КАЛИБРОВКИ (2023-2024) ===")
    print(events.to_string(index=False))

    # --- 3. Остатки прогнозной модели (если есть результаты шага 2) ---
    fc_results = Path(args.source) / "forecast_results.csv"
    if fc_results.exists():
        from src.changepoint.detectors import run_detector
        from src.changepoint.diffusion import spatial_trigger
        from src.changepoint.residuals import residuals_from_forecast
        from src.data.loader import load_all

        data_cfg = load_config("data")
        cons, ma, conn = load_all(data_cfg)
        pred = pd.read_csv(fc_results)
        if "cutoff" not in pred:
            raise ValueError("Прогнозы устарели: повторите шаг 02 с новой валидацией")
        pred = pred[(pred.model == cp_cfg["residual_based"].get("model", "lightgbm")) &
                    (pred.horizon == 1)].copy()
        resid = residuals_from_forecast(pred, cons,
                                       cp_cfg["residual_based"].get("std_window", 6))
        resid.to_csv(out_dir / "residuals.csv", index=False)
        logger.info("Остатки прогнозов сохранены: %d строк", len(resid))

        # детектор CUSUM на остатках реальных серий (по месяцам)
        signals_rows = []
        detector = cp_cfg["residual_based"].get("detector", "bocpd")
        for (tid, cat), grp in resid.groupby(["territory_id", "category"]):
            grp = grp.sort_values("date")
            grp = grp.dropna(subset=["zscore"])
            blocks = grp.date.map(lambda d: d.ordinal).diff().fillna(1).ne(1).cumsum()
            for _, block in grp.groupby(blocks):
                cps = run_detector(detector, block["zscore"].to_numpy(), cp_cfg)
                for i in cps:
                    if i < len(block):
                        signals_rows.append({"territory_id": tid, "category": cat,
                                             "date": block.iloc[i]["date"], "signal": 1})
        signals = pd.DataFrame(signals_rows, columns=["territory_id", "category", "date", "signal"])
        sig_out = out_dir / f"signals_{detector}.csv"
        signals.to_csv(sig_out, index=False)
        logger.info("%s-сигналы на реальных остатках: %d (%s)",
                    detector, len(signals), sig_out)

        # пространственная диффузия: срабатывания у соседей
        if not signals.empty:
            sp = spatial_trigger(signals, conn, k=cp_cfg["benchmark"].get("neighbor_k", 8),
                                 window=3, min_neighbors=2)
            sp_out = out_dir / "spatial_trigger.csv"
            sp.to_csv(sp_out, index=False)
            n_trig = int(sp.spatial_trigger.sum())
            logger.info("Пространственный триггер: %d МО-месяцев с сигналом от соседей",
                        n_trig)
            print(f"\nПространственный триггер диффузии: {n_trig} срабатываний "
                  f"(соседи сигналили раньше) сохранён в outputs/changepoint/spatial_trigger.csv")
    else:
        logger.warning("Нет outputs/forecasting/forecast_results.csv — шаг 2 не выполнен")


if __name__ == "__main__":
    main()
