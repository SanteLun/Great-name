# -*- coding: utf-8 -*-
"""Шаг 2. Walk-forward валидация моделей прогноза на горизонтах 1/3/6/12 мес.

Сравниваются: naive(last), naive(seasonal), Prophet (база Сбера), LightGBM
(глобальная модель). Метрики: MAE (обязательная), RMSE, MAPE, R2.
Результаты: outputs/forecasting/forecast_results.csv и forecast_summary.csv.

Запуск:  python scripts/02_run_forecasting.py [--horizons 1 3 6 12]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.loader import load_all
from src.evaluation.walk_forward import evaluate_horizon, summarize, save_results
from src.forecasting.lightgbm_model import default_feature_cols
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("sber_index.02")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward валидация прогноза")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Горизонты (по умолчанию из конфига)")
    parser.add_argument("--territories", type=int, default=0, help="Воспроизводимая подвыборка МО")
    parser.add_argument("--quick", action="store_true", help="Проверочный запуск: 100 итераций бустинга")
    parser.add_argument("--output-dir", default=None, help="Отдельная папка результатов")
    parser.add_argument("--models", nargs="+", help="Запустить только перечисленные модели YAML")
    parser.add_argument("--resume", action="store_true", help="Продолжить совпадающий запуск по проверенным checkpoints")
    args = parser.parse_args()
    if args.territories < 0:
        parser.error("--territories must be nonnegative; zero means all")
    if args.horizons and (any(h < 1 for h in args.horizons) or len(set(args.horizons)) != len(args.horizons)):
        parser.error("--horizons must contain unique positive integers")

    data_cfg = load_config("data")
    fc_cfg = load_config("forecasting")
    fc_cfg["use_log"] = data_cfg["transform"]["apply"]
    if args.models:
        unknown = set(args.models) - set(fc_cfg["models"])
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        for name in fc_cfg["models"]:
            fc_cfg["models"][name]["enable"] = name in args.models
        if "pooled_seasonal" not in args.models:
            fc_cfg["selection"]["enable"] = False
    if args.output_dir:
        fc_cfg["output_dir"] = args.output_dir
    if args.quick:
        fc_cfg["validation"]["n_splits_by_horizon"] = {"1": 4}
        fc_cfg["models"]["lightgbm"]["params"]["n_estimators"] = 100
        fc_cfg["models"]["catboost"]["params"]["iterations"] = 100
        fc_cfg["models"]["prophet"]["subsample"] = 8

    features_path = Path(data_cfg["paths"]["processed_dir"]) / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} не найден. Сначала запустите scripts/01_prepare_data.py"
        )
    features = pd.read_parquet(features_path)
    features["date"] = pd.PeriodIndex(features["date"], freq="M")
    if "observed" not in features:
        raise ValueError("Признаки устарели: повторите шаг 01")
    if args.territories:
        import numpy as np
        rng = np.random.default_rng(fc_cfg["validation"].get("seed", 42))
        tids = sorted(features.territory_id.unique())
        selected = rng.choice(tids, min(args.territories, len(tids)), replace=False)
        features = features[features.territory_id.isin(selected)].copy()
    logger.info("Загружены признаки: %d строк, %d колонок",
                len(features), len(features.columns))

    horizons = args.horizons or fc_cfg["horizons"]
    metrics = fc_cfg.get("metrics", ["mae", "rmse", "mape", "r2"])
    from src.forecasting.metrics import METRICS
    if "mae" not in metrics or set(metrics) - set(METRICS):
        raise ValueError("Метрики должны включать MAE и поддерживаемые имена mae/rmse/mape/r2")
    if fc_cfg["validation"].get("strategy", "expanding") != "expanding":
        raise ValueError("Реализована только expanding walk-forward валидация")
    feature_cols = default_feature_cols(features)

    all_results = []
    out_dir = Path(fc_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    from src.utils.config import save_dict_yaml
    from src.evaluation.provenance import reproducibility_metadata
    import platform
    import hashlib
    metadata = {"config": fc_cfg, "territories": int(features.territory_id.nunique()),
                **reproducibility_metadata(Path(__file__).resolve().parents[1]),
                "quick": args.quick, "python": platform.python_version(),
                "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
                "horizons": list(horizons),
                "territory_ids": [int(t) for t in sorted(features.territory_id.unique())],
                "date_min": str(features.date.min()), "date_max": str(features.date.max())}
    if fc_cfg["models"].get("chronos", {}).get("enable"):
        from src.forecasting.chronos_model import model_manifest
        metadata["foundation_model"] = model_manifest(fc_cfg["models"]["chronos"])
    config_path = out_dir / "run_config.yaml"
    checkpoints = {}
    if args.resume:
        import yaml
        if not config_path.exists():
            raise ValueError("Нет манифеста для продолжения запуска")
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in metadata.items()):
            raise ValueError("Данные, код, параметры или окружение изменились; используйте новый output-dir")
        checkpoints = saved.get("checkpoints", {})
    metadata.update(status="running", checkpoints=checkpoints)
    save_dict_yaml(metadata, config_path)
    for h in horizons:
        t0 = time.time()
        checkpoint = out_dir / f"horizon_{h}_checkpoint.csv"
        if args.resume and str(h) in checkpoints:
            if not checkpoint.exists() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoints[str(h)]:
                raise ValueError(f"Checkpoint h={h} повреждён или отсутствует")
            from src.evaluation.audit import validate_forecasts
            res = validate_forecasts(pd.read_csv(checkpoint, low_memory=False))
            if not res.horizon.eq(h).all():
                raise ValueError(f"Checkpoint h={h}: другой горизонт")
            logger.info("Горизонт %d восстановлен из проверенного checkpoint", h)
        else:
            res = evaluate_horizon(features, h, fc_cfg, feature_cols)
        logger.info("Горизонт %d завершён за %.1f c", h, time.time() - t0)
        if not res.empty:
            if not (args.resume and str(h) in checkpoints):
                res.to_csv(checkpoint, index=False)
            checkpoints[str(h)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if fc_cfg["models"].get("chronos", {}).get("enable"):
                metadata["foundation_model"] = model_manifest(fc_cfg["models"]["chronos"])
            save_dict_yaml(metadata, config_path)
            all_results.append(res)

    if not all_results:
        raise RuntimeError("Нет результатов прогноза — проверьте конфиги и ошибки моделей")

    results = pd.concat(all_results, ignore_index=True)
    missing_horizons = set(horizons) - set(results.horizon.unique())
    if missing_horizons:
        raise RuntimeError(f"Не рассчитаны горизонты {sorted(missing_horizons)}; checkpoints сохранены")

    # убираем дубликаты (один и тот же прогноз мог быть посчитан дважды)
    key = ["model", "horizon", "territory_id", "category", "cutoff", "date"]
    if all(c in results.columns for c in key):
        results = results.drop_duplicates(subset=key)
    results = results.sort_values(["horizon", "model", "date"]).reset_index(drop=True)

    summary = summarize(results, metrics)
    out_dir = Path(fc_cfg["output_dir"])
    save_results(results, summary, str(out_dir))
    metadata["status"] = "complete"
    metadata["forecast_results_sha256"] = hashlib.sha256((out_dir / "forecast_results.csv").read_bytes()).hexdigest()
    save_dict_yaml(metadata, config_path)
    print("\n=== СВОДНАЯ ТАБЛИЦА (MAE/R2) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
