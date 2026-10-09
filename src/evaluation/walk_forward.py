# -*- coding: utf-8 -*-
"""Walk-forward валидация прогнозных моделей на горизонтах 1/3/6/12 мес."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.cleaner import inverse_transform
from src.forecasting.metrics import compute_metrics
from src.forecasting.lightgbm_model import predict_lightgbm, default_feature_cols

logger = logging.getLogger("sber_index.evaluation")

PRED_COLS = ["territory_id", "category", "date", "predicted"]


def make_cutoffs(dates: list[pd.Period], horizon: int,
                 min_train: int, n_splits: int | None = None,
                 step: int = 1) -> list[pd.Period]:
    """Список точек среза для горизонта h.

    Условие: к моменту cutoff накоплено >= min_train месяцев истории и
    cutoff + h <= последняя дата (чтобы был факт для сравнения).
    """
    if horizon < 1 or min_train < 2 or step < 1:
        raise ValueError("horizon/step >= 1, min_train >= 2")
    if n_splits is not None and n_splits < 0:
        raise ValueError("n_splits must be nonnegative")
    dates = sorted(set(dates))
    if not dates:
        return []
    if dates != list(pd.period_range(dates[0], dates[-1], freq="M")):
        raise ValueError("Validation requires a contiguous monthly calendar")
    if len(dates) <= horizon + 1:
        return []
    min_train = min_train - 1
    cutoffs = []
    for i in range(min_train, len(dates) - horizon, step):
        cutoffs.append(dates[i])
    if n_splits and len(cutoffs) > n_splits:
        idx = np.linspace(0, len(cutoffs) - 1, n_splits).round().astype(int)
        cutoffs = [cutoffs[j] for j in np.unique(idx)]
    return cutoffs


def attach_actuals(pred_df: pd.DataFrame, features_df: pd.DataFrame,
                   use_log: bool = True) -> pd.DataFrame:
    """Присоединяет фактические значения и обратную трансформацию."""
    fact = features_df[["date", "territory_id", "category", "consumption"]].copy()
    if "observed" in features_df:
        fact = fact[features_df["observed"]]
    out = pred_df.merge(fact, on=["date", "territory_id", "category"], how="left")
    if use_log:
        for col in ["predicted", "lower", "upper"]:
            if col in out:
                out[col] = inverse_transform(out[col], use_log=True).clip(lower=0)
        out["actual"] = inverse_transform(out["consumption"], use_log=True)
    else:
        out["actual"] = out["consumption"]
    return out.drop(columns=["consumption"])


def run_model_predictions(model_name: str, predict_fn, features_df: pd.DataFrame,
                          cutoffs: list[pd.Period], horizon: int,
                          extra_kwargs: dict | None = None) -> pd.DataFrame:
    """Прогоняет модель по всем точкам среза и возвращает прогнозы + факты."""
    parts = []
    for cutoff in cutoffs:
        try:
            pred = predict_fn(features_df[features_df.date <= cutoff], cutoff, horizon,
                              **(extra_kwargs or {}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Модель %s: сбой на cutoff %s: %s", model_name, cutoff, exc)
            continue
        if pred is None or pred.empty:
            continue
        pred = pred[pred.date == cutoff + horizon].copy()
        pred["cutoff"] = cutoff
        parts.append(attach_actuals(pred, features_df, use_log=True))
    if not parts:
        logger.warning("Модель %s: нет прогнозов для h=%d", model_name, horizon)
        return pd.DataFrame()
    res = pd.concat(parts, ignore_index=True)
    res = res.dropna(subset=["actual", "predicted"])
    res["model"] = model_name
    res["horizon"] = horizon
    logger.info("Модель %s h=%d: пар факт/прогноз %d", model_name, horizon, len(res))
    return res


def _predict_all_models(features_df: pd.DataFrame, cutoffs: list[pd.Period],
                        horizon: int, cfg: dict,
                        lightgbm_feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Сырые прогнозы всех включённых моделей (без фактов).

    Возвращает {model_name: DataFrame(territory_id, category, date, predicted)}.
    """
    raw: dict[str, pd.DataFrame] = {}

    for mode in ["last", "seasonal"]:
        if not cfg["models"].get("naive", {}).get("enable", True):
            continue
        name = f"naive_{mode}"
        parts = []
        for cutoff in cutoffs:
            try:
                pred = _naive_impl(features_df[features_df.date <= cutoff], cutoff, horizon, mode)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: сбой на %s: %s", name, cutoff, exc)
                continue
            if pred is not None and not pred.empty:
                pred = pred[pred.date == cutoff + horizon].copy()
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw[name] = pd.concat(parts, ignore_index=True)

    if cfg["models"].get("prophet", {}).get("enable", True):
        pcfg = dict(cfg["models"]["prophet"])
        pcfg["use_log"] = cfg.get("use_log", True)
        subsample = pcfg.get("subsample", 300)
        parts = []
        for cutoff in cutoffs:
            try:
                pred = _prophet_impl(features_df[features_df.date <= cutoff], cutoff, horizon, pcfg, subsample)
            except PermissionError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("prophet: сбой на %s: %s", cutoff, exc)
                continue
            if pred is not None and not pred.empty:
                pred = pred[pred.date == cutoff + horizon].copy()
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw["prophet"] = pd.concat(parts, ignore_index=True)

    if cfg["models"].get("lightgbm", {}).get("enable", True):
        lcfg = cfg["models"]["lightgbm"]
        params = dict(lcfg["params"])
        params["quantiles"] = lcfg.get("quantiles", {})
        parts = []
        for cutoff in cutoffs:
            try:
                pred = _lightgbm_impl(features_df[features_df.date <= cutoff], cutoff, horizon, params,
                                      lightgbm_feature_cols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("lightgbm: сбой на %s: %s", cutoff, exc)
                continue
            if pred is not None and not pred.empty:
                pred = pred[pred.date == cutoff + horizon].copy()
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw["lightgbm"] = pd.concat(parts, ignore_index=True)

    if cfg["models"].get("catboost", {}).get("enable", True):
        ccfg = cfg["models"]["catboost"]
        parts = []
        for cutoff in cutoffs:
            try:
                pred = _catboost_impl(features_df[features_df.date <= cutoff], cutoff, horizon,
                                      ccfg["params"], lightgbm_feature_cols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("catboost: сбой на %s: %s", cutoff, exc)
                continue
            if pred is not None and not pred.empty:
                pred = pred[pred.date == cutoff + horizon].copy()
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw["catboost"] = pd.concat(parts, ignore_index=True)

    scfg = cfg["models"].get("short_history", {})
    if scfg.get("enable", False):
        from src.forecasting.short_history import predict_short_history
        parts = []
        for cutoff in cutoffs:
            pred = predict_short_history(features_df, cutoff, horizon, scfg)
            if not pred.empty:
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw["short_history"] = pd.concat(parts, ignore_index=True)

    pooled = cfg["models"].get("pooled_seasonal", {})
    if pooled.get("enable", False):
        from src.forecasting.pooled_seasonal import predict_pooled_seasonal
        parts = []
        for cutoff in cutoffs:
            pred = predict_pooled_seasonal(features_df, cutoff, horizon, pooled)
            if not pred.empty:
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw["pooled_seasonal"] = pd.concat(parts, ignore_index=True)

    chcfg = cfg["models"].get("chronos", {})
    if chcfg.get("enable", False):
        from src.forecasting.chronos_model import predict_chronos
        parts = []
        for cutoff in cutoffs:
            pred = predict_chronos(features_df, cutoff, horizon,
                                   {**chcfg, "use_log": cfg.get("use_log", True)})
            if not pred.empty:
                pred["cutoff"] = cutoff
                parts.append(pred)
        if parts:
            raw["chronos"] = pd.concat(parts, ignore_index=True)

    return raw


def evaluate_horizon(features_df: pd.DataFrame, horizon: int,
                     cfg: dict, lightgbm_feature_cols: list[str]) -> pd.DataFrame:
    """Сравнение всех включённых моделей на одном горизонте + ансамбль."""
    dates = sorted(features_df.date.unique())
    overrides = cfg["validation"].get("min_train_by_horizon", {})
    min_train = overrides.get(str(horizon), overrides.get(horizon, cfg["validation"]["min_train_months"]))
    n_splits = cfg["validation"].get("n_splits")
    overrides = cfg["validation"].get("n_splits_by_horizon", {})
    n_splits = overrides.get(str(horizon), overrides.get(horizon, n_splits))
    step = cfg["validation"].get("step", 1)
    cutoffs = make_cutoffs(dates, horizon, min_train, n_splits, step)
    if not cutoffs:
        logger.info("Горизонт %d: нет точек среза", horizon)
        return pd.DataFrame()
    logger.info("Горизонт %d: %d точек среза", horizon, len(cutoffs))

    raw = _predict_all_models(features_df, cutoffs, horizon, cfg,
                              lightgbm_feature_cols)
    if not raw:
        return pd.DataFrame()

    results = []
    for name, pred in raw.items():
        with_fact = attach_actuals(pred, features_df, use_log=cfg.get("use_log", True))
        with_fact = with_fact.dropna(subset=["actual", "predicted"])
        if with_fact.empty:
            continue
        with_fact["model"] = name
        with_fact["horizon"] = horizon
        results.append(with_fact)

    if cfg.get("ensemble", {}).get("enable", True) and len(raw) >= 2:
        from src.forecasting.prequential import prequential_combine
        ecfg = cfg.get("ensemble", {})
        ens = prequential_combine(raw, features_df, cfg.get("use_log", True),
                                  ecfg.get("min_pairs", 20), ecfg.get("shrinkage", 20.),
                                  ecfg.get("min_models", 2))
        if not ens.empty:
            ens = attach_actuals(ens, features_df, cfg.get("use_log", True))
            ens = ens.dropna(subset=["actual", "predicted"])
            ens["model"], ens["horizon"] = "ensemble", horizon
            results.append(ens)

    if not results:
        return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    selection = cfg.get("selection", {})
    if selection.get("enable", False):
        from src.forecasting.selection import causal_select
        selected = causal_select(combined, selection.get("min_pairs", 20),
                                 selection.get("min_origins", 2),
                                 selection.get("warmup_model", "pooled_seasonal"))
        combined = pd.concat([combined, selected], ignore_index=True)
    return combined


# --- обёртки predict-функций с единым интерфейсом ---

def _naive_impl(df, cutoff, horizon, mode):
    from src.forecasting.naive import predict_naive
    return predict_naive(df, cutoff, horizon, mode=mode)


def _prophet_impl(df, cutoff, horizon, params, subsample):
    from src.forecasting.prophet_model import predict_prophet
    return predict_prophet(df, cutoff, horizon, params, subsample=subsample)


def _lightgbm_impl(df, cutoff, horizon, params, feature_cols):
    return predict_lightgbm(df, cutoff, horizon, params, feature_cols=feature_cols)


def _catboost_impl(df, cutoff, horizon, params, feature_cols):
    from src.forecasting.catboost_model import predict_catboost
    return predict_catboost(df, cutoff, horizon, params, feature_cols=feature_cols)


def summarize(results: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Сводная таблица метрик по (model, horizon)."""
    metrics = metrics or ["mae", "rmse", "mape", "r2"]
    if results.empty:
        return pd.DataFrame()
    rows = []
    for (model, horizon), grp in results.groupby(["model", "horizon"]):
        m = compute_metrics(grp["actual"].to_numpy(), grp["predicted"].to_numpy(), metrics)
        m.update({"model": model, "horizon": int(horizon),
                  "n_pairs": len(grp)})
        rows.append(m)
    summary = pd.DataFrame(rows).sort_values(["horizon", metrics[0]])
    return summary[["model", "horizon", "n_pairs"] + metrics]


def save_results(results: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> None:
    """Сохраняет результаты прогноза и сводку в CSV."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "forecast_results.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "forecast_summary.csv"), index=False)
    if not results.empty:
        summarize(common_results(results)).to_csv(os.path.join(out_dir, "forecast_common_summary.csv"), index=False)
    logger.info("Результаты сохранены в %s", out_dir)

def common_results(results: pd.DataFrame) -> pd.DataFrame:
    """Общие серии и даты начала прогноза всех моделей для каждого горизонта."""
    keys = ["territory_id", "category", "cutoff", "date"]
    parts = []
    for _, group in results.groupby("horizon"):
        counts = group.groupby(keys, observed=True).model.nunique()
        common = counts[counts == group.model.nunique()].reset_index()[keys]
        parts.append(group.merge(common, on=keys, validate="many_to_one"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
