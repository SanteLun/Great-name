# -*- coding: utf-8 -*-
"""Глобальная модель CatBoost (direct multi-horizon, как LightGBM).

CatBoost нативно работает с категориальными признаками (territory_id_cat,
category_cat) и пропусками, что упрощает пайплайн и часто улучшает качество
на табличных данных с категориями.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.forecasting.lightgbm_model import ID_COLS, _make_target

logger = logging.getLogger("sber_index.forecasting")

CAT_FEATURES = ["territory_id_cat", "category_cat"]


def _default_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


def _fit_horizon(train: pd.DataFrame, horizon: int, params: dict,
                 feature_cols: list[str]) -> CatBoostRegressor:
    data = _make_target(train, horizon)
    data = data.dropna(subset=["target"])
    if data.empty:
        raise ValueError("No training pairs")
    X = data[feature_cols].copy()
    y = (data["target"] - data["consumption"]).to_numpy(dtype=float)
    cat_idx = [feature_cols.index(c) for c in CAT_FEATURES
               if c in feature_cols]
    p = dict(params)
    p.pop("early_stopping_rounds", None)
    p.pop("quantiles", None)
    p.setdefault("thread_count", 4)
    p.setdefault("verbose", 0)
    p.setdefault("allow_writing_files", False)
    model = CatBoostRegressor(**p)
    model.fit(X, y, cat_features=cat_idx if cat_idx else None)
    return model


def predict_catboost(features_df: pd.DataFrame, cutoff: pd.Period,
                     horizon: int, params: dict,
                     feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or _default_feature_cols(features_df)
    train = features_df[features_df.date <= cutoff]
    point = features_df[features_df.date == cutoff]
    if point.empty:
        return pd.DataFrame(columns=["territory_id", "category", "date",
                                     "predicted"])

    rows = []
    for h in [horizon]:
        try:
            model = _fit_horizon(train, h, params, feature_cols)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CatBoost h=%d: %s", h, exc)
            if _make_target(train, h)["target"].notna().sum() == 0:
                from src.forecasting.naive import predict_naive
                fallback = predict_naive(train, cutoff, h, mode="seasonal")
                fallback = fallback[fallback.date == cutoff + h].copy()
                fallback["fallback"] = "seasonal_no_training_pairs"
                return fallback
            raise
        Xp = point[feature_cols].copy()
        pred = point["consumption"].to_numpy(dtype=float) + model.predict(Xp)
        tmp = point[["territory_id", "category"]].copy()
        tmp["date"] = cutoff + h
        tmp["predicted"] = pred
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["territory_id", "category", "date",
                                     "predicted"])
    res = pd.concat(rows, ignore_index=True)
    logger.info("CatBoost h<=%d cutoff=%s: прогнозов %d", horizon, cutoff, len(res))
    return res
