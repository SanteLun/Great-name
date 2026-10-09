# -*- coding: utf-8 -*-
"""Глобальная градиентная модель LightGBM (обучается на всех сериях сразу).

Подход: direct multi-horizon — для каждого горизонта h обучается отдельная модель,
которая по признакам месяца t предсказывает consumption[t+h]. При прогнозе из точки
cutoff берётся строка с date == cutoff, модель даёт прогноз на cutoff+h.
Это исключает накопление ошибки (в отличие от рекурсивного прогноза) и корректно
использует только прошлые значения в признаках.

Ограничение: при короткой истории (24 мес) для длинных горизонтов (h ~ 12) в train
не остаётся пар «признаки_t → цель_{t+h}». В этом случае явно помечается
сезонный fallback: его нельзя интерпретировать как качество обученной ML-модели.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger("sber_index.forecasting")

# колонки-идентификаторы, не являющиеся признаками
ID_COLS = ["date", "territory_id", "category", "consumption", "observed",
           "cutoff", "target", "target_observed"]


def default_feature_cols(df: pd.DataFrame) -> list[str]:
    """Все колонки панели, кроме идентификаторов и цели."""
    return [c for c in df.columns if c not in ID_COLS]


def _encode_features(df, feature_cols, encoders=None):
    """Freeze training category order; unseen prediction labels receive -1."""
    X = df[feature_cols].copy()
    mappings = dict(encoders or {})
    for column in feature_cols:
        if (column in mappings or isinstance(X[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(X[column].dtype)):
            if column not in mappings:
                mappings[column] = (list(X[column].cat.categories) if isinstance(X[column].dtype, pd.CategoricalDtype)
                                    else list(pd.Index(X[column].dropna().unique()).sort_values()))
            labels = X[column].astype(object).where(X[column].isin(mappings[column]))
            X[column] = pd.Categorical(labels, categories=mappings[column]).codes
    return X, mappings


def _make_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Целевая переменная direct-прогноза: consumption[t+h]."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    out = df.copy()
    keys = ["territory_id", "category", "date"]
    if out.duplicated(keys).any():
        raise ValueError("Duplicate monthly series keys")
    dates = pd.PeriodIndex(out.date, freq="M")
    lookup = out.assign(date=dates).set_index(keys)
    future = pd.MultiIndex.from_arrays([out.territory_id, out.category, dates + horizon], names=keys)
    out["target"] = lookup.consumption.reindex(future).to_numpy()
    if "observed" in out:
        observed = lookup.observed.reindex(future).fillna(False).astype(bool).to_numpy()
        out["target_observed"] = observed
        out.loc[~observed, "target"] = np.nan
    return out


def fit_horizon_model(train: pd.DataFrame, horizon: int, params: dict,
                      feature_cols: list[str]) -> lgb.LGBMRegressor:
    """Обучает модель для одного горизонта на train-панели (date <= cutoff).

    NaN в признаках оставляем как есть: LightGBM корректно обрабатывает пропуски,
    а требование полного отсутствия NaN убивало бы первые ~12 месяцев истории.
    """
    data = _make_target(train, horizon)
    data = data.dropna(subset=["target"])

    X, encoders = _encode_features(data, feature_cols)
    y = (data["target"] - data["consumption"]).to_numpy(dtype=float)
    cat_features = list(encoders)

    p = dict(params)
    p.pop("early_stopping_rounds", None)
    p.pop("quantiles", None)   # служебный параметр доверительных интервалов
    model = lgb.LGBMRegressor(**p)
    if len(X) == 0:
        raise ValueError("Нет обучающих строк для горизонта %d" % horizon)
    model.fit(X, y, categorical_feature=cat_features)
    model.feature_encoders_ = encoders
    logger.info("LightGBM h=%d: обучено на %d строках, %d признаков",
                horizon, len(X), len(feature_cols))
    return model


def _prepare_X(df: pd.DataFrame, feature_cols: list[str], encoders=None) -> pd.DataFrame:
    """Подготовка матрицы признаков (категориальные -> коды)."""
    return _encode_features(df, feature_cols, encoders)[0]


def _fit_quantile_model(train: pd.DataFrame, horizon: int, params: dict,
                        feature_cols: list[str], alpha: float) -> lgb.LGBMRegressor:
    """Квантильная модель LightGBM для доверительных интервалов."""
    p = dict(params)
    p.pop("quantiles", None)
    p.pop("early_stopping_rounds", None)
    p["objective"] = "quantile"
    p["alpha"] = alpha
    data = _make_target(train, horizon)
    data = data.dropna(subset=["target"])
    X, encoders = _encode_features(data, feature_cols)
    y = (data["target"] - data["consumption"]).to_numpy(dtype=float)
    cat_features = list(encoders)
    model = lgb.LGBMRegressor(**p)
    model.fit(X, y, categorical_feature=cat_features)
    model.feature_encoders_ = encoders
    return model


def predict_lightgbm(features_df: pd.DataFrame, cutoff: pd.Period,
                     horizon: int, params: dict,
                     feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Прогноз из точки cutoff ровно на horizon месяцев (все серии разом).

    Для горизонтов, где direct-обучение невозможно (короткая история),
    используется явно помеченный сезонный fallback.
    """
    feature_cols = feature_cols or default_feature_cols(features_df)
    train = features_df[features_df.date <= cutoff]
    point = features_df[features_df.date == cutoff]
    if point.empty:
        return pd.DataFrame(columns=["territory_id", "category", "date", "predicted"])

    # Отдельная прямая модель для запрошенного горизонта.
    qcfg = params.get("quantiles") or {}
    q_enable = bool(qcfg.get("enable", False))
    alphas = sorted(qcfg.get("alphas", [0.1, 0.5, 0.9]))

    rows = []
    for h in [horizon]:
        try:
            direct = fit_horizon_model(train, h, params, feature_cols)
        except ValueError:
            if _make_target(train, h).target.notna().any():
                raise
            from src.forecasting.naive import predict_naive
            fallback = predict_naive(train, cutoff, h, mode="seasonal")
            fallback = fallback[fallback.date == cutoff + h].copy()
            fallback["fallback"] = "seasonal_no_training_pairs"
            return fallback
        Xp = _prepare_X(point, feature_cols, getattr(direct, "feature_encoders_", None))
        anchor = point["consumption"].to_numpy(dtype=float)
        pred = anchor + direct.predict(Xp)

        lower = upper = None
        if q_enable and direct is not None:
            lo = _fit_quantile_model(train, h, params, feature_cols, alphas[0])
            hi = _fit_quantile_model(train, h, params, feature_cols, alphas[-1])
            lower = anchor + lo.predict(Xp)
            upper = anchor + hi.predict(Xp)

        tmp = point[["territory_id", "category"]].copy()
        tmp["date"] = cutoff + h
        tmp["predicted"] = pred
        if lower is not None:
            tmp["lower"] = np.minimum(lower, upper)
            tmp["upper"] = np.maximum(lower, upper)
        rows.append(tmp)

    if not rows:
        logger.warning("LightGBM h<=%d cutoff=%s: нет прогнозов", horizon, cutoff)
        return pd.DataFrame(columns=["territory_id", "category", "date", "predicted"])
    res = pd.concat(rows, ignore_index=True)
    logger.info("LightGBM h<=%d cutoff=%s: прогнозов %d", horizon, cutoff, len(res))
    return res
