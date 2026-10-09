# -*- coding: utf-8 -*-
"""Конвейер новостных признаков и согласование с данными СберИндекса.

Ключевая идея согласования (из SOLUTION.md):
1. Дневные новости агрегируются в месячные индексы (интенсивность x тональность).
2. Эффект новости распределяется по месяцам с экспоненциальным затуханием
   (News Impact Curve): w(lag) = exp(-lambda * lag).
3. Общероссийский индекс масштабируется на каждое МО через региональную
   эластичность (market_access): в более «рыночных» регионах эффект сильнее.
4. Признак становится доступным только после публикации новости; будущая
   дата события не переносит информацию в прошлое.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.news.sentiment import classify_topic, lexicon_sentiment

logger = logging.getLogger("sber_index.news")


def load_news_csv(path: str) -> pd.DataFrame:
    """Загрузка новостей из CSV: колонки date (YYYY-MM-DD), title, text (опц.)."""
    df = pd.read_csv(path, parse_dates=["date"])
    if "text" not in df:
        df["text"] = df["title"]
    df["text"] = df["text"].fillna(df["title"]).fillna("")
    df = df.drop_duplicates(subset=["date", "text"])
    return df


def score_news(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Добавляет тональность и тему каждой новости."""
    out = df.copy()
    out["sentiment"] = out["text"].apply(lexicon_sentiment)
    out["topic"] = out["text"].apply(lambda t: classify_topic(t, cfg["topics"]))
    out = out.dropna(subset=["topic"])
    return out


def monthly_news_impact(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Месячные индексы новостей по темам с учётом экспоненциальных лагов.

    Возвращает DataFrame: index = Period('M'), колонки news_{topic}_int,
    news_{topic}_sent, news_{topic}_shock (int * |sent| с отрицательным знаком
    для негативных).
    """
    lag_decay = cfg["alignment"]["lag_decay"]
    max_lag = cfg["alignment"]["max_lag_months"]

    daily = df.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    # Дата доступности информации совпадает с датой публикации.
    daily["event_date"] = daily["date"]

    topics = list(cfg["topics"].keys())
    rows = {}
    for _, r in daily.iterrows():
        month = pd.Period(r["event_date"], freq="M")
        for lag in range(0, max_lag + 1):
            m = month + lag
            key = m
            if key not in rows:
                rows[key] = {t: {"int": 0.0, "sent": 0.0, "n": 0} for t in topics}
            w = np.exp(-lag_decay * lag)
            t = r["topic"]
            rows[key][t]["int"] += w
            rows[key][t]["sent"] += w * r["sentiment"]
            rows[key][t]["n"] += 1

    idx = pd.PeriodIndex(sorted(rows.keys()), freq="M")
    cols = {}
    for t in topics:
        vals_int = np.array([rows[m][t]["int"] for m in idx])
        vals_sent = np.array([rows[m][t]["sent"] for m in idx])
        n = np.array([rows[m][t]["n"] for m in idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sent = np.divide(vals_sent, vals_int, out=np.zeros_like(vals_sent), where=vals_int > 0)
        cols[f"news_{t}_int"] = vals_int
        cols[f"news_{t}_sent"] = mean_sent
        cols[f"news_{t}_shock"] = -vals_sent  # негатив => положительный сигнал риска
    res = pd.DataFrame(cols, index=idx)
    res.index.name = "date"
    return res


def align_to_mo(monthly: pd.DataFrame, market_access: pd.DataFrame | None,
                cfg: dict) -> pd.DataFrame:
    """Масштабирование общероссийских новостных индексов на МО.

    Эластичность региона моделируется нормированным market_access:
    эффект_МО = эффект_РФ * (1 + b * (market_access - mean)/std).
    Без market_access эластичность = 1 для всех МО.
    """
    if market_access is None:
        return monthly.reset_index()

    ma = market_access[["territory_id", "market_access"]].copy()
    mu = ma.market_access.mean()
    sd = max(ma.market_access.std(), 1e-6)
    b = 0.25  # коэффициент эластичности
    ma["elasticity"] = 1.0 + b * (ma.market_access - mu) / sd
    ma["elasticity"] = ma["elasticity"].clip(lower=0.1).fillna(1.0)
    if not cfg["alignment"].get("regional_elasticity", True):
        ma["elasticity"] = 1.0

    # месяц как колонка (merge сбросил бы индекс)
    monthly2 = monthly.reset_index()
    res = monthly2.merge(ma[["territory_id", "elasticity"]], how="cross")
    feat_cols = [c for c in monthly.columns if c.startswith("news_")]
    for c in feat_cols:
        res[c] = res[c] * res["elasticity"]
    return res.drop(columns=["elasticity"])


def merge_news_features(features_df: pd.DataFrame, news_mo: pd.DataFrame) -> pd.DataFrame:
    """Встраивает новостные признаки в панель признаков (merge по дате и территории)."""
    out = features_df.drop(columns=[c for c in features_df if c.startswith("news_")]).copy()
    nf = news_mo.copy()
    if "date" not in nf:
        nf = nf.reset_index()
    nf["date"] = pd.PeriodIndex(nf["date"], freq="M")
    out["date"] = pd.PeriodIndex(out["date"], freq="M")
    key = ["date", "territory_id"] if "territory_id" in nf else ["date"]
    out = out.merge(nf, on=key, how="left", validate="many_to_one")
    cols = [c for c in out if c.startswith("news_")]
    out[cols] = out[cols].fillna(0.0)
    logger.info("Новостные признаки добавлены: %d колонок",
                len([c for c in out.columns if c.startswith("news_")]))
    return out
