# -*- coding: utf-8 -*-
"""Granger-причинность: объясняют ли новостные индексы прогнозные остатки.

Классический подход: сравниваем ограниченную модель остатков (только свои лаги)
с полной моделью (свои лаги + лаги новостного индекса). Значимый F-тест (p<0.05)
означает, что новости помогают объяснять отклонения фактических расходов
от прогноза — т.е. новости содержат сигнал о будущих шоках.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger("sber_index.news")


def granger_test(y: np.ndarray, x: np.ndarray, maxlag: int = 3) -> dict:
    """Granger-тест: x -> y.

    Возвращает dict с f-статистикой, p-значением и df.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = min(len(y), len(x))
    y, x = y[:n], x[:n]
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return {"f_stat": np.nan, "p_value": np.nan, "best_lag": 0}

    def _build_lags(arr: np.ndarray, p: int) -> np.ndarray:
        return np.column_stack([arr[p - 1 - i: n - 1 - i] for i in range(p)])

    results = {}
    for p in range(1, maxlag + 1):
        if n <= p + 2:
            continue
        yy = y[p:]
        Xr = _build_lags(y, p)
        Xr = sm.add_constant(Xr)
        rss_r = float(np.sum((yy - sm.OLS(yy, Xr).fit().predict(Xr)) ** 2))

        Xf = np.column_stack([Xr, _build_lags(x, p)])
        fit_f = sm.OLS(yy, Xf).fit()
        rss_f = float(np.sum((yy - fit_f.predict(Xf)) ** 2))
        df1 = p
        df2 = len(yy) - Xf.shape[1]
        if rss_f <= 0 or df2 < 5 or np.linalg.matrix_rank(Xf) < Xf.shape[1]:
            continue
        f_stat = max(0., ((rss_r - rss_f) / df1) / (rss_f / df2))
        from scipy.stats import f as f_dist
        p_value = float(1.0 - f_dist.cdf(f_stat, df1, df2))
        results[p] = {"f_stat": f_stat, "p_value": p_value, "df1": df1, "df2": df2}

    if not results:
        return {"f_stat": np.nan, "p_value": np.nan, "best_lag": 0}
    best = min(results.items(), key=lambda kv: kv[1]["p_value"])
    return {**best[1], "raw_p_value": best[1]["p_value"],
            "p_value": min(1., best[1]["p_value"] * len(results)), "best_lag": best[0]}


def test_news_on_residuals(residuals: pd.DataFrame, news_monthly: pd.DataFrame,
                           news_col: str, maxlag: int = 3,
                           n_series_sample: int = 50) -> dict:
    """Granger-тест новостного индекса на остатках по выборке серий.

    residuals: DataFrame(territory_id, category, date, residual).
    news_monthly: DataFrame(date, news_*).
    Возвращает сводку по всем протестированным сериям.
    """
    if residuals.empty or news_monthly is None or news_monthly.empty:
        return {"n_tested": 0, "share_significant": np.nan, "avg_p": np.nan, "median_p": np.nan}

    nidx = pd.PeriodIndex(news_monthly["date"], freq="M")
    news_series = pd.Series(news_monthly[news_col].to_numpy(), index=nidx)

    keys = residuals.groupby(["territory_id", "category"]).size()
    keys = keys.sample(min(len(keys), n_series_sample), random_state=42)

    pvals = []
    for (tid, cat) in keys.index:
        sub = (residuals[(residuals.territory_id == tid) &
                         (residuals.category == cat)]
               .sort_values("date"))
        y = sub["residual"].to_numpy()
        idx = sub["date"]
        x = np.array([news_series.get(d, np.nan) for d in idx], dtype=float)
        dates = pd.PeriodIndex(idx, freq="M")
        if len(dates) < 10 or any(np.diff(dates.asi8) != 1):
            continue
        res = granger_test(y, x, maxlag=maxlag)
        if np.isfinite(res["p_value"]):
            pvals.append(res["p_value"])

    if not pvals:
        return {"n_tested": 0, "share_significant": np.nan, "avg_p": np.nan, "median_p": np.nan}
    # Benjamini-Hochberg correction across tested municipality/category series.
    ordered = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    ranked = np.asarray(pvals)[ordered] * len(pvals) / np.arange(1, len(pvals) + 1)
    adjusted[ordered] = np.minimum(1., np.minimum.accumulate(ranked[::-1])[::-1])
    return {
        "n_tested": len(pvals),
        "share_significant": float(np.mean(adjusted < 0.05)),
        "correction": "Bonferroni across lags; Benjamini-Hochberg across series",
        "avg_p": float(np.mean(pvals)),
        "median_p": float(np.median(pvals)),
    }
