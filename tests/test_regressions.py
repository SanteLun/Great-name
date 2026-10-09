"""Regression tests for causal features, forecast origins and shock signals."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.data.cleaner import impute_time, spatial_impute
from src.features.spatial import spatial_lag_values
from src.evaluation.walk_forward import make_cutoffs, attach_actuals, evaluate_horizon, common_results
from src.changepoint.residuals import residuals_from_forecast, standardized_residuals
from src.changepoint.diffusion import neighbor_signal_score
from src.changepoint.detectors import BOCPD, detect_cusum
from src.news.pipeline import monthly_news_impact, merge_news_features
from src.forecasting.ensemble import combine
from src.changepoint.benchmark import match_predictions


class RegressionTests(unittest.TestCase):
    def test_boosting_extrapolates_constant_increment_from_last_value(self):
        from src.forecasting.lightgbm_model import predict_lightgbm
        dates = pd.period_range("2023-01", periods=16, freq="M")
        panel = pd.DataFrame({"date": dates, "territory_id": 1, "category": "x",
                              "consumption": np.arange(16.) + 5, "current_value": np.arange(16.) + 5,
                              "observed": True})
        result = predict_lightgbm(panel, dates[-1], 1,
                                  {"n_estimators": 5, "min_child_samples": 1,
                                   "num_leaves": 4, "n_jobs": 1, "verbosity": -1}, ["current_value"])
        self.assertAlmostEqual(result.predicted.iloc[0], 21.)

    def test_prophet_temp_permission_error_fails_without_retry_loop(self):
        from pathlib import Path
        from src.forecasting.prophet_model import _fit_predict_series
        with patch.object(Path, "open", side_effect=PermissionError("sandbox")) as opening:
            with self.assertRaisesRegex(PermissionError, "CmdStan TEMP"):
                _fit_predict_series(pd.DataFrame(), 1, {}, 1, "x")
        self.assertEqual(opening.call_count, 1)

    def test_ensemble_aligns_keys_and_ignores_empty_models(self):
        a = pd.DataFrame({"territory_id": [1, 2], "category": "x", "date": "2023-01", "predicted": [10., 20.]})
        b = a.iloc[::-1].copy()
        b["predicted"] += 10
        result = combine({"empty": pd.DataFrame(), "a": a, "b": b}, method="simple_average")
        np.testing.assert_allclose(result.predicted, [15., 25.])

    def test_alarm_before_shock_is_not_a_causal_true_positive(self):
        tp, fp, fn, delays = match_predictions(np.array([9]), [10], allow_early=False)
        self.assertEqual((tp, fp, fn), (0, 1, 1))

    def test_imputation_never_reads_future_and_respects_limit(self):
        panel = pd.DataFrame({"date": pd.period_range("2023-01", periods=6, freq="M"),
                              "territory_id": 1, "category": "x",
                              "consumption": [1., np.nan, np.nan, np.nan, np.nan, 99.]})
        filled = impute_time(panel, 1)
        self.assertEqual(filled.consumption.iloc[1], 1.)
        self.assertTrue(filled.consumption.iloc[2:5].isna().all())
        pd.testing.assert_series_equal(filled.consumption.iloc[:5],
                                       impute_time(panel.iloc[:5], 1).consumption)

    def test_neighbor_imputation_uses_other_territories(self):
        panel = pd.DataFrame({"date": [pd.Period("2023-01")] * 3,
                              "territory_id": [1, 2, 3], "category": "x",
                              "consumption": [np.nan, 10., 30.]})
        conn = pd.DataFrame({"territory_id_x": [1, 1], "territory_id_y": [2, 3],
                             "distance": [1., 3.], "type": "highway"})
        self.assertAlmostEqual(spatial_impute(panel, conn, k=2).consumption.iloc[0], 15.)

    def test_missing_graph_nodes_do_not_poison_spatial_lags(self):
        panel = pd.DataFrame({"date": pd.period_range("2023-01", periods=2, freq="M"),
                              "territory_id": 2, "category": "x", "consumption": [10., 20.]})
        w = pd.DataFrame([[0., .5, .5], [1., 0., 0.], [1., 0., 0.]], index=[1, 2, 3], columns=[1, 2, 3])
        lag = spatial_lag_values(panel, w, "x", [1])
        self.assertEqual(lag[(lag.territory_id == 1) & (lag.date == pd.Period("2023-02"))].sp_lag_1.iloc[0], 10.)
        self.assertTrue(lag[lag.date == pd.Period("2023-01")].sp_lag_1.isna().all())

    def test_year_horizon_requires_explicit_short_history_setting(self):
        dates = list(pd.period_range("2023-01", periods=24, freq="M"))
        self.assertEqual(make_cutoffs(dates, 12, 13), [])
        self.assertEqual(make_cutoffs(dates, 12, 12), [pd.Period("2023-12")])

    def test_intervals_and_observed_targets(self):
        fact = pd.DataFrame({"date": [pd.Period("2023-01"), pd.Period("2023-02")],
                             "territory_id": 1, "category": "x",
                             "consumption": np.log1p([10., 20.]), "observed": [True, False]})
        pred = fact.drop(columns=["consumption", "observed"]).copy()
        pred["predicted"], pred["lower"], pred["upper"] = np.log1p(10), np.log1p(8), np.log1p(12)
        result = attach_actuals(pred, fact)
        self.assertAlmostEqual(result.lower.iloc[0], 8.)
        self.assertTrue(pd.isna(result.actual.iloc[1]))

    def test_residual_group_indices_and_causality(self):
        frames = []
        for tid in [1, 2]:
            frames.append(pd.DataFrame({"date": pd.period_range("2023-01", periods=8, freq="M"),
                                        "territory_id": tid, "category": "x",
                                        "consumption": np.arange(8.) + tid}))
        fact = pd.concat(frames, ignore_index=True)
        fc = fact.drop(columns="consumption").copy()
        fc["predicted"], fc["model"], fc["horizon"] = 0., "test", 1
        result = residuals_from_forecast(fc, fact)
        self.assertTrue(result.groupby("territory_id").zscore.count().eq(5).all())
        a = standardized_residuals(np.arange(8.), np.zeros(8))
        b = standardized_residuals(np.r_[np.arange(7.), 999.], np.zeros(8))
        pd.testing.assert_series_equal(a.iloc[:7], b.iloc[:7])

    def test_neighbors_warn_territory_without_own_signal(self):
        signals = pd.DataFrame({"territory_id": [2], "date": [pd.Period("2023-01")], "signal": [1]})
        score = neighbor_signal_score(signals, {1: [2], 2: [1]}, window=2)
        self.assertEqual(score[(score.territory_id == 1) & (score.date == pd.Period("2023-01"))].neighbor_shocks.iloc[0], 0)
        self.assertEqual(score[(score.territory_id == 1) & (score.date == pd.Period("2023-02"))].neighbor_shocks.iloc[0], 1)
        # Extend timeline to February using another municipality's signal.
        signals.loc[1] = [3, pd.Period("2023-02"), 1]
        score = neighbor_signal_score(signals, {1: [2], 2: [1]}, window=2)
        self.assertEqual(score[(score.territory_id == 1) & (score.date == pd.Period("2023-02"))].neighbor_shocks.iloc[0], 1)

    def test_online_detectors_are_prefix_invariant(self):
        y = np.r_[np.zeros(12), np.ones(12) * 3]
        a, p = BOCPD().process(y)
        b, q = BOCPD().process(np.r_[y, np.ones(5) * 99])
        np.testing.assert_allclose(p, q[:len(y)])
        np.testing.assert_array_equal(a, b[b < len(y)])
        np.testing.assert_array_equal(detect_cusum(y), detect_cusum(np.r_[y, 99])[detect_cusum(np.r_[y, 99]) < len(y)])

    def test_news_sign_and_repeat_merge(self):
        cfg = {"topics": {"x": []}, "alignment": {"lag_decay": .3, "max_lag_months": 1}}
        news = pd.DataFrame({"date": ["2023-01-10"], "topic": ["x"], "sentiment": [-1.]})
        monthly = monthly_news_impact(news, cfg)
        self.assertEqual(monthly.news_x_shock.iloc[0], 1.)
        feat = pd.DataFrame({"date": [pd.Period("2023-01")], "territory_id": [1]})
        once = merge_news_features(feat, monthly.reset_index())
        pd.testing.assert_frame_equal(once, merge_news_features(once, monthly.reset_index()))

    def test_exact_horizon_retains_origins(self):
        fact = pd.DataFrame({"date": pd.period_range("2023-01", periods=8, freq="M"),
                             "territory_id": 1, "category": "x", "consumption": np.arange(8.)})
        def predict(hist, cutoff, horizon, mode):
            self.assertLessEqual(hist.date.max(), cutoff)
            return pd.DataFrame({"territory_id": 1, "category": "x",
                                 "date": [cutoff + i for i in range(1, horizon + 1)], "predicted": 1.})
        cfg = {"use_log": False, "validation": {"min_train_months": 3, "n_splits": 3},
               "models": {"naive": {"enable": True}, "prophet": {"enable": False},
                          "lightgbm": {"enable": False}, "catboost": {"enable": False}},
               "ensemble": {"enable": True}}
        with patch("src.evaluation.walk_forward._naive_impl", side_effect=predict):
            result = evaluate_horizon(fact, 3, cfg, [])
        self.assertEqual(len(result), 9)
        self.assertTrue(all(d.ordinal - c.ordinal == 3 for d, c in zip(result.date, result.cutoff)))
        self.assertEqual(len(common_results(result)), 9)


if __name__ == "__main__":
    unittest.main()
