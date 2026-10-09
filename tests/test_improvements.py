import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from src.forecasting.prequential import prequential_combine
from src.forecasting.short_history import predict_short_history
from src.changepoint.diffusion import neighbor_signal_score


class ImprovementTests(unittest.TestCase):
    def test_residual_window_keeps_calendar_gap(self):
        from src.changepoint.residuals import residuals_from_forecast
        dates = pd.PeriodIndex(["2023-01", "2023-02", "2023-03", "2023-04", "2024-01"], freq="M")
        fact = pd.DataFrame({"date": dates, "territory_id": 1, "category": "x", "consumption": np.arange(5.)})
        prediction = fact.drop(columns="consumption").assign(predicted=0., model="test", horizon=1)
        result = residuals_from_forecast(prediction, fact, std_window=6)
        self.assertTrue(pd.isna(result.zscore.iloc[-1]))
    def test_interval_calibration_uses_only_already_observed_outcomes(self):
        from src.evaluation.calibration import prequential_intervals
        dates = pd.period_range("2024-01", periods=6, freq="M")
        frame = pd.DataFrame({"date": dates, "cutoff": dates - 1, "territory_id": 1,
                              "category": "x", "model": "test", "horizon": 1,
                              "predicted": 10., "actual": [9., 11., 12., 10., 9., 8.]})
        first = prequential_intervals(frame, min_pairs=2)
        frame.loc[5, "actual"] = 99999.
        changed = prequential_intervals(frame, min_pairs=2)
        pd.testing.assert_series_equal(first.calibrated_upper, changed.calibrated_upper)
        self.assertTrue(pd.isna(first.calibrated_upper.iloc[0]))
        self.assertTrue((first.calibrated_lower.dropna() >= 0).all())
    def test_training_membership_keeps_territory_without_holdout_facts(self):
        from src.evaluation.provenance import training_territory_ids, verify_features
        predictions = pd.DataFrame({"territory_id": [1]})
        self.assertEqual(training_territory_ids({"territory_ids": [1, 2]}, predictions), [1, 2])
        with patch("pathlib.Path.read_bytes", return_value=b"changed"), self.assertRaises(ValueError):
            verify_features("features.parquet", {"features_sha256": "wrong"})
    def test_prophet_extends_forecast_across_missing_latest_observation(self):
        from src.forecasting.prophet_model import _fit_predict_series
        panel = pd.DataFrame({"date": pd.period_range("2024-01", periods=3, freq="M"),
                              "consumption": np.log1p([10., 11., 11.]), "observed": [True, True, False]})
        model = MagicMock()
        model.make_future_dataframe.return_value = pd.DataFrame({"ds": pd.date_range("2024-03-01", periods=2, freq="MS")})
        model.predict.return_value = model.make_future_dataframe.return_value.assign(yhat=12., yhat_lower=9., yhat_upper=15.)
        with patch("src.forecasting.prophet_model.Prophet", return_value=model), patch("pathlib.Path.open"), patch("pathlib.Path.unlink"):
            result = _fit_predict_series(panel, 1, {"seed": 42}, 1, "x")
        model.make_future_dataframe.assert_called_once_with(periods=2, freq="MS")
        self.assertEqual(result[-1]["date"], pd.Period("2024-04"))
    def test_macro_lag_respects_missing_calendar_month(self):
        from src.features.macro import add_macro_features
        panel = pd.DataFrame({"date": pd.PeriodIndex(["2024-02", "2024-03", "2024-04"], freq="M")})
        macro = pd.DataFrame({"date": pd.PeriodIndex(["2024-01", "2024-03"], freq="M"), "key_rate": [10., 20.]})
        result = add_macro_features(panel, macro, [1])
        self.assertEqual(result.macro_key_rate_lag1.iloc[0], 10.)
        self.assertTrue(pd.isna(result.macro_key_rate_lag1.iloc[1]))
        self.assertEqual(result.macro_key_rate_lag1.iloc[2], 20.)
    def test_early_warning_requires_positive_lead_and_unique_event_match(self):
        from src.changepoint.early_warning import evaluate_warnings
        alerts = pd.DataFrame({"territory_id": [1, 1, 2], "date": ["2024-01", "2024-02", "2024-02"]})
        events = pd.DataFrame({"territory_id": [1, 2], "date": ["2024-02", "2024-02"]})
        score = evaluate_warnings(alerts, events)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 2, 1))
        self.assertEqual(score["mean_lead_months"], 1.)
    def test_bocpd_extreme_observations_preserve_probability_mass(self):
        from src.changepoint.detectors import BOCPD
        detector = BOCPD()
        for value in [0., 0., 1000., -1000., 0.]:
            detector.update(value)
            self.assertTrue(np.isfinite(detector.R).all())
            self.assertAlmostEqual(detector.R.sum(), 1.)
    def test_direct_target_uses_calendar_months_with_gaps_and_unsorted_rows(self):
        from src.forecasting.lightgbm_model import _make_target
        panel = pd.DataFrame({"date": pd.PeriodIndex(["2024-03", "2024-01", "2024-04"], freq="M"),
                              "territory_id": 1, "category": "x", "consumption": [30., 10., 40.]})
        result = _make_target(panel, 1)
        self.assertEqual(result.target.iloc[0], 40.)
        self.assertTrue(pd.isna(result.target.iloc[1]))

    def test_granger_rejects_missing_and_corrects_lag_search(self):
        from src.news.causality import granger_test
        self.assertTrue(np.isnan(granger_test(np.array([1., np.nan]), np.ones(2))["p_value"]))
        rng = np.random.default_rng(12)
        result = granger_test(rng.normal(size=50), rng.normal(size=50))
        self.assertGreaterEqual(result["p_value"], result["raw_p_value"])

    def test_pooled_seasonal_learns_shared_shape_without_future(self):
        from src.forecasting.pooled_seasonal import predict_pooled_seasonal
        months = pd.period_range("2023-01", periods=24, freq="M")
        shape = .2 * np.sin(2 * np.pi * np.arange(24) / 12)
        panel = pd.concat([pd.DataFrame({"date": months, "territory_id": tid, "category": "x",
                                        "consumption": 5. + tid + shape, "observed": True})
                           for tid in [1, 2]], ignore_index=True)
        cutoff = pd.Period("2023-12")
        first = predict_pooled_seasonal(panel, cutoff, 1, {"level_window": 1})
        panel.loc[panel.date > cutoff, "consumption"] = 999.
        second = predict_pooled_seasonal(panel, cutoff, 1, {"level_window": 1})
        pd.testing.assert_frame_equal(first, second)
        self.assertAlmostEqual(first.predicted.iloc[0], 6.)

    def test_ensemble_cannot_learn_from_future_outcomes(self):
        dates = pd.period_range("2023-01", periods=8, freq="M")
        fact = pd.DataFrame({"date": dates, "territory_id": 1, "category": "x",
                             "consumption": np.arange(8.), "observed": True})
        keys = fact.iloc[2:].drop(columns=["consumption", "observed"]).copy()
        keys["cutoff"] = keys.date - 2
        raw = {"a": keys.assign(predicted=1.), "b": keys.assign(predicted=9.)}
        original = prequential_combine(raw, fact, use_log=False, min_pairs=1)
        changed = fact.copy()
        changed.loc[changed.date >= dates[5], "consumption"] = 99999.
        altered = prequential_combine(raw, changed, use_log=False, min_pairs=1)
        mask = original.cutoff < dates[5]
        np.testing.assert_allclose(original.loc[mask, "predicted"], altered.loc[mask, "predicted"])
        self.assertEqual(original.weight_history_pairs.iloc[0], 0)

    def test_identical_fallbacks_have_one_vote(self):
        fact = pd.DataFrame({"date": [pd.Period("2024-01")], "territory_id": 1,
                             "category": "x", "consumption": 10.})
        keys = fact.drop(columns="consumption").assign(cutoff=pd.Period("2023-01"))
        raw = {"a": keys.assign(predicted=10.), "b": keys.assign(predicted=10.),
               "c": keys.assign(predicted=20.)}
        result = prequential_combine(raw, fact, use_log=False)
        self.assertEqual(result.predicted.iloc[0], 15.)
        self.assertEqual(len(json.loads(result.ensemble_weights.iloc[0])), 2)

    def test_short_history_ignores_future_and_unobserved(self):
        fact = pd.DataFrame({"date": pd.period_range("2023-01", periods=14, freq="M"),
                             "territory_id": 1, "category": "x", "consumption": 5., "observed": True})
        cutoff = pd.Period("2023-12")
        first = predict_short_history(fact, cutoff, 12)
        fact.loc[fact.date > cutoff, "consumption"] = 999.
        second = predict_short_history(fact, cutoff, 12)
        pd.testing.assert_frame_equal(first, second)
        self.assertAlmostEqual(first.predicted.iloc[0], 5.)
        self.assertEqual(first.date.iloc[0], cutoff + 12)

    def test_spatial_alerts_preserve_category(self):
        signals = pd.DataFrame({"date": [pd.Period("2024-01"), pd.Period("2024-01")],
                                "territory_id": [2, 3], "category": ["food", "cars"], "signal": 1})
        result = neighbor_signal_score(signals, {1: [2, 3]}, window=1)
        feb = result[result.date == pd.Period("2024-02")]
        self.assertTrue(feb.neighbor_shocks.eq(1).all())


if __name__ == "__main__":
    unittest.main()
