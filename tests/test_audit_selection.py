import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.audit import validate_forecasts, pairwise_skill
from src.forecasting.selection import causal_select


def sample():
    dates = pd.period_range("2024-01", periods=6, freq="M")
    base = pd.DataFrame({"date": dates, "cutoff": dates - 1, "territory_id": 1,
                         "category": "food", "horizon": 1, "actual": 10.})
    return pd.concat([base.assign(model="pooled_seasonal", predicted=12.),
                      base.assign(model="prophet", predicted=10.)], ignore_index=True)


class AuditSelectionTests(unittest.TestCase):
    def test_selector_is_prefix_invariant_and_requires_past_origins(self):
        original = sample()
        first = causal_select(original, min_pairs=1, min_origins=2)
        self.assertEqual(first.selected_model.iloc[:2].tolist(), ["pooled_seasonal"] * 2)
        self.assertEqual(first.selected_model.iloc[2], "prophet")
        changed = original.copy()
        boundary = pd.Period("2024-04")
        changed.loc[changed.date >= boundary, "actual"] = 10000.
        second = causal_select(changed, min_pairs=1, min_origins=2)
        mask = first.cutoff < boundary
        pd.testing.assert_series_equal(first.loc[mask, "predicted"], second.loc[mask, "predicted"])

    def test_selector_long_horizon_does_not_use_pending_outcomes(self):
        frame = sample()
        frame["horizon"] = 6
        frame["date"] = frame.cutoff + 6
        selected = causal_select(frame, min_pairs=1, min_origins=1)
        self.assertTrue(selected.selected_model.eq("pooled_seasonal").all())
        self.assertTrue(selected.selection_history_pairs.eq(0).all())

    def test_audit_rejects_duplicates_wrong_dates_and_inconsistent_facts(self):
        frame = sample()
        validate_forecasts(frame)
        with self.assertRaises(ValueError):
            validate_forecasts(pd.concat([frame, frame.iloc[:1]]))
        changed = frame.copy()
        changed.loc[0, "actual"] = 11.
        with self.assertRaises(ValueError):
            validate_forecasts(changed)
        changed = frame.copy()
        changed.loc[0, "date"] += 1
        with self.assertRaises(ValueError):
            validate_forecasts(changed)

    def test_pairwise_metrics_compare_exactly_matched_rows(self):
        frame = sample()
        frame.loc[(frame.model == "prophet") & (frame.date == frame.date.max()), "predicted"] = 100.
        frame = frame.drop(index=5)
        result = pairwise_skill(frame).iloc[0]
        self.assertEqual(result.n_pairs, 5)
        self.assertEqual(result.baseline_mae_same_pairs, 0.)

    def test_lags_and_diffs_respect_calendar_gaps_and_zero_denominators(self):
        from src.features.lags import build_lags, build_diffs
        frame = pd.DataFrame({"date": pd.PeriodIndex(["2024-01", "2024-03", "2024-04"], freq="M"),
                              "territory_id": 1, "category": "x", "consumption": [10., 0., 20.]})
        out = build_diffs(build_lags(frame, [1]), [1])
        self.assertTrue(pd.isna(out.lag_1.iloc[1]))
        self.assertEqual(out.diff_1.iloc[2], 20.)
        self.assertTrue(pd.isna(out.diff_1_rel.iloc[2]))

    def test_future_prediction_passes_parameters_and_quantiles(self):
        spec = importlib.util.spec_from_file_location("future", Path(__file__).resolve().parents[1] / "scripts/11_forecast_future.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = {"models": {"pooled_seasonal": {"level_window": 7},
                          "lightgbm": {"params": {"n_estimators": 3}, "quantiles": {"enable": True}}}}
        frame = pd.DataFrame({"date": [pd.Period("2024-01")]})
        with patch("src.forecasting.pooled_seasonal.predict_pooled_seasonal") as predict:
            module.predict_selected(frame, frame.date.iloc[0], 1, "pooled_seasonal", cfg)
            self.assertEqual(predict.call_args.args[-1]["level_window"], 7)
        with patch.object(module, "predict_lightgbm") as predict:
            module.predict_selected(frame, frame.date.iloc[0], 1, "lightgbm", cfg)
            self.assertTrue(predict.call_args.args[3]["quantiles"]["enable"])

    def test_rolling_preserves_calendar_window_and_trend_gap_positions(self):
        from src.features.lags import build_rolling
        frame = pd.DataFrame({"date": pd.PeriodIndex(["2024-01", "2024-03", "2024-04"], freq="M"),
                              "territory_id": 1, "category": "x", "consumption": [1., 3., 4.]})
        out = build_rolling(frame, [1, 3], ["mean", "trend"])
        self.assertTrue(pd.isna(out.roll_1_mean.iloc[1]))
        self.assertEqual(out.roll_3_mean.iloc[2], 2.)
        self.assertAlmostEqual(out.roll_3_trend.iloc[2], 1.)

    def test_metrics_reject_nonfinite_and_mismatched_inputs(self):
        from src.forecasting.metrics import compute_metrics
        with self.assertRaises(ValueError):
            compute_metrics([1., 2.], [1.])
        with self.assertRaises(ValueError):
            compute_metrics([1.], [np.inf])

    def test_insufficient_conformal_quantile_does_not_issue_narrow_interval(self):
        from src.evaluation.calibration import prequential_intervals
        out = prequential_intervals(sample().query("model == 'prophet'"), alpha=.01, min_pairs=1)
        self.assertTrue(out.calibrated_upper.isna().all())

    def test_cutoff_validation_handles_invalid_short_inputs(self):
        from src.evaluation.walk_forward import make_cutoffs
        with self.assertRaises(ValueError):
            make_cutoffs([], 0, 2)
        with self.assertRaises(ValueError):
            make_cutoffs(list(pd.PeriodIndex(["2024-01", "2024-03"], freq="M")), 1, 2)

    def test_warning_events_with_repeated_dataframe_index_match_independently(self):
        from src.changepoint.early_warning import evaluate_warnings
        alerts = pd.DataFrame({"territory_id": [1, 2, 2], "date": ["2024-01"] * 3})
        events = pd.DataFrame({"territory_id": [1, 2], "date": ["2024-02"] * 2}, index=[0, 0])
        result = evaluate_warnings(alerts, events)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))

    def test_categorical_encoding_retains_training_order_at_prediction(self):
        from src.forecasting.lightgbm_model import _encode_features
        train = pd.DataFrame({"label": ["b", "a"]})
        encoded, mapping = _encode_features(train, ["label"])
        point = pd.DataFrame({"label": pd.Categorical(["a", "b", "unknown"], categories=["unknown", "b", "a"])})
        predicted, _ = _encode_features(point, ["label"], mapping)
        self.assertEqual(encoded.label.tolist(), [1, 0])
        self.assertEqual(predicted.label.tolist(), [0, 1, -1])

    def test_subsampled_model_does_not_limit_ensemble_coverage(self):
        from src.forecasting.prequential import prequential_combine
        fact = pd.DataFrame({"date": [pd.Period("2024-01")] * 2, "territory_id": [1, 2],
                             "category": "x", "consumption": [10., 20.]})
        keys = fact.drop(columns="consumption").assign(cutoff=pd.Period("2023-12"))
        raw = {"a": keys.assign(predicted=[10., 20.]), "b": keys.assign(predicted=[12., 22.]),
               "sampled": keys.iloc[:1].assign(predicted=14.)}
        result = prequential_combine(raw, fact, use_log=False)
        self.assertEqual(len(result), 2)
        result = result.set_index("territory_id")
        self.assertEqual(result.loc[1, "predicted"], 12.)
        self.assertEqual(result.loc[2, "predicted"], 21.)
        self.assertEqual(result.loc[2, "ensemble_n_models"], 2)

    def test_incomplete_runs_are_not_used_as_completed_evidence(self):
        from src.evaluation.provenance import verify_features
        with self.assertRaises(ValueError):
            verify_features("unused", {"status": "running"})

    def test_missing_prophet_exports_readable_comparison_schema(self):
        result = pairwise_skill(sample().query("model == 'pooled_seasonal'"))
        self.assertTrue(result.empty)
        self.assertIn("baseline_mae_same_pairs", result.columns)

    def test_yaml_numeric_horizon_override_is_used(self):
        from src.evaluation.walk_forward import evaluate_horizon
        frame = pd.DataFrame({"date": pd.period_range("2023-01", periods=24, freq="M")})
        cfg = {"validation": {"min_train_months": 13, "min_train_by_horizon": {12: 12}}, "models": {}}
        with patch("src.evaluation.walk_forward._predict_all_models", return_value={}) as predict:
            evaluate_horizon(frame, 12, cfg, [])
        self.assertEqual(predict.call_args.args[1], [pd.Period("2023-12")])


if __name__ == "__main__":
    unittest.main()
