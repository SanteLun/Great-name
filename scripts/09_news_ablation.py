"""Matched causal LightGBM ablation with real dated CBR headlines."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.news.pipeline import load_news_csv, score_news, monthly_news_impact, merge_news_features
from src.utils.config import load_config, save_dict_yaml
from src.forecasting.lightgbm_model import predict_lightgbm, default_feature_cols
from src.evaluation.walk_forward import attach_actuals, summarize


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/validated")
    parser.add_argument("--dense", action="store_true", help="All monthly h=1 origins; 80 municipalities, seed 42")
    args = parser.parse_args()
    source = Path(args.source) / "forecast_results.csv"
    if args.dense and not source.exists():
        source = Path("outputs/forecasting/forecast_results.csv")
    saved = pd.read_csv(source)
    saved = saved[(saved.model == "lightgbm") & (saved.horizon == 1)]
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.PeriodIndex(features.date, freq="M")
    import yaml
    configuration = Path(args.source) / "run_config.yaml"
    selected = saved.territory_id.unique()
    if configuration.exists():
        metadata = yaml.safe_load(configuration.read_text(encoding="utf-8"))
        from src.evaluation.provenance import training_territory_ids, verify_features
        selected = training_territory_ids(metadata, saved)
        verify_features("data/processed/features.parquet", metadata)
    features = features[features.territory_id.isin(selected)].copy()
    features = features.drop(columns=[c for c in features if c.startswith("news_")])
    nc = load_config("news")
    news = score_news(load_news_csv("data/news/cbr_releases.csv"), nc)
    monthly = monthly_news_impact(news, nc).reset_index()
    augmented = merge_news_features(features, monthly)
    params = load_config("forecasting")["models"]["lightgbm"]["params"]
    rows = []
    cutoffs = sorted(pd.PeriodIndex(saved.cutoff.unique(), freq="M"))
    if args.dense:
        from src.evaluation.walk_forward import make_cutoffs
        cutoffs = make_cutoffs(sorted(features.date.unique()), 1, 13)
    for name, panel in [("without_news", features), ("with_news", augmented)]:
        for cutoff in cutoffs:
            prediction = predict_lightgbm(panel, cutoff, 1, params, default_feature_cols(panel))
            prediction["cutoff"] = cutoff
            result = attach_actuals(prediction, panel).dropna(subset=["actual", "predicted"])
            result["model"], result["horizon"] = name, 1
            rows.append(result)
    result = pd.concat(rows, ignore_index=True)
    out = Path("outputs/news_ablation")
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "predictions.csv", index=False)
    summary = summarize(result)
    summary.to_csv(out / "summary.csv", index=False)
    news.to_csv(out / "scored_official_news.csv", index=False)
    save_dict_yaml({"mode": "official_cbr_headlines", "n_articles": len(news),
                    "config": nc, "model_params": params,
                    "cutoffs": [str(c) for c in cutoffs], "territories": int(features.territory_id.nunique()),
                    "limitations": "Small monetary-policy corpus; no local coverage; nominal spending direction is not sentiment ground truth."},
                   out / "manifest.yaml")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
