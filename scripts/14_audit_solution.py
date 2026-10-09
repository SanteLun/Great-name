"""Audit saved forecasts and evaluate causal model selection without retraining."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.evaluation.audit import validate_forecasts, coverage_table, pairwise_skill
from src.evaluation.walk_forward import save_results, summarize, common_results
from src.forecasting.selection import causal_select
from src.utils.config import load_config, save_dict_yaml
from src.evaluation.provenance import reproducibility_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/final")
    parser.add_argument("--output-dir", default="outputs/audited")
    args = parser.parse_args()
    source = Path(args.source) / "forecast_results.csv"
    output = Path(args.output_dir)
    if source.parent.resolve() == output.resolve():
        parser.error("Audit output must differ from source to preserve original evidence")
    import yaml
    from src.evaluation.provenance import verify_features
    for name in ["run_config.yaml", "baseline_run_config.yaml", "evaluation_manifest.yaml"]:
        path = source.parent / name
        if path.exists():
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
            verify_features("data/processed/features.parquet", manifest)
            expected = manifest.get("forecast_results_sha256") if name == "run_config.yaml" else None
            if expected and hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("Forecast results differ from their run manifest")
    results = validate_forecasts(pd.read_csv(source, low_memory=False))
    results = results[results.model != "causal_selector"]
    cfg = load_config("forecasting").get("selection", {})
    selected = causal_select(results, cfg.get("min_pairs", 20), cfg.get("min_origins", 2),
                             cfg.get("warmup_model", "pooled_seasonal"))
    combined = validate_forecasts(pd.concat([results, selected], ignore_index=True))
    output.mkdir(parents=True, exist_ok=True)
    save_results(combined, summarize(combined), str(output))
    panel_common = summarize(common_results(combined[combined.model != "prophet"]))
    panel_common.to_csv(output / "forecast_panel_common_summary.csv", index=False)
    coverage = coverage_table(combined)
    coverage.to_csv(output / "coverage_audit.csv", index=False)
    skill = pairwise_skill(combined)
    skill.to_csv(output / "pairwise_vs_prophet.csv", index=False)
    selected.groupby(["horizon", "category", "selected_model", "selection_reason"], observed=True).size().rename("n_pairs").reset_index().to_csv(output / "selection_history.csv", index=False)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("improve_evaluation", Path(__file__).with_name("06_improve_evaluation.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.paired_bootstrap(combined).to_csv(output / "paired_bootstrap.csv", index=False)
    save_dict_yaml({"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "reproducibility": reproducibility_metadata(Path(__file__).resolve().parents[1]),
                    "selection": cfg, "validation": "finite, unique, exact horizons, consistent actuals",
                    "note": "Selection protocol added after existing experiments: use an independent future holdout for confirmation."},
                   output / "audit_manifest.yaml")
    import shutil
    for name in ["run_config.yaml", "baseline_run_config.yaml", "evaluation_manifest.yaml",
                 "refreshed_prophet_config.yaml", "refreshed_lightgbm_config.yaml"]:
        path = source.parent / name
        if path.exists():
            shutil.copy2(path, output / name)
            if name == "run_config.yaml":
                manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
                manifest["parent_forecast_results_sha256"] = manifest.get("forecast_results_sha256")
                manifest["forecast_results_sha256"] = hashlib.sha256((output / "forecast_results.csv").read_bytes()).hexdigest()
                save_dict_yaml(manifest, output / name)
        else:
            (output / name).unlink(missing_ok=True)
    summary = summarize(common_results(combined))
    lines = ["# Аудит решения для трека прогнозирования", "",
             "Проверены уникальность ключей, точные горизонты, конечность прогнозов и согласованность фактов между моделями.",
             "", "## Сравнение с Prophet на одинаковых наблюдениях", "",
             "Положительный MAE skill означает снижение ошибки относительно Prophet. Покрытие указано относительно его прогнозов.", "",
             "| Модель | Горизонт | Пар | MAE | MAE Prophet | Улучшение, % | Покрытие Prophet |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in skill.itertuples():
        lines.append(f"| {row.model} | {row.horizon} | {row.n_pairs} | {row.mae:.2f} | {row.baseline_mae_same_pairs:.2f} | {row.mae_skill_pct:.2f} | {row.baseline_pair_coverage:.1%} |")
    if skill.empty:
        lines += ["", "В этом запуске нет пар для сравнения с Prophet; преимущество над ним не измерено."]
    lines += ["", "## Проверяемый выбор модели", "",
              "causal_selector выбирает модель отдельно для категории и горизонта по MAE только прошлых прогнозов, факты которых уже известны к cutoff. Для обучения выбора нужны минимум 20 общих пар и 2 прошлые точки старта. До этого используется заранее фиксированная pooled_seasonal. Недостающие прогнозы выбранной модели явно заменяются прогнозом pooled_seasonal.",
              "", "Выбор победителя по всей таблице теста остаётся ретроспективным ориентиром. Протокол селектора добавлен после предыдущих экспериментов, поэтому для подтверждения потребуется независимый будущий holdout.",
              "", "## Ограничения и подготовка к конкурсу", "",
              "- История 24 месяца: годовой горизонт имеет одну точку старта; direct ML использует резервный прогноз при отсутствии обучающих пар.",
              "- Prophet покрывает подвыборку серий. coverage_audit.csv отдельно показывает покрытие, количество серий и origins, среднюю и медианную MAE по сериям и долю резервных прогнозов.",
              "- Общий R² зависит от различий масштаба МО; его нужно читать вместе с MAE и ошибками отдельных серий.",
              "- cutoff трактуется как конец месяца, когда его расходы уже известны. Реальные даты публикации исходной панели не предоставлены: промышленный прогноз требует отдельно учесть этот лаг доступности.",
              "- paired_bootstrap.csv содержит интервалы разницы MAE с Prophet при кластеризации по МО/категории; он не устраняет эффект выбора протокола после экспериментов.",
              "- Оценка детекторов на синтетике не подтверждает предупреждение реальных будущих шоков. Нужна независимая датированная разметка событий.",
              "- Новостной корпус решений ЦБ узкий: нужны дополнительные исторические региональные источники с датами публикации.",
              "- forecast_panel_common_summary.csv сравнивает модели полной панели на одинаковых фактах. Prophet сравнивается отдельно на пересечении его сохранённой выборки.",
              "- Перед подачей проверьте состав МО и моделей по манифестам и приложите отчёт именно выбранного итогового прогона.", "",
              f"Источник: `{source}`. Все исходные результаты сохранены без изменений."]
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
