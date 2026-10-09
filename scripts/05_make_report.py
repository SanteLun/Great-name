# -*- coding: utf-8 -*-
"""Шаг 5. Итоговый отчёт и визуализация результатов.

Читает outputs/forecasting/forecast_summary.csv и
outputs/changepoint/benchmark_summary.csv, строит сравнительные графики и
сохраняет итоговый отчёт reports/методологический_отчёт.md.

Запуск:  python scripts/05_make_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("sber_index.05")

OUT_DIR = Path("outputs/report")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_architecture():
    """A compact architecture figure for the competition report."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = {
        "data": (.02, .58, .18, .24, "СберИндекс\nРасходы МО\nТранспорт и рынки"),
        "features": (.27, .58, .20, .24, "Признаки до cutoff\nКалендарь, лаги\nПространственные связи"),
        "models": (.54, .58, .20, .24, "Прогнозные модели\nСезонность, бустинги\nProphet, Chronos"),
        "evaluation": (.80, .58, .18, .24, "Walk-forward\n1 / 3 / 6 / 12 мес.\nMAE, R², покрытие"),
        "news": (.02, .15, .18, .23, "Новости и макро\nДата публикации\nЗатухание эффекта"),
        "selection": (.54, .15, .20, .23, "Выбор и ансамбль\nТолько известные\nпрошлые ошибки"),
        "monitor": (.80, .15, .18, .23, "Мониторинг\nОстатки → детекторы\nСигналы соседей"),
    }
    for x, y, width, height, label in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01",
                                   facecolor="#eef7f0", edgecolor="#21a038", linewidth=1.5))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                    linewidth=1.4, color="#52665b"))
    arrow((.21, .70), (.26, .70))
    arrow((.48, .70), (.53, .70))
    arrow((.75, .70), (.79, .70))
    arrow((.21, .27), (.36, .56))
    arrow((.64, .56), (.64, .40))
    arrow((.75, .27), (.79, .27))
    arrow((.89, .56), (.89, .40))
    ax.text(.5, .96, "Прогноз потребления и обнаружение структурных изменений", ha="center", fontsize=15, weight="bold")
    ax.text(.5, .035, "Срез задаёт доступную информацию. Обнаруженная аномалия требует проверки по реальным событиям.",
            ha="center", fontsize=10, color="#52665b")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture.png", dpi=160)
    fig.savefig(OUT_DIR / "architecture.svg")
    plt.close(fig)
    return str(OUT_DIR / "architecture.png")


def plot_forecast_summary(summary: pd.DataFrame) -> str:
    """График MAE по моделям и горизонтам."""
    metrics = [column for column in ["mae", "r2"] if column in summary and summary[column].notna().any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.5))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        pivot = summary.pivot(index="horizon", columns="model", values=metric)
        pivot.sort_index().plot(marker="o", ax=ax)
        ax.set_title(f"Сравнение моделей: {metric.upper()}")
        ax.set_xlabel("Горизонт прогноза, мес.")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "forecast_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_changepoint(summary: pd.DataFrame) -> str:
    """График F1 и задержки детекторов."""
    if summary.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = summary["detector"].tolist()
    x = np.arange(len(labels))
    ax.bar(x - 0.15, summary["micro_f1"], width=0.3, label="F1 (micro)")
    ax.bar(x + 0.15, summary["micro_recall"], width=0.3, label="Recall (micro)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Синтетический бенчмарк детекторов шоков")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "changepoint_benchmark.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def compute_feature_importance(feat_path: Path) -> str:
    """Важность признаков: LightGBM h=1 на последнем доступном срезе."""
    import lightgbm as lgb

    from src.data.cleaner import inverse_transform
    from src.forecasting.lightgbm_model import ID_COLS, _make_target, fit_horizon_model

    if not feat_path.exists():
        return ""
    features = pd.read_parquet(feat_path)
    features["date"] = pd.PeriodIndex(features["date"], freq="M")
    cutoff = features.date.max() - 1
    train = features[features.date <= cutoff]
    data = _make_target(train, 1).dropna(subset=["target"])
    cols = [c for c in data.columns if c not in ID_COLS]
    X = data[cols].copy()
    cat_features = [c for c in cols if str(X[c].dtype) in ("category", "object")]
    for c in cat_features:
        X[c] = X[c].cat.codes
    model = fit_horizon_model(train, 1,
                             {"n_estimators": 200, "learning_rate": 0.03,
                              "num_leaves": 31, "n_jobs": 4,
                              "random_state": 42, "verbosity": -1}, cols)
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(np.arange(len(top))[::-1], top.values, color="#21a038")
    ax.set_yticks(np.arange(len(top))[::-1])
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_title("Feature importance (LightGBM, горизонт 1 мес)")
    fig.tight_layout()
    path = OUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    imp.rename_axis("feature").rename("importance").to_csv(OUT_DIR / "feature_importance.csv")
    return str(path)


def granger_section(out_dir: Path) -> list[str]:
    """Granger-тест новостей на прогнозных остатках."""
    resid_path = out_dir / "residuals.csv"
    news_path = Path("outputs/news/news_monthly.csv")
    import yaml
    manifest = Path("outputs/news/news_manifest.yaml")
    if manifest.exists() and yaml.safe_load(manifest.read_text(encoding="utf-8")).get("mode") == "demo":
        return ["\n### Новости\n", "Новостной конвейер проверен на демонстрационном корпусе; "
                "Granger-выводы и прирост качества на этих синтетических новостях не оцениваются.\n"]
    if not resid_path.exists() or not news_path.exists():
        return []
    from src.news.causality import test_news_on_residuals

    resid = pd.read_csv(resid_path)
    resid["date"] = pd.PeriodIndex(resid["date"], freq="M")
    news = pd.read_csv(news_path)
    news_cols = [c for c in news.columns if c.endswith("_shock")]
    lines = ["\n### Granger-причинность: новости -> отклонения (шоки)\n"]
    lines.append("| Новостной индекс | доля значимых серий (p<0.05) | медиана p | серий |")
    lines.append("|---|---|---|---|")
    for nc in news_cols:
        res = test_news_on_residuals(resid, news, nc, maxlag=2, n_series_sample=60)
        lines.append(f"| {nc} | {res['share_significant']:.2f} | "
                     f"{res['median_p']:.3f} | {res['n_tested']} |")
    return lines


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    from src.utils.config import latest_forecast_source
    parser.add_argument("--source", default=latest_forecast_source())
    args = parser.parse_args()
    source = Path(args.source)
    plot_architecture()
    audit_lines = []
    audit_path = source / "audit_report.md"
    if audit_path.exists():
        audit_lines = ["\n## Аудит и причинный выбор модели\n", audit_path.read_text(encoding="utf-8")]
    fc_path = source / "forecast_common_summary.csv"
    cp_path = Path("outputs/changepoint/benchmark_summary.csv")
    fc_summary = pd.read_csv(fc_path) if fc_path.exists() else pd.DataFrame()
    for column in ["rmse", "mape", "r2"]:
        if not fc_summary.empty and column not in fc_summary:
            fc_summary[column] = np.nan
    cp_summary = pd.read_csv(cp_path) if cp_path.exists() else pd.DataFrame()
    from src.utils.config import load_config
    import yaml
    run_path = source / "run_config.yaml"
    if not run_path.exists():
        run_path = source / "baseline_run_config.yaml"
    run_info = yaml.safe_load(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    from src.evaluation.provenance import verify_features
    if Path("data/processed/features.parquet").exists():
        verify_features("data/processed/features.parquet", run_info)
    prophet_sample = run_info.get("config", {}).get("models", {}).get("prophet", {}).get("subsample", 40)
    prophet_note = (f"Prophet — до {prophet_sample} серий" if not fc_summary.empty and fc_summary.model.eq("prophet").any()
                    else "Прогнозов Prophet в этом запуске нет")

    md = ["# Методологический отчёт\n",
          "## Прогнозирование и обнаружение точек структурных изменений\n",
          "Решение прогнозирует расходы каждого муниципалитета через 1, 3, 6 и 12 месяцев. "
          "Модели получают только известную к началу прогноза историю. После поступления факта "
          "система проверяет, насколько необычно его отклонение от прогноза, и учитывает прежние "
          "сигналы соседних территорий. Такие сигналы помогают выбрать случаи для проверки; "
          "предупреждение реальных будущих шоков пока не подтверждено независимой разметкой.\n",
          "### 1. Данные\n",
          "- consumption: 24 мес (2023-01..2024-12), 2190 МО, 6 категорий расходов\n",
          "- market_access: индекс доступности рынков (2571 МО)\n",
          "- connection: граф расстояний highway/railway (5.9 млн рёбер)\n",
          "\n![Архитектура решения](презентация/assets/architecture.png)\n",
          "\n### 2. Прогнозирование (walk-forward, MAE обязателен)\n"]

    if not fc_summary.empty:
        md.append("Сравнение на **одинаковых наблюдениях** всех моделей внутри каждого горизонта. "
                  f"Метрики всех доступных серий отдельно: `{source / 'forecast_summary.csv'}`.\n")
        md.append(f"Прогон: {run_info.get('territories', '?')} МО; "
                  f"проверочный режим: {run_info.get('quick', False)}. "
                  "Проверочный режим ограничивает бустинг 100 итерациями.\n")
        md.append(f"{prophet_note}. Таблица MAE/R2 по точным горизонтам:\n")
        md.append("| model | horizon | n_pairs | mae | rmse | mape | r2 |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in fc_summary.iterrows():
            md.append(
                f"| {r['model']} | {r['horizon']} | {r['n_pairs']} | "
                f"{r['mae']:.2f} | {r['rmse']:.2f} | {r['mape']:.2f} | {r['r2']:.4f} |")
        md.append(f"\n![Сравнение прогнозов](презентация/assets/forecast_comparison.png)\n")
        logger.info("График прогнозов: %s", plot_forecast_summary(fc_summary))
    else:
        md.append("Прогнозы не выполнены — запустите scripts/02_run_forecasting.py\n")

    md.append("\n### 3. Обнаружение структурных изменений (синтетический бенчмарк)\n")
    if not cp_summary.empty:
        md.append("Micro-метрики суммируют TP/FP/FN по всему корпусу. "
                  "Офлайн-точка изменения — локализация, а не время доступности предупреждения.\n")
        md.append("| detector | precision | recall | f1 | задержка онлайн-сигнала |")
        md.append("|---|---|---|---|---|")
        for _, r in cp_summary.iterrows():
            md.append(
                f"| {r['detector']} | {r['micro_precision']:.3f} | {r['micro_recall']:.3f} | "
                f"{r['micro_f1']:.3f} | {r['detection_delay']:.2f} |")
        img = plot_changepoint(cp_summary)
        best_cp = cp_summary.sort_values("micro_f1", ascending=False).iloc[0]
        online_cp = cp_summary[cp_summary.detector.isin(["cusum", "bocpd"])]
        md.append(f"\nПо micro-F1 на этом корпусе лучший метод — **{best_cp.detector}** "
                  f"({best_cp.micro_f1:.3f}); это оценка данного синтетического корпуса.\n")
        if not online_cp.empty:
            best_online = online_cp.sort_values("micro_f1", ascending=False).iloc[0]
            md.append(f"Для онлайн-мониторинга выбран **{best_online.detector}** "
                      f"(micro-F1 {best_online.micro_f1:.3f}). Переносимость на реальные шоки "
                      "пока не подтверждена.\n")
        if img:
            md.append(f"\n![Детекторы шоков](презентация/assets/changepoint_benchmark.png)\n")
    else:
        md.append("Бенчмарк не выполнен — запустите scripts/03_run_changepoint.py\n")

    # --- пространственная диффузия шоков ---
    examples_html = ""
    forecast_file = source / "forecast_results.csv"
    if forecast_file.exists() and not fc_summary.empty:
        from src.evaluation.walk_forward import common_results
        predictions = pd.read_csv(forecast_file, low_memory=False)
        for col in ["date", "cutoff"]:
            predictions[col] = pd.PeriodIndex(predictions[col], freq="M")
        shared = common_results(predictions)
        one_month = fc_summary[fc_summary.horizon == 1]
        if not one_month.empty:
            winner = one_month.sort_values("mae").iloc[0].model
            examples = shared[(shared.model == winner) & (shared.horizon == 1)].copy()
            examples["absolute_error"] = (examples.actual - examples.predicted).abs()
            examples = examples.sort_values("absolute_error")
            if not examples.empty:
                selected = examples.iloc[sorted(set([0, len(examples) // 2, len(examples) - 1]))]
                cols = ["territory_id", "category", "cutoff", "date", "actual", "predicted", "absolute_error"]
                selected = selected[cols].copy()
                md.append(f"\n### 3.1 Реальные примеры прогноза ({winner}, h=1)\n")
                md.append("Три наблюдения общей выборки: минимальная, медианная и максимальная "
                          "абсолютная ошибка. Это ошибки прогноза, а не автоматически размеченные шоки.\n")
                md.append("| МО | категория | cutoff | месяц | факт, руб. | прогноз, руб. | ошибка, руб. |")
                md.append("|---|---|---|---|---|---|---|")
                for _, r in selected.iterrows():
                    md.append(f"| {r.territory_id} | {r.category} | {r.cutoff} | {r.date} | "
                              f"{r.actual:.2f} | {r.predicted:.2f} | {r.absolute_error:.2f} |")
                examples_html = selected.to_html(index=False, float_format=lambda v: f"{v:.2f}")

    sp_path = Path("outputs/changepoint/spatial_trigger.csv")
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        n_trig = int(sp.get("spatial_trigger", pd.Series(dtype=int)).sum())
        n_rows = len(sp)
        md.append(f"\n### 3.1 Пространственная диффузия шоков\n")
        md.append(f"- **{n_trig}** МО-месяцев получили предупреждение о шоке от соседей "
                  f"(из {n_rows} наблюдений). Используются сигналы прошлых месяцев; "
                  f"способность предсказывать будущий шок требует отдельной проверки.\n")

    # --- Granger-причинность новостей ---
    md.extend(granger_section(Path("outputs/changepoint")))

    # --- важность признаков ---
    imp_img = compute_feature_importance(Path("data/processed/features.parquet"))
    if imp_img:
        md.append("\n### 3.2 Важность признаков\n")
        md.append(f"![Feature importance](презентация/assets/feature_importance.png)\n")
        imp = pd.read_csv(OUT_DIR / "feature_importance.csv")
        top5 = imp.head(5)
        md.append("Топ-5 признаков (LightGBM, горизонт 1):\n")
        for _, r in top5.iterrows():
            md.append(f"- `{r['feature']}` (importance {r['importance']})")
        md.append("")

    extra_html = ""
    for title, path in [
        ("Полная панель: модели на одинаковых наблюдениях", source / "forecast_panel_common_summary.csv"),
        ("Попарное сравнение с Prophet на совпадающих наблюдениях", source / "pairwise_vs_prophet.csv"),
        ("Покрытие, ошибки по сериям и резервные прогнозы", source / "coverage_audit.csv"),
        ("Парное сравнение с Prophet: кластерный bootstrap, 95% интервал", source / "paired_bootstrap.csv"),
        ("Настроенные детекторы: независимый синтетический holdout", Path("outputs/calibration/heldout_summary.csv")),
        ("Вклад реальных заголовков Банка России: с новостями / без новостей", Path("outputs/news_ablation/summary.csv")),
        ("Проверка прогнозных интервалов: покрытие и средняя ширина", source / "interval_coverage.csv"),
        ("Раннее предупреждение: отдельный синтетический сценарий распространения", Path("outputs/diffusion_benchmark/summary.csv")),
        ("Причинная калибровка интервалов: цель 80%, оценка только после разогрева", source / "calibrated_coverage_summary.csv"),
    ]:
        if path.exists():
            try:
                table = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if table.empty:
                continue
            md.append(f"\n### {title}\n")
            md.append("| " + " | ".join(table.columns) + " |")
            md.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
            for row in table.itertuples(index=False, name=None):
                md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
            extra_html += f"<section><h2>{title}</h2>{table.to_html(index=False, float_format=lambda v: f'{v:.3f}')}</section>"
    md.append("\n### 4. Выбор моделей и ограничения\n")
    md.append("В парном bootstrap отрицательная разность MAE означает преимущество перед Prophet; "
              "если интервал включает ноль, устойчивое преимущество не подтверждено. "
              "Покрытие интервалов необходимо оценивать отдельно от точечного MAE.\n")
    for h, group in fc_summary.groupby("horizon") if not fc_summary.empty else []:
        best = group.sort_values("mae").iloc[0]
        md.append(f"- Горизонт {h}: минимальный MAE у **{best['model']}** ({best['mae']:.2f}).")
    md.append("""

- При 24 месяцах истории годовой holdout имеет только одну точку старта.
  В 12 месяцах обучения нет пар t→t+12: LightGBM/CatBoost используют явно
  помеченный сезонный fallback, а не обученную годовую ML-модель.
- Ансамбль обучает веса только на ошибках с известным к cutoff фактом.
  Одинаковые fallback-прогнозы получают один голос; усреднение происходит в рублях.
  Годовой holdout имеет один cutoff: его выигрыш нельзя считать устойчивым.
- Prophet оценивается на фиксированной подвыборке из метаданных прогона. Все модели сравниваются
  на пересечении её наблюдений; результаты всей панели сохранены отдельно.
- Пропуски заполняются причинно. Искусственно заполненные цели исключены из
  обучения и оценки. Макро-признаки используют лаги 1/3 месяца, без текущего ИПЦ.
- Индекс доступности и транспортная сеть описывают 2024 год: это статические
  ковариаты данного датасета, не полностью воспроизводимый исторический vintage.
- Демонстрационные новости проверяют технический конвейер; они не являются
  доказательством прогностической ценности реальных новостей. Для неё нужны
  исторический корпус с датами публикаций и сравнение с/без новостей.
- События ЦБ выводятся как контекст; повышения ставки сами по себе не являются
  достоверной разметкой локальных шоков потребления. Chronos-Bolt Tiny подключён
  отдельно и проверяется без дообучения на CPU. Его опубликованный checkpoint
  используется ретроспективно: он не был доступен на всех исторических cutoffs.
  Его присутствие не означает автоматический выигрыш у Prophet.
- Bootstrap сохраняет серии целиком; малое число серий и общий для всех МО
  календарь ограничивают выводы. Интервал не учитывает неопределённость выбора моделей.
- Официальный новостной корпус содержит только решения о ставке: он узкий и
  не заменяет локальные новости. Направление номинальных расходов нельзя считать
  известным из тональности заголовка. Тест Грейнджера не доказывает причинность.
- Синтетическая проверка диффузии задаёт распространение по слоям графа за
  четыре месяца и сдвиги уровня на три сигмы. Положительный lead time там
  относится только к этому сценарию, не к доказанным будущим шокам реальных МО.
- Калибровка интервалов использует только уже известные относительные ошибки
  той же модели, категории и горизонта. До 20 ошибок интервал недоступен.
  На горизонтах 6/12 в историческом holdout нет подходящей прошлой выборки.
  Зависимость во времени нарушает предпосылку обменности: покрытие 80% не
  гарантируется, измеряется фактически. Столбец native_coverage_same_pairs
  сравнивает штатные интервалы на той же подвыборке после разогрева.
- Проверочный режим использует четыре cutoffs горизонта 1. После разогрева
  стандартизации этого недостаточно для оценки реального раннего предупреждения;
  полный режим использует ежемесячные прогнозы горизонта 1.

### 5. Воспроизводимость

- Конфиги: configs/*.yaml; фактические параметры прогона: run_config.yaml в выбранном каталоге --source.
- Тесты: python -m unittest discover -s tests -v.
- Порядок: подготовка → новости (опционально) → прогноз → шоки → отчёт.
- Зависимости: requirements.txt; окружение проверочного запуска: requirements-lock.txt.
""")

    report = Path("reports/методологический_отчёт.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    md.extend(audit_lines)
    md.append("\n### Источники и лицензии\n\n"
              "Лаборатория СберИндекс: потребительские безналичные расходы МО, индекс доступности рынков, "
              "автодорожные и железнодорожные связи. "
              "[Источник](https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim). "
              "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). "
              "Изменения: месячная панель, причинное заполнение с флагом observed, признаки, прогнозы, метрики и графики. "
              "Содержащие данные переработанные части таблиц и графиков сохраняют CC BY-SA 4.0. "
              "Дата скачивания не сохранена. Данные без гарантий; источник не одобряет проект. "
              "Новости и ключевая ставка: Пресс-служба Банка России; URL записей сохранены в data/news/cbr_releases.csv. "
              "[Условия](https://www.cbr.ru/about/). "
              "Chronos-Bolt Tiny: Amazon, Apache-2.0. Подробнее: [DATA_LICENSES.md](../DATA_LICENSES.md).\n")
    report.write_text("\n".join(md), encoding="utf-8")
    from html import escape
    html_path = Path("reports/презентация/index.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    assets = html_path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    generated = {"architecture.png", "architecture.svg"}
    if not fc_summary.empty:
        generated.add("forecast_comparison.png")
    if not cp_summary.empty:
        generated.add("changepoint_benchmark.png")
    if imp_img:
        generated.add("feature_importance.png")
    for name in ["architecture.png", "architecture.svg", "forecast_comparison.png", "changepoint_benchmark.png", "feature_importance.png"]:
        if name in generated:
            shutil.copy2(OUT_DIR / name, assets / name)
        else:
            (assets / name).unlink(missing_ok=True)
    forecast_image = '<img src="assets/forecast_comparison.png">' if not fc_summary.empty else ''
    changepoint_image = '<img src="assets/changepoint_benchmark.png">' if not cp_summary.empty else ''
    winners = " ".join(escape(x.replace("*", "")) for x in md if "минимальный MAE" in x)
    html_path.write_text(f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>СберИндекс — прогнозирование и шоки</title>
<link rel="stylesheet" href="assets/presentation.css">
<h1>Прогноз потребления и обнаружение шоков</h1>
<p>2190 МО · 6 категорий · 24 месяца · горизонты 1/3/6/12 · MAE и R²</p>
<section><h2>Методология</h2><p>СберИндекс → причинное заполнение → лаги, календарь,
макро и транспортные соседи → LightGBM / CatBoost / Prophet / naive → остатки → детекторы.</p>
<p>Прогнозы сравниваются на одинаковых фактах, каждый горизонт означает ровно cutoff+h.
Новостной контур агрегирует опубликованные новости по месяцам с затуханием эффекта.</p>
<p>После получения нового факта система проверяет отклонение от прогноза и прошлые сигналы соседей.
Это повод для проверки события; точность предупреждения реальных будущих шоков ещё не подтверждена.</p>
<img src="assets/architecture.png" alt="Архитектура решения"></section>
<section id="forecasts"><h2>Сравнение прогнозов</h2><label>Горизонт:
<select id="horizon"><option value="all">Все</option><option>1</option><option>3</option><option>6</option><option>12</option></select></label><p>Проверочный режим: {run_info.get('quick', False)}.
100 итераций бустинга в проверочном режиме; {prophet_note}; выборка — {run_info.get('territories', '?')} МО.</p>
{fc_summary.to_html(index=False, float_format=lambda v: f'{v:.3f}')}
<p>{winners}</p>{forecast_image}</section>
<section><h2>Реальные примеры прогноза</h2><p>Минимальная, медианная и максимальная
ошибка победителя h=1 на общей выборке. Отклонение прогноза само по себе не доказывает шок.</p>{examples_html}</section>
<section><h2>Детекторы структурных изменений</h2>
{cp_summary.to_html(index=False, float_format=lambda v: f'{v:.3f}')}
<p>PELT/BinSeg работают офлайн. Их точность локализации не доказывает раннее предупреждение.</p>
{changepoint_image}</section>
{extra_html}
<section><h2>Ограничения</h2><p>Годовой ML-прогноз — сезонный fallback из-за отсутствия обучающих пар.
Chronos-Bolt оценивается отдельно; факт подключения не гарантирует улучшение.
Официальные заголовки ЦБ проверены отдельным экспериментом с/без новостей.
Преимущество выбирается по фактическому MAE, без гарантии превосходства Prophet.</p>
<a href="../методологический_отчёт.md">Подробный методологический отчёт</a></section>
<section><h2>Источники и лицензии</h2><p>Лаборатория СберИндекс: расходы МО, доступность рынков,
транспортные связи. <a href="https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim">Источник данных</a> ·
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
<p>Изменения: месячная панель, причинное заполнение, признаки, прогнозы, метрики и графики.
Содержащие данные переработанные части таблиц и графиков сохраняют CC BY-SA 4.0.
Дата скачивания не сохранена. Данные без гарантий; источник не одобряет проект.</p>
<p>Новости и ставка: Пресс-служба Банка России; первоисточники каждой записи сохранены в корпусе.
<a href="https://www.cbr.ru/about/">Условия использования</a>.
Chronos-Bolt Tiny: Amazon, <a href="https://huggingface.co/amazon/chronos-bolt-tiny">Apache-2.0</a>.</p>
<p><a href="../../DATA_LICENSES.md">Полные сведения об источниках и лицензиях</a> ·
<a href="../../CONTEST_COMPLIANCE.md">Проверка требований конкурса</a></p></section>
<script>document.getElementById('horizon').addEventListener('change',function(){{
document.querySelectorAll('#forecasts tbody tr').forEach(row=>{{row.hidden=this.value!=='all'&&row.cells[1].textContent.trim()!==this.value;}});
}});</script><script src="assets/presentation.js"></script></html>''', encoding="utf-8")
    logger.info("Отчёт сохранён: %s", report)


if __name__ == "__main__":
    main()
