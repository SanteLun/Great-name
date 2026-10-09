# -*- coding: utf-8 -*-
"""Шаг 4. Новостной контур.

1. Загрузка новостей: из CSV-файла data/news/news.csv (date, title, text),
   либо фетч RSS-ленты (опционально, при сетевом доступе).
2. Тональность + тематическая классификация.
3. Агрегация в месячные индексы с экспоненциально-затухающими лагами
   (News Impact Curve) и масштабированием на МО через market_access.
4. Встраивание новостных признаков в features.parquet.

Запуск:  python scripts/04_news_pipeline.py [--csv data/news/news.csv]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.loader import load_market_access
from src.news.pipeline import (load_news_csv, score_news, monthly_news_impact,
                               align_to_mo, merge_news_features)
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("sber_index.04")


def fetch_rss(url: str, limit: int = 500) -> pd.DataFrame:
    """Фетч RSS-ленты и парсинг в DataFrame(date, title, text)."""
    import xml.etree.ElementTree as ET
    from urllib.request import urlopen

    logger.info("Фетч RSS: %s", url)
    with urlopen(url, timeout=20) as resp:
        root = ET.fromstring(resp.read())
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        link = item.findtext("link")
        if not title or not pub:
            continue
        try:
            date = pd.to_datetime(pub, utc=True).tz_localize(None)
        except Exception:  # noqa: BLE001
            continue
        rows.append({"date": date, "title": title, "text": title, "link": link})
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Новостной контур")
    parser.add_argument("--csv", default=None, help="Путь к CSV с новостями")
    parser.add_argument("--fetch", action="store_true", help="Фетч RSS из конфига")
    parser.add_argument("--no-merge", action="store_true", help="Проверить новости без изменения признаков прогноза")
    args = parser.parse_args()

    news_cfg = load_config("news")
    data_cfg = load_config("data")
    feat_path = Path(data_cfg["paths"]["processed_dir"]) / "features.parquet"

    # --- загрузка новостей ---
    if args.csv:
        if not Path(args.csv).exists():
            raise FileNotFoundError(args.csv)
        df_news = load_news_csv(args.csv)
        logger.info("Загружены новости из CSV: %d", len(df_news))
    elif args.fetch:
        rss = next((s for s in news_cfg["sources"] if s.get("enabled")), None)
        if rss is None:
            raise SystemExit("Нет включённого источника в configs/news.yaml")
        df_news = fetch_rss(rss["url"])
        logger.info("Фетч завершён: %d новостей", len(df_news))
    else:
        # демо-режим: синтетический корпус на период 2023-2024
        dates = pd.date_range("2023-01-05", "2024-11-25", freq="7D")
        samples = [
            ("ЦБ повысил ключевую ставку, ужесточение денежно-кредитной политики", 0),
            ("Инфляция ускорилась, рост цен на продовольствие", 0),
            ("Реальные доходы населения выросли, спрос восстанавливается", 1),
            ("Кризис в розничной торговле, снижение продаж и закрытия", 2),
            ("Маркетплейсы отчитались о рекордных продажах", 3),
            ("Санкции усиливают неопределённость в экономике", 4),
        ]
        n = len(dates)
        titles = ([s[0] for s in samples] * (n // len(samples) + 1))[:n]
        df_news = pd.DataFrame({
            "date": dates,
            "title": titles,
            "text": titles,
        })
        logger.info("Демо-режим: %d новостей", len(df_news))

    # --- скоринг ---
    scored = score_news(df_news, news_cfg)
    logger.info("После скоринга: %d (тем: %s)",
                len(scored), scored.topic.value_counts().to_dict())

    # --- месячные индексы ---
    monthly = monthly_news_impact(scored, news_cfg)
    logger.info("Месячные индексы: %d месяцев", len(monthly))

    # --- масштабирование на МО ---
    ma = load_market_access(data_cfg["paths"]["market_access"])
    news_mo = align_to_mo(monthly, ma, news_cfg)
    logger.info("Новостные признаки на уровне МО: %d строк", len(news_mo))

    # сохраняем месячные индексы (для Granger-теста в шаге 3/5)
    from pathlib import Path as _P
    out_n = _P(news_cfg["output_dir"]); out_n.mkdir(parents=True, exist_ok=True)
    monthly_save = monthly.copy(); monthly_save["date"] = monthly_save.index.astype(str)
    monthly_save.to_csv(out_n / "news_monthly.csv", index=False)
    news_mo.to_csv(out_n / "news_monthly_mo.csv", index=False)
    from src.utils.config import save_dict_yaml
    save_dict_yaml({"mode": "csv" if args.csv else "rss" if args.fetch else "demo",
                    "source": args.csv, "rows": len(df_news), "config": news_cfg},
                   out_n / "news_manifest.yaml")

    # --- встраивание в панель признаков ---
    if feat_path.exists() and not args.no_merge:
        features = pd.read_parquet(feat_path)
        features["date"] = pd.PeriodIndex(features["date"], freq="M")
        features = merge_news_features(features, news_mo)
        # вернём date в строковый формат для parquet-совместимости
        features["date"] = features["date"].astype(str)
        features.to_parquet(feat_path, index=False)
        logger.info("Признаки с новостями сохранены: %s", feat_path)
    elif not args.no_merge:
        out = Path(news_cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        news_mo["date"] = news_mo["date"].astype(str)
        news_mo.to_csv(out / "news_monthly_mo.csv", index=False)
        logger.info("Новостные признаки сохранены: %s",
                    out / "news_monthly_mo.csv (запустите сначала шаг 1 для встраивания)")


if __name__ == "__main__":
    main()
