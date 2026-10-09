# -*- coding: utf-8 -*-
"""Тональность и тематическая классификация русскоязычных новостей.

Базовый режим — лексический (словари позитивных/негативных слов, быстрый,
без внешних моделей). Опциональный режим — RuBERT через transformers.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("sber_index.news")

# Базовые словари (расширяемые)
POSITIVE_WORDS = {
    "рост", "растёт", "растет", "вырос", "повышение", "повысился", "увеличение",
    "увеличился", "рекорд", "улучшение", "улучшился", "поддержка", "стабильно",
    "стабильность", "восстановление", "прибыль", "снижение ставки", "смягчение",
    "позитив", "оптимизм", "успех", "достижение", "рост доходов", "инвестиции",
}
NEGATIVE_WORDS = {
    "повысить ключевую ставку", "повышение ключевой ставки",
    "падение", "упал", "снижение", "снизился", "обвал", "кризис", "санкции",
    "санкции", "шок", "рецессия", "девальвация", "инфляция", "рост цен",
    "безработица", "увольнения", "закрытие", "банкротство", "дефицит",
    "неопределённость", "неопределенность", "пессимизм", "угроза", "потери",
    "убытки", "повышение ставки", "ужесточение", "заморозка", "ограничения",
}

RU_PUNCT_RE = re.compile(r"[^\w\s\-а-яёА-ЯЁ]")


def tokenize(text: str) -> list[str]:
    text = RU_PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if t]


def lexicon_sentiment(text: str) -> float:
    """Словарная тональность в [-1, 1]. 0 — нейтрально."""
    words = tokenize(text)
    if not words:
        return 0.0
    # Сначала распознаём фразы, чтобы «рост цен» не считался позитивным ростом.
    normalized = " ".join(words)
    pos, neg = 0, 0
    for lexicon, sign in [(NEGATIVE_WORDS, -1), (POSITIVE_WORDS, 1)]:
        for phrase in sorted((w for w in lexicon if " " in w), key=len, reverse=True):
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            count = len(re.findall(pattern, normalized))
            if sign < 0:
                neg += count
            else:
                pos += count
            normalized = re.sub(pattern, " ", normalized)
    pos += sum(1 for w in normalized.split() if w in POSITIVE_WORDS)
    neg += sum(1 for w in normalized.split() if w in NEGATIVE_WORDS)
    n = pos + neg
    if n == 0:
        return 0.0
    return (pos - neg) / n


def classify_topic(text: str, topics: dict[str, list[str]]) -> str | None:
    """Тематическая классификация по ключевым словам; возвращает имя темы."""
    low = text.lower()
    best, best_hits = None, 0
    for topic, keywords in topics.items():
        hits = sum(1 for kw in keywords if kw.lower() in low)
        if hits > best_hits:
            best_hits = hits
            best = topic
    return best if best_hits > 0 else None


def rubert_sentiment(texts: list[str], model_name: str = "cointegrated/rubert-tiny") -> list[float]:
    """Тональность через RuBERT (тяжёлый режим, требует transformers)."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        logger.warning("transformers не установлен, используем лексический метод: %s", exc)
        return [lexicon_sentiment(t) for t in texts]
    clf = pipeline("sentiment-analysis", model=model_name)
    out = []
    for res in clf(texts[:200]):
        label = res["label"].lower()
        score = float(res["score"])
        out.append(score if "pos" in label else -score)
    return out
