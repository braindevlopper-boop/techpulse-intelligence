"""Keyword extraction using KeyBERT."""

import logging
from keybert import KeyBERT

from . import db

log = logging.getLogger(__name__)

_kw_model = None


def _get_model():
    global _kw_model
    if _kw_model is None:
        log.info("Loading KeyBERT model...")
        _kw_model = KeyBERT("BAAI/bge-small-en-v1.5")
        log.info("KeyBERT loaded")
    return _kw_model


def extract_keywords(text: str, top_n: int = 8) -> list[tuple[str, float]]:
    """Extract top keywords from text."""
    model = _get_model()
    return model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )


def run_keyword_extraction(cur, articles: list[dict]) -> int:
    """Extract keywords from articles and store in DB."""
    total = 0

    for article in articles:
        text = article.get("full_text") or f"{article['title']} {article.get('description', '')}"
        if len(text) < 30:
            continue

        keywords = extract_keywords(text[:3000])
        for kw, score in keywords:
            if score < 0.3 or len(kw) < 3:
                continue
            db.upsert_keyword(
                cur, kw, category="concept", source="keybert",
                reason=f"from article: {article['title'][:50]}",
            )
            total += 1

    log.info("Extracted %d keywords from %d articles", total, len(articles))
    return total
