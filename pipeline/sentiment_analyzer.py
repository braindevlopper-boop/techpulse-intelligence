"""Sentiment analysis using twitter-roberta-base-sentiment."""

import logging
from transformers import pipeline as hf_pipeline

from . import db
from .hf_utils import handle_hf_unavailable, hf_steps_disabled

log = logging.getLogger(__name__)

_sentiment = None


def _get_model():
    global _sentiment
    if _sentiment is None:
        log.info("Loading sentiment model...")
        _sentiment = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,
        )
        log.info("Sentiment model loaded")
    return _sentiment


LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


def run_sentiment_analysis(cur, articles: list[dict]) -> int:
    """Analyze sentiment of articles."""
    if hf_steps_disabled():
        log.info("Sentiment skipped: TECHPULSE_SKIP_HF_ML enabled")
        return 0

    try:
        model = _get_model()
    except Exception as exc:
        if handle_hf_unavailable(log, "Sentiment", exc):
            return 0

    total = len(articles)
    analyzed = 0

    for i, article in enumerate(articles, 1):
        text = f"{article['title']} {article.get('description', '')}"[:512]
        result = model(text)[0]

        label = LABEL_MAP.get(result["label"], "neutral")
        score = float(result["score"])
        db.update_article_sentiment(cur, article["id"], label, score)
        analyzed += 1

        if i % 20 == 0 or i == total:
            log.info("Sentiment: %d/%d (%.0f%%)", i, total, i / total * 100)

    log.info("Analyzed sentiment for %d articles", analyzed)
    return analyzed
