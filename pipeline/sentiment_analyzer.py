"""Sentiment analysis using twitter-roberta-base-sentiment."""

import logging
from transformers import pipeline as hf_pipeline

from . import db

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
    model = _get_model()
    analyzed = 0

    for article in articles:
        text = f"{article['title']} {article.get('description', '')}"[:512]
        result = model(text)[0]

        label = LABEL_MAP.get(result["label"], "neutral")
        score = round(result["score"], 3)
        db.update_article_sentiment(cur, article["id"], label, score)
        analyzed += 1

    log.info("Analyzed sentiment for %d articles", analyzed)
    return analyzed
