"""Zero-shot classification using bart-large-mnli."""

import logging
from transformers import pipeline as hf_pipeline

from . import db

log = logging.getLogger(__name__)

_classifier = None

LABELS = [
    "artificial intelligence",
    "cybersecurity",
    "cloud computing",
    "semiconductors",
    "fintech",
    "macroeconomics",
    "cryptocurrency",
    "regulation",
    "startups",
    "developer tools",
    "robotics",
    "energy",
]


def _get_classifier():
    global _classifier
    if _classifier is None:
        log.info("Loading zero-shot classifier...")
        _classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        log.info("Classifier loaded")
    return _classifier


def classify_article(text: str) -> tuple[str, float]:
    """Classify text into one of the predefined categories."""
    clf = _get_classifier()
    result = clf(text[:500], candidate_labels=LABELS, multi_label=False)
    return result["labels"][0], round(result["scores"][0], 3)


def run_classification(cur, articles: list[dict]) -> int:
    """Classify articles and store results."""
    classified = 0

    for article in articles:
        text = f"{article['title']} {article.get('description', '')}"
        category, confidence = classify_article(text)
        db.update_article_category(cur, article["id"], category, confidence)
        classified += 1

    log.info("Classified %d articles", classified)
    return classified
