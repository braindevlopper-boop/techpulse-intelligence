"""Zero-shot classification using DeBERTa-v3-base-mnli.

DeBERTa-v3-base is 4x smaller than BART-large but scores equally
on MNLI (90.4% vs 90.1%). Much faster on CPU.
"""

import logging
from transformers import pipeline as hf_pipeline

from . import db
from .hf_utils import handle_hf_unavailable, hf_steps_disabled

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
        log.info("Loading zero-shot classifier (DeBERTa-v3-base)...")
        _classifier = hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=-1,
        )
        log.info("Classifier loaded")
    return _classifier


def classify_article(text: str) -> tuple[str, float]:
    """Classify text into one of the predefined categories."""
    clf = _get_classifier()
    result = clf(text[:300], candidate_labels=LABELS, multi_label=False)
    return result["labels"][0], float(result["scores"][0])


def run_classification(cur, articles: list[dict]) -> int:
    """Classify articles and store results."""
    if hf_steps_disabled():
        log.info("Classification skipped: TECHPULSE_SKIP_HF_ML enabled")
        return 0

    try:
        _get_classifier()
    except Exception as exc:
        if handle_hf_unavailable(log, "Classification", exc):
            return 0

    total = len(articles)
    classified = 0

    for i, article in enumerate(articles, 1):
        text = f"{article['title']} {article.get('description', '')}"
        category, confidence = classify_article(text)
        db.update_article_category(cur, article["id"], category, confidence)
        classified += 1

        if i % 20 == 0 or i == total:
            log.info("Classification: %d/%d (%.0f%%) — last: %s → %s",
                     i, total, i / total * 100, article["title"][:40], category)

    log.info("Classified %d articles", classified)
    return classified
