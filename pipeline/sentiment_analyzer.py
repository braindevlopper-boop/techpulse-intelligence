"""Sentiment analysis — disabled after the D1 migration.

Wrote to Neon's articles.sentiment/sentiment_score columns, which have no D1
equivalent in the pipeline contract. Disabled as a clean no-op (was already
gated by TECHPULSE_SKIP_HF_ML=true in production).
"""

import logging

log = logging.getLogger(__name__)


def run_sentiment_analysis(*_args, **_kwargs) -> int:
    log.warning("Sentiment analysis skipped: articles.sentiment has no D1 equivalent")
    return 0
