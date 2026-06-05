"""Weak signal detection — find emerging topics before they go mainstream."""

import logging

from . import db

log = logging.getLogger(__name__)


def detect_weak_signals(cur, min_sources: int = 3, max_mentions: int = 15,
                        min_growth: float = 2.0) -> list[dict]:
    """Detect clusters that look like weak signals.

    Criteria:
    - article_count < max_mentions (not yet mainstream)
    - source_diversity >= min_sources (confirmed across channels)
    - growth_score > min_growth * 10 (growing fast)
    - first_seen < 72h (recent)
    """
    cur.execute(
        """
        SELECT c.id, c.title, c.article_count, c.source_diversity,
               c.growth_score, c.novelty_score, c.importance_score,
               c.first_seen_at
        FROM clusters c
        WHERE c.status IN ('active', 'growing')
          AND c.article_count < %s
          AND c.source_diversity >= %s
          AND c.growth_score > %s
          AND c.first_seen_at > NOW() - INTERVAL '72 hours'
        ORDER BY c.growth_score DESC
        LIMIT 10
        """,
        (max_mentions, min_sources, int(min_growth * 10)),
    )

    signals = cur.fetchall()
    log.info("Detected %d weak signals", len(signals))

    for signal in signals:
        cur.execute(
            "UPDATE clusters SET status = 'growing' WHERE id = %s AND status = 'active'",
            (signal["id"],),
        )

    return signals
