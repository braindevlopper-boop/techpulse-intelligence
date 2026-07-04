"""Podcast generation — disabled after the D1 migration.

Depended on Neon-only tables with no D1 equivalent: fetch_top_clusters'
importance/growth scores (scorer.py, now disabled), fetch_serendipity_cards_by_ids
(serendipity_generator.py, now disabled), and podcasts (no D1 route). Disabled
as a clean no-op rather than inventing new Worker routes for a secondary
feature — the core veille pipeline (clustering + cluster analyses) does not
depend on this module.
"""

import logging

log = logging.getLogger(__name__)


def resolve_topics(*_args, **_kwargs) -> list[dict]:
    log.warning("Podcast topic resolution skipped: clusters/serendipity scoring has no D1 equivalent")
    return []


def generate_podcast(*_args, **_kwargs) -> str | None:
    log.warning("Podcast generation skipped: podcasts table has no D1 equivalent")
    return None
