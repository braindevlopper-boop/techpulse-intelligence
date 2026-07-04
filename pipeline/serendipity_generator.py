"""Sérendipité scientifique — disabled after the D1 migration.

Depended on Neon's serendipity_cards table (and article_intelligence for
candidate selection), neither of which has a D1 equivalent/route. Disabled
as a clean no-op rather than inventing a new Worker route for a secondary
feature.
"""

import logging

log = logging.getLogger(__name__)


def run_serendipity(*_args, **_kwargs) -> int:
    log.warning("Serendipity generation skipped: serendipity_cards has no D1 equivalent")
    return 0
