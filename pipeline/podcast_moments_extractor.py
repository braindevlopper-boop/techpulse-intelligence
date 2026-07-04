"""Podcast moments extraction — disabled after the D1 migration.

Read podcast transcripts from Neon `articles` (source_type='podcast') and
wrote to `ai_analyses` (analysis_type='podcast_moments'), neither reachable
through the D1 pipeline contract. Disabled as a clean no-op.
"""

import logging

log = logging.getLogger(__name__)


def run_podcast_moments_extraction(*_args, **_kwargs) -> int:
    log.warning("Podcast moments extraction skipped: ai_analyses has no D1 equivalent")
    return 0
