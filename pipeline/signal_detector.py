"""Weak signal detection — disabled after the D1 migration.

This relied on clusters.growth_score/novelty_score (Neon-only, see scorer.py)
to flag emerging topics. No D1 equivalent exists, so this is now a no-op.
"""

import logging

log = logging.getLogger(__name__)


def detect_weak_signals(*_args, **_kwargs) -> list[dict]:
    log.warning("Weak signal detection skipped: growth/novelty scores have no D1 equivalent")
    return []
