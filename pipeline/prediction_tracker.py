"""Prediction tracking — disabled after the D1 migration.

Wrote to Neon's `predictions` table, extracted from cluster analyses / podcast
moments content that D1 doesn't store (see llm_analyzer._map_analysis_for_push,
which only keeps summary/impact/reliability/risk fields). No D1 equivalent
exists, so both extraction entry points are now no-ops.
"""

import logging

log = logging.getLogger(__name__)


def extract_predictions_from_cluster_analysis(*_args, **_kwargs) -> int:
    log.warning("Prediction extraction skipped: predictions table has no D1 equivalent")
    return 0


def extract_predictions_from_podcast_moments(*_args, **_kwargs) -> int:
    log.warning("Prediction extraction skipped: predictions table has no D1 equivalent")
    return 0
