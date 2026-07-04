"""Cluster scoring — disabled after the D1 migration.

Importance/growth/novelty scores and trend_snapshots lived only in Neon
(clusters.importance_score/growth_score/novelty_score, trend_snapshots table)
with no D1 equivalent in the /pipeline/clusters contract. Ranking for LLM
analysis now uses article_count directly (see llm_analyzer.run_llm_analysis).
"""

import logging

log = logging.getLogger(__name__)


def run_scoring(*_args, **_kwargs) -> int:
    log.warning("Cluster scoring skipped: importance/growth/novelty have no D1 equivalent")
    return 0
