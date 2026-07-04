"""KeyBERT keyword discovery — disabled after the D1 migration.

Wrote to Neon's standalone `keywords` table, which has no D1 route. Article-
and cluster-level keywords now flow only through keywords_json on the
/pipeline/articles/enrich and /pipeline/clusters payloads. Disabled as a
clean no-op (was already gated by TECHPULSE_SKIP_HF_ML=true in production).
"""

import logging

log = logging.getLogger(__name__)


def run_keyword_extraction(*_args, **_kwargs) -> int:
    log.warning("Keyword extraction skipped: standalone keywords table has no D1 equivalent")
    return 0
