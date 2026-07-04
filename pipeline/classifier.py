"""Zero-shot article classification — disabled after the D1 migration.

Wrote to Neon's articles.category/category_confidence columns, which have no
D1 equivalent in the pipeline contract (article enrichment only accepts
impact_fr/reliability/why_interesting_fr/score_interest/keywords_json).
Disabled as a clean no-op (was already gated by TECHPULSE_SKIP_HF_ML=true in
production).
"""

import logging

log = logging.getLogger(__name__)


def run_classification(*_args, **_kwargs) -> int:
    log.warning("Classification skipped: articles.category has no D1 equivalent")
    return 0
