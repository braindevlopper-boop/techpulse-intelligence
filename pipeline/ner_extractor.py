"""Named Entity Recognition — disabled after the D1 migration.

Wrote to Neon's `entities`/`article_entities` tables, which have no D1 route.
Disabled as a clean no-op (was already gated by TECHPULSE_SKIP_HF_ML=true in
production).
"""

import logging

log = logging.getLogger(__name__)


def run_ner(*_args, **_kwargs) -> int:
    log.warning("NER skipped: entities/article_entities have no D1 equivalent")
    return 0
