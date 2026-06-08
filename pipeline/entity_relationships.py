"""Build entity relationship graph edges from clustered article evidence."""

import logging

from . import db

log = logging.getLogger(__name__)


def _relation_type(source_type: str | None, target_type: str | None) -> str:
    types = {source_type or "unknown", target_type or "unknown"}
    if types & {"regulation", "country"}:
        if types & {"company", "person", "product", "model"}:
            return "policy_context"
        return "regulatory_context"
    if types & {"financial_asset", "sector"}:
        if types & {"company", "product", "model"}:
            return "market_exposure"
        return "market_context"
    if types & {"technology", "concept"}:
        if types & {"company", "person", "product", "model"}:
            return "technology_link"
        return "technical_context"
    if types <= {"company", "person", "product", "model"}:
        return "actor_link"
    return "cooccurs_in_cluster"


def _strength_score(row: dict) -> int:
    evidence_count = int(row.get("evidence_count") or 0)
    article_signal = int(row.get("article_signal") or 0)
    source_signal = int(row.get("source_signal") or 0)

    score = (
        min(evidence_count, 6) * 22
        + min(article_signal, 20) * 2
        + min(source_signal, 16) * 3
    )
    return max(1, min(100, score))


def build_entity_relationships(cur) -> int:
    """Rebuild graph edges after clustering and scoring.

    The relationship table is optional during rollout: if the migration has not
    been applied yet, the pipeline logs and skips the graph step.
    """
    if not db.has_entity_relationships_table(cur):
        log.warning("entity_relationships table missing; graph build skipped")
        return 0

    candidates = db.fetch_entity_relationship_candidates(cur)
    relationships = []
    for row in candidates:
        evidence_clusters = row.get("evidence_clusters") or []
        relationships.append({
            "source_entity_id": row["source_entity_id"],
            "target_entity_id": row["target_entity_id"],
            "relation_type": _relation_type(row.get("source_entity_type"), row.get("target_entity_type")),
            "strength_score": _strength_score(row),
            "evidence_count": int(row.get("evidence_count") or 0),
            "evidence_cluster_ids": row.get("evidence_cluster_ids") or [],
            "evidence_article_ids": [],
            "evidence_summary": {
                "clusters": evidence_clusters[:5],
                "article_signal": int(row.get("article_signal") or 0),
                "source_signal": int(row.get("source_signal") or 0),
            },
            "first_seen_at": row.get("first_seen_at"),
            "last_seen_at": row.get("last_seen_at"),
        })

    created = db.replace_entity_relationships(cur, relationships)
    log.info("Entity relationship graph rebuilt: %d relationships", created)
    return created
