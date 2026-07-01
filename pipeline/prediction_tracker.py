"""Extract and store predictions from cluster analyses and podcast moments.

Predictions are extracted from:
1. Cluster LLM analyses (content.predictions)
2. Podcast moments (content.predictions)

Each prediction is stored in the `predictions` table for tracking over time.
"""

import json
import logging

from . import db

log = logging.getLogger(__name__)


def extract_predictions_from_cluster_analysis(cur, cluster_id: str, cluster_title: str,
                                               analysis_content: dict) -> int:
    """Extract predictions from a cluster LLM analysis and store them."""
    predictions = analysis_content.get("predictions") or []
    if not predictions:
        return 0

    # Get source info from cluster
    cur.execute(
        """SELECT c.main_theme, ca.article_id, a.source_name
           FROM clusters c
           LEFT JOIN cluster_articles ca ON ca.cluster_id = c.id
           LEFT JOIN articles a ON a.id = ca.article_id
           WHERE c.id = %s
           ORDER BY ca.role = 'primary' DESC
           LIMIT 1""",
        (cluster_id,),
    )
    row = cur.fetchone()
    source_name = row["sourceName"] if row else None
    domain = (row["mainTheme"] if row else None) or "other"

    inserted = 0
    for pred in predictions:
        if not isinstance(pred, dict) or not pred.get("prediction"):
            continue

        # Check if this prediction already exists (avoid duplicates)
        pred_text = pred["prediction"][:500]
        cur.execute(
            "SELECT id FROM predictions WHERE source_type = 'cluster' AND source_id = %s AND prediction = %s",
            (cluster_id, pred_text),
        )
        if cur.fetchone():
            continue

        pred_id = db.gen_id()
        cur.execute(
            """INSERT INTO predictions
               (id, prediction, horizon, confidence, source_type, source_id,
                source_title, source_name, speaker, domain, status, created_at, updated_at)
               VALUES (%s, %s, %s, %s, 'cluster', %s, %s, %s, NULL, %s, 'pending', NOW(), NOW())""",
            (
                pred_id,
                pred_text,
                pred.get("horizon", "unknown"),
                pred.get("confidence", "implied"),
                cluster_id,
                cluster_title[:200],
                source_name,
                domain,
            ),
        )
        inserted += 1

    if inserted:
        log.info("[Predictions] Extracted %d from cluster '%s'", inserted, cluster_title[:50])
    return inserted


def extract_predictions_from_podcast_moments(cur, article_id: str, article_title: str,
                                              source_name: str, moments_content: dict) -> int:
    """Extract predictions from podcast moments and store them."""
    predictions = moments_content.get("predictions") or []
    if not predictions:
        return 0

    domain = "other"
    # Try to infer domain from the podcast moments
    concepts = moments_content.get("concepts_explained") or []
    for concept in concepts:
        if isinstance(concept, dict) and concept.get("domain"):
            domain = concept["domain"].lower()
            break

    inserted = 0
    for pred in predictions:
        if not isinstance(pred, dict) or not pred.get("prediction"):
            continue

        pred_text = pred["prediction"][:500]
        cur.execute(
            "SELECT id FROM predictions WHERE source_type = 'podcast' AND source_id = %s AND prediction = %s",
            (article_id, pred_text),
        )
        if cur.fetchone():
            continue

        pred_id = db.gen_id()
        cur.execute(
            """INSERT INTO predictions
               (id, prediction, horizon, confidence, source_type, source_id,
                source_title, source_name, speaker, domain, status, created_at, updated_at)
               VALUES (%s, %s, %s, %s, 'podcast', %s, %s, %s, %s, %s, 'pending', NOW(), NOW())""",
            (
                pred_id,
                pred_text,
                pred.get("horizon", "unknown"),
                pred.get("confidence", "implied"),
                article_id,
                article_title[:200],
                source_name,
                pred.get("speaker"),
                domain,
            ),
        )
        inserted += 1

    if inserted:
        log.info("[Predictions] Extracted %d from podcast '%s'", inserted, article_title[:50])
    return inserted
