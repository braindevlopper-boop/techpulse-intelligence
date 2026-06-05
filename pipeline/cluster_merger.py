"""Pass 2 — LLM-based cluster merging.

After the embedding-based clustering (Pass 1), many clusters cover
the same story with slightly different wording. This module sends
cluster titles to DeepSeek V4 Flash and asks it to group them.

Cost: ~$0.002 per run (one API call with all cluster titles).
"""

import json
import logging

from . import db
from .llm_analyzer import analyze_with_deepseek

log = logging.getLogger(__name__)

MERGE_PROMPT = """Tu es un analyste spécialisé en veille technologique et financière.

Voici {count} clusters d'articles détectés aujourd'hui. Chaque cluster a un titre et un nombre d'articles.

{clusters_text}

Ton travail : identifier les clusters qui parlent du MÊME sujet ou de la MÊME histoire et qui devraient être fusionnés.

Règles :
- Ne fusionne que les clusters qui couvrent vraiment le même événement/sujet
- "Nvidia AI chips" et "Nvidia Computex expansion" = même histoire → fusionner
- "Nvidia AI chips" et "AMD new GPU" = sujet lié mais différent → NE PAS fusionner
- Un cluster seul qui ne ressemble à aucun autre reste tel quel

Réponds avec un JSON :
{{
  "merge_groups": [
    {{
      "merged_title": "titre unifié court et clair",
      "cluster_ids": ["id1", "id2", "id3"],
      "reason": "explication courte"
    }}
  ]
}}

Ne retourne QUE les groupes à fusionner. Les clusters isolés ne doivent pas apparaître."""


def build_merge_prompt(clusters: list[dict]) -> str:
    """Build the merge prompt with all cluster titles."""
    clusters_text = ""
    for c in clusters:
        clusters_text += f"- ID: {c['id']} | Articles: {c['article_count']} | \"{c['title']}\"\n"

    return MERGE_PROMPT.format(
        count=len(clusters),
        clusters_text=clusters_text,
    )


def execute_merges(cur, merge_groups: list[dict]) -> int:
    """Execute the LLM-suggested merges in the database."""
    total_merged = 0

    for group in merge_groups:
        cluster_ids = group.get("cluster_ids", [])
        new_title = group.get("merged_title", "")

        if len(cluster_ids) < 2 or not new_title:
            continue

        # Pick the cluster with the most articles as the target
        candidates = []
        for cid in cluster_ids:
            cur.execute(
                "SELECT id, article_count FROM clusters WHERE id = %s",
                (cid,),
            )
            row = cur.fetchone()
            if row:
                candidates.append(row)

        if len(candidates) < 2:
            continue

        candidates.sort(key=lambda x: x["article_count"], reverse=True)
        target_id = candidates[0]["id"]
        source_ids = [c["id"] for c in candidates[1:]]

        # Move articles from source clusters to target
        for source_id in source_ids:
            # Update articles table
            cur.execute(
                "UPDATE articles SET cluster_id = %s WHERE cluster_id = %s",
                (target_id, source_id),
            )

            # Move cluster_articles entries
            cur.execute(
                "UPDATE cluster_articles SET cluster_id = %s WHERE cluster_id = %s",
                (target_id, source_id),
            )

            # Move timeline events
            cur.execute(
                "UPDATE timeline_events SET cluster_id = %s WHERE cluster_id = %s",
                (target_id, source_id),
            )

            # Delete the emptied cluster
            cur.execute("DELETE FROM clusters WHERE id = %s", (source_id,))

        # Update the target cluster
        cur.execute(
            """
            UPDATE clusters
            SET title = %s,
                article_count = (
                    SELECT COUNT(*) FROM cluster_articles WHERE cluster_id = %s
                ),
                source_diversity = (
                    SELECT COUNT(DISTINCT a.source_type)
                    FROM articles a WHERE a.cluster_id = %s
                ),
                last_updated_at = NOW()
            WHERE id = %s
            """,
            (new_title, target_id, target_id, target_id),
        )

        merged_count = len(source_ids)
        total_merged += merged_count
        log.info("Merged %d clusters → \"%s\"", merged_count + 1, new_title)

    return total_merged


def run_cluster_merging(cur) -> int:
    """Run LLM-based cluster merging (Pass 2).

    Only processes clusters with 2+ articles to keep the prompt short.
    """
    # Fetch clusters worth merging (2+ articles)
    cur.execute(
        """
        SELECT id, title, article_count, source_diversity
        FROM clusters
        WHERE status IN ('active', 'growing')
          AND article_count >= 1
        ORDER BY article_count DESC
        LIMIT 100
        """,
    )
    clusters = cur.fetchall()

    if len(clusters) < 5:
        log.info("Too few clusters for merging (%d)", len(clusters))
        return 0

    log.info("Running LLM cluster merge on %d clusters...", len(clusters))
    prompt = build_merge_prompt(clusters)

    result = analyze_with_deepseek(prompt, model="deepseek-v4-flash")
    if not result:
        log.warning("LLM merge failed, skipping")
        return 0

    merge_groups = result.get("merge_groups", [])
    if not merge_groups:
        log.info("LLM found no clusters to merge")
        return 0

    log.info("LLM suggests %d merge groups", len(merge_groups))
    merged = execute_merges(cur, merge_groups)

    # Update final cluster count
    cur.execute("SELECT COUNT(*) as c FROM clusters WHERE status IN ('active', 'growing')")
    remaining = cur.fetchone()["c"]
    log.info("Cluster merging done: %d merges, %d clusters remaining", merged, remaining)

    return merged
