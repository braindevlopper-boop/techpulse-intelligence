"""Pass 2 — LLM-based cluster merging.

After the embedding-based clustering (Pass 1), many clusters cover
the same story with slightly different wording. This module sends
cluster titles to DeepSeek V4 Flash and asks it to group them.

Cost: ~$0.002 per run (one API call with all cluster titles).
"""

import json
import logging

from . import db
from .llm_analyzer import analyze_with_deepseek, analyze_with_gemini
from .prompt_registry import render_prompt

log = logging.getLogger(__name__)

MERGE_PROMPT = """Tu es un analyste spécialisé en veille technologique et financière.

Voici {count} clusters d'articles détectés aujourd'hui. Chaque cluster a un titre et un nombre d'articles.

{clusters_text}

Ton travail : identifier les clusters qui parlent du MÊME événement ou de la MÊME histoire précise et qui devraient être fusionnés.

Règles :
- Ne fusionne que les doublons ou variantes rédactionnelles du même événement.
- Si les fingerprints/hints décrivent des événements différents, ne fusionne pas.
- Ne crée jamais de panier large comme "Global AI governance", "latest developments", "financial news" ou "industry challenges".
- OpenAI policy, Trump executive order, US House AI bill et EU sovereignty = liés mais différents → NE PAS fusionner.
- SpaceX IPO price, SpaceX revenue forecast et Google compute deal = liés mais différents → NE PAS fusionner.
- Summer Game Fest, Xbox Showcase et GTA VI release calendar = liés gaming mais différents → NE PAS fusionner.
- Deux titres sur les mêmes fuites ISS et le même shelter Dragon = même histoire → fusionner.
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


def build_merge_prompt(cur, clusters: list[dict]) -> str:
    """Build the merge prompt with all cluster titles."""
    clusters_text = ""
    for c in clusters:
        fingerprints = ", ".join(c.get("event_fingerprints") or [])
        hints = ", ".join(c.get("cluster_hints") or [])
        topics = ", ".join(c.get("topics") or [])
        clusters_text += (
            f"- ID: {c['id']} | Articles: {c['article_count']} | \"{c['title']}\"\n"
            f"  topics: {topics or 'n/a'}\n"
            f"  fingerprints: {fingerprints or 'n/a'}\n"
            f"  hints: {hints or 'n/a'}\n"
        )

    rendered = render_prompt(
        cur,
        task="cluster_merge",
        theme="general",
        fallback_template=MERGE_PROMPT,
        values={
            "count": len(clusters),
            "clusters_text": clusters_text,
        },
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
    )
    if rendered.source == "db":
        log.info("Prompt cluster_merge: %s v%s", rendered.theme, rendered.version)
    return rendered.text


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
                    SELECT COUNT(DISTINCT a.source_name)
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
    # Only send clusters with 2+ articles to keep prompt short
    cur.execute(
        """
        SELECT c.id, c.title, c.article_count, c.source_diversity,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.topic) FILTER (WHERE ai.topic IS NOT NULL),
                 ARRAY[]::text[]
               ) AS topics,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.event_fingerprint)
                   FILTER (WHERE ai.event_fingerprint IS NOT NULL),
                 ARRAY[]::text[]
               ) AS event_fingerprints,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.cluster_hint)
                   FILTER (WHERE ai.cluster_hint IS NOT NULL),
                 ARRAY[]::text[]
               ) AS cluster_hints
        FROM clusters c
        LEFT JOIN cluster_articles ca ON ca.cluster_id = c.id
        LEFT JOIN article_intelligence ai ON ai.article_id = ca.article_id
        WHERE c.status IN ('active', 'growing')
          AND c.article_count >= 2
        GROUP BY c.id
        ORDER BY c.article_count DESC
        LIMIT 60
        """,
    )
    clusters = cur.fetchall()

    if len(clusters) < 3:
        log.info("Too few multi-article clusters for merging (%d)", len(clusters))
        return 0

    log.info("Running LLM cluster merge on %d clusters (2+ articles)...", len(clusters))
    prompt = build_merge_prompt(cur, clusters)

    result = analyze_with_deepseek(prompt, model="deepseek-v4-flash")
    if not result:
        log.info("DeepSeek failed, trying Gemini...")
        result = analyze_with_gemini(prompt)
    if not result:
        log.warning("LLM merge failed on all providers, skipping")
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
