"""Pass 2 — LLM-based cluster merging.

After the URL/title/keyword dedup pass (clusterer.py), some clusters still
cover the same story under different wording. This sends cluster titles to
DeepSeek V4 Flash and asks it to group them; merges are re-pushed to the
Worker as a single cluster with the combined article_hashes.

Cost: ~$0.002 per run (one API call with all cluster titles).
"""

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
- Ne crée jamais de panier large comme "Global AI governance", "latest developments", "financial news" ou "industry challenges".
- OpenAI policy, Trump executive order, US House AI bill et EU sovereignty = liés mais différents → NE PAS fusionner.
- SpaceX IPO price, SpaceX revenue forecast et Google compute deal = liés mais différents → NE PAS fusionner.
- Summer Game Fest, Xbox Showcase et GTA VI release calendar = liés gaming mais différents → NE PAS fusionner.
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
        clusters_text += (
            f"- ID: {c['id']} | Articles: {len(c.get('article_hashes') or [])} | \"{c['title']}\"\n"
            f"  theme: {c.get('theme') or 'n/a'}\n"
        )

    rendered = render_prompt(
        None,
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


def run_cluster_merging(clusters: list[dict]) -> tuple[list[dict], int]:
    """Ask the LLM to suggest merges, apply them in-memory, return the final
    cluster list (post-merge) plus how many merges were applied.

    `clusters` is the same payload shape pushed by clusterer.py (id, title,
    theme, dedup_title, keywords_json, article_hashes, founder_hash, status).
    """
    multi_article = [c for c in clusters if len(c.get("article_hashes") or []) >= 1]
    if len(multi_article) < 3:
        log.info("Too few clusters for merging (%d)", len(multi_article))
        return clusters, 0

    candidates = sorted(clusters, key=lambda c: len(c.get("article_hashes") or []), reverse=True)[:60]
    log.info("Running LLM cluster merge on %d clusters...", len(candidates))
    prompt = build_merge_prompt(candidates)

    result = analyze_with_deepseek(prompt, model="deepseek-v4-flash")
    if not result:
        log.info("DeepSeek failed, trying Gemini...")
        result = analyze_with_gemini(prompt)
    if not result:
        log.warning("LLM merge failed on all providers, skipping")
        return clusters, 0

    merge_groups = result.get("merge_groups", [])
    if not merge_groups:
        log.info("LLM found no clusters to merge")
        return clusters, 0

    by_id = {c["id"]: c for c in clusters}
    merged_away = set()
    total_merged = 0

    for group in merge_groups:
        cluster_ids = [cid for cid in group.get("cluster_ids", []) if cid in by_id]
        new_title = group.get("merged_title", "")
        if len(cluster_ids) < 2 or not new_title:
            continue

        cluster_ids = [cid for cid in cluster_ids if cid not in merged_away]
        if len(cluster_ids) < 2:
            continue

        ranked = sorted(cluster_ids, key=lambda cid: len(by_id[cid].get("article_hashes") or []), reverse=True)
        target_id = ranked[0]
        source_ids = ranked[1:]

        target = by_id[target_id]
        target["title"] = new_title[:200]
        for source_id in source_ids:
            source = by_id[source_id]
            target["article_hashes"] = list(dict.fromkeys(
                (target.get("article_hashes") or []) + (source.get("article_hashes") or [])
            ))
            existing_kw = target.get("keywords_json") or []
            merged_kw = list(dict.fromkeys(existing_kw + (source.get("keywords_json") or [])))
            target["keywords_json"] = merged_kw
            merged_away.add(source_id)

        total_merged += len(source_ids)
        log.info("Merged %d clusters → \"%s\"", len(source_ids) + 1, new_title)

    remaining = [c for c in clusters if c["id"] not in merged_away]
    log.info("Cluster merging done: %d merges, %d clusters remaining", total_merged, len(remaining))
    return remaining, total_merged
