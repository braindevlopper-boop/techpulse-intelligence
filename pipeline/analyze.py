"""Main intelligence pipeline — entry point for GitHub Actions.

Steps (post D1 migration):
  1. Fetch recent articles from the Worker (D1 bridge)
  2. Article Intelligence — lightweight LLM enrichment (impact/reliability/etc.)
  3. Clustering — URL/title/keyword dedup (no embeddings)
  4. Cluster merging — LLM-suggested merges of near-duplicate clusters
  5. LLM Analysis — analyze top clusters, push cluster_analyses to D1
  6. Notifications — push via FCM

Optional/legacy steps (scoring, weak signals, podcasts, serendipity, NER,
classification, sentiment, keyword extraction, prediction tracking) are
disabled: they depended on Neon-only tables/columns with no D1 equivalent in
the /pipeline/* contract. Each disabled module logs a warning and returns a
no-op so this orchestrator doesn't need special-casing per step.
"""

import logging
import os
import sys

from . import db
from .article_intelligence import run_article_intelligence
from .clusterer import CLUSTER_WINDOW_HOURS, run_clustering
from .cluster_merger import run_cluster_merging
from .llm_analyzer import run_llm_analysis
from .notifier import notify_pipeline_complete
from .prompt_lab import propose_and_evaluate_prompt
from .prompt_registry import seed_default_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("intelligence")


def prompt_lab_task() -> str:
    return os.getenv("TECHPULSE_PROMPT_LAB_TASK", "").strip()


def seed_prompt_registry(cur) -> None:
    from .article_intelligence import ARTICLE_INTELLIGENCE_MODEL, ARTICLE_INTELLIGENCE_PROMPT
    from .llm_analyzer import CLUSTER_ANALYSIS_PROMPT
    from .cluster_merger import MERGE_PROMPT
    from .prompt_profiles import DOMAIN_PROFILES, build_impact_fields_section, build_stakeholders_hint, build_quality_hint

    seed_default_prompt(
        cur,
        task="article_intelligence",
        template=ARTICLE_INTELLIGENCE_PROMPT,
        model_provider="deepseek",
        model_name=ARTICLE_INTELLIGENCE_MODEL,
    )
    seed_default_prompt(
        cur,
        task="cluster_analysis",
        template=CLUSTER_ANALYSIS_PROMPT,
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
    )

    for domain_key, profile in DOMAIN_PROFILES.items():
        if domain_key == "general":
            continue  # déjà seedé ci-dessus

        domain_label = profile["label"]
        quality_hint = build_quality_hint(domain_key)

        domain_article_prompt = ARTICLE_INTELLIGENCE_PROMPT.replace(
            "{domain_label}", domain_label
        ).replace(
            "{quality_hint}", quality_hint
        )
        seed_default_prompt(
            cur,
            task="article_intelligence",
            theme=domain_key,
            template=domain_article_prompt,
            model_provider="deepseek",
            model_name=ARTICLE_INTELLIGENCE_MODEL,
        )

        impact_fields_str = build_impact_fields_section(domain_key)
        stakeholders_hint = build_stakeholders_hint(domain_key)
        domain_cluster_prompt = CLUSTER_ANALYSIS_PROMPT.replace(
            "{domain_label}", domain_label
        ).replace(
            "{impact_fields}", impact_fields_str
        ).replace(
            "{stakeholders_hint}", stakeholders_hint
        ).replace(
            "{quality_hint}", quality_hint
        )
        seed_default_prompt(
            cur,
            task="cluster_analysis",
            theme=domain_key,
            template=domain_cluster_prompt,
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
        )

    seed_default_prompt(
        cur,
        task="cluster_merge",
        template=MERGE_PROMPT,
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
    )


def run_optional_enrichment(step_name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        log.warning("%s skipped after failure: %s", step_name, exc, exc_info=True)


def run():
    log.info("=" * 60)
    log.info("TechPulse Intelligence Pipeline — Starting")
    log.info("=" * 60)

    prompt_task = prompt_lab_task()
    if prompt_task:
        prompt_theme = os.getenv("TECHPULSE_PROMPT_LAB_THEME", "general").strip() or "general"
        prompt_goal = os.getenv(
            "TECHPULSE_PROMPT_LAB_GOAL",
            "Améliorer la profondeur, la fiabilité et la différenciation UX sans augmenter fortement le coût.",
        ).strip()
        log.info("Prompt Lab requested for %s/%s", prompt_task, prompt_theme)
        seed_prompt_registry(None)
        propose_and_evaluate_prompt(
            None,
            task=prompt_task,
            theme=prompt_theme,
            improvement_goal=prompt_goal,
        )
        return

    run_id = db.insert_pipeline_run(None, "intelligence")
    seed_prompt_registry(None)

    stats = {
        "articles_fetched": 0,
        "articles_enriched": 0,
        "clusters_created": 0,
        "clusters_updated": 0,
        "analyses_generated": 0,
    }

    try:
        # ── Step 1: Fetch recent articles from the Worker ──
        log.info("Step 1: Fetching recent articles from Worker...")
        articles = db.fetch_processed_articles(hours=CLUSTER_WINDOW_HOURS, limit=500)
        stats["articles_fetched"] = len(articles)
        log.info("Fetched %d articles", len(articles))

        # ── Step 2: Article Intelligence (lightweight enrichment) ──
        log.info("Step 2: Running article intelligence...")
        stats["articles_enriched"] = run_article_intelligence(None, articles)

        # ── Step 3: Clustering (URL/title/keyword dedup, no embeddings) ──
        log.info("Step 3: Clustering articles...")
        created, updated = run_clustering(articles)
        stats["clusters_created"] = created
        stats["clusters_updated"] = updated

        # ── Step 4: LLM cluster merging (Pass 2) ──
        # run_clustering() already pushed clusters to D1; re-fetch the fresh
        # article→cluster_id mapping so merging works off the latest state.
        log.info("Step 4: Merging similar clusters (LLM)...")
        articles_after_cluster = db.fetch_processed_articles(hours=CLUSTER_WINDOW_HOURS, limit=500)
        clusters_by_id: dict[str, dict] = {}
        for a in articles_after_cluster:
            cid = a.get("cluster_id")
            if not cid:
                continue
            c = clusters_by_id.setdefault(cid, {
                "id": cid,
                "title": a.get("title") or "",
                "theme": a.get("classified_theme") or a.get("theme") or "",
                "dedup_title": a.get("title") or "",
                "keywords_json": [],
                "article_hashes": [],
                "founder_hash": a["hash"],
                "status": "active",
                "_articles": [],
            })
            c["article_hashes"].append(a["hash"])
            c["_articles"].append(a)

        cluster_list = list(clusters_by_id.values())
        merged_clusters, merged_count = run_cluster_merging(cluster_list)
        if merged_count:
            # Re-push the merged shape (drop the local-only _articles key).
            push_payload = [
                {k: v for k, v in c.items() if k != "_articles"}
                for c in merged_clusters
            ]
            db.push_clusters(push_payload)
        log.info("Merged %d cluster groups", merged_count)

        # ── Step 5: LLM Analysis on top clusters ──
        log.info("Step 5: Running LLM analysis on top clusters...")
        analyses = run_llm_analysis(merged_clusters, limit=10)
        stats["analyses_generated"] = analyses

        # ── Step 6: Finalize + Notify ──
        db.complete_pipeline_run(None, run_id, stats)
        notify_pipeline_complete(stats)

        log.info("=" * 60)
        log.info("Core intelligence pipeline complete: %s", stats)
        log.info("=" * 60)

    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        db.fail_pipeline_run(run_id, str(e))
        sys.exit(1)


if __name__ == "__main__":
    run()
