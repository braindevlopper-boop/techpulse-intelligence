"""Main intelligence pipeline — entry point for GitHub Actions.

Steps:
  1. Article Intelligence — LLM structured parsing/classification
  2. Clustering — group articles by similarity and article signals
  3. Cluster merging — merge duplicate stories
  4. Scoring — compute importance, growth, novelty
  5. Entity relationships — build graph edges from cluster evidence
  6. LLM Analysis — analyze top clusters with Gemini/OpenAI
  7. Weak signals — detect emerging topics
  8. Podcast — generate daily audio (Edge TTS)
  9. Notifications — push via FCM
  10. Optional HF enrichments — NER, classification, keywords, sentiment
"""

import logging
import os
import sys

from . import db
from .ner_extractor import run_ner
from .classifier import run_classification
from .keyword_extractor import run_keyword_extraction
from .sentiment_analyzer import run_sentiment_analysis
from .article_intelligence import run_article_intelligence
from .clusterer import run_clustering
from .cluster_merger import MERGE_PROMPT, run_cluster_merging
from .entity_relationships import build_entity_relationships
from .scorer import run_scoring
from .llm_analyzer import CLUSTER_ANALYSIS_PROMPT, WEAK_SIGNAL_PROMPT, run_llm_analysis, run_weak_signal_analysis
from .signal_detector import detect_weak_signals
from .podcast_generator import generate_podcast
from .notifier import notify_pipeline_complete, notify_weak_signal
from .prompt_lab import propose_and_evaluate_prompt
from .prompt_registry import seed_default_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("intelligence")


def should_recluster_all() -> bool:
    return os.getenv("TECHPULSE_RECLUSTER_ALL", "").strip().lower() in {"1", "true", "yes", "on"}


def should_skip_hf_ml() -> bool:
    return os.getenv("TECHPULSE_SKIP_HF_ML", "").strip().lower() in {"1", "true", "yes", "on"}


def should_skip_legacy_podcast() -> bool:
    return os.getenv("TECHPULSE_SKIP_LEGACY_PODCAST", "").strip().lower() in {"1", "true", "yes", "on"}


def prompt_lab_task() -> str:
    return os.getenv("TECHPULSE_PROMPT_LAB_TASK", "").strip()


def seed_prompt_registry(cur) -> None:
    from .article_intelligence import ARTICLE_INTELLIGENCE_MODEL, ARTICLE_INTELLIGENCE_PROMPT

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
    seed_default_prompt(
        cur,
        task="weak_signal_analysis",
        template=WEAK_SIGNAL_PROMPT,
        model_provider="grok",
        model_name="grok-4.3",
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
        with db.get_cursor() as cur:
            seed_prompt_registry(cur)
            candidate_id = propose_and_evaluate_prompt(
                cur,
                task=prompt_task,
                theme=prompt_theme,
                improvement_goal=prompt_goal,
            )
            log.info("Prompt Lab candidate: %s", candidate_id)
        return

    with db.get_cursor() as cur:
        run_id = db.insert_pipeline_run(cur, "intelligence")
        seed_prompt_registry(cur)

    stats = {
        "articles_fetched": 0,
        "articles_embedded": 0,
        "articles_enriched": 0,
        "clusters_created": 0,
        "clusters_updated": 0,
        "analyses_generated": 0,
    }

    try:
        if should_recluster_all():
            log.warning("TECHPULSE_RECLUSTER_ALL enabled: rebuilding all cluster-derived data")
            with db.get_cursor() as cur:
                reset_stats = db.reset_clusters_for_rebuild(cur)
                log.warning("Cluster rebuild reset: %s", reset_stats)

        # ── Step 1: Article Intelligence ──
        log.info("Step 1: Running article intelligence...")
        with db.get_cursor() as cur:
            stats["articles_enriched"] = run_article_intelligence(cur)

        # ── Step 2: Clustering ──
        # Keep the core veille product path before optional HF enrichments.
        log.info("Step 2: Clustering articles...")
        with db.get_cursor() as cur:
            repair_stats = db.repair_cluster_integrity(cur)
            log.info("Cluster integrity before clustering: %s", repair_stats)
            created, updated = run_clustering(cur)
            stats["clusters_created"] = created
            stats["clusters_updated"] = updated

        # ── Step 3: LLM cluster merging (Pass 2) ──
        log.info("Step 3: Merging similar clusters (LLM)...")
        with db.get_cursor() as cur:
            merged = run_cluster_merging(cur)
            repair_stats = db.repair_cluster_integrity(cur)
            log.info("Merged %d cluster groups", merged)
            log.info("Cluster integrity after merging: %s", repair_stats)

        # ── Step 4: Scoring ──
        log.info("Step 4: Scoring clusters...")
        with db.get_cursor() as cur:
            run_scoring(cur)

        # ── Step 5: Entity relationship graph ──
        log.info("Step 5: Building entity relationship graph...")
        with db.get_cursor() as cur:
            build_entity_relationships(cur)

        # ── Step 6: LLM Analysis ──
        log.info("Step 6: Running LLM analysis on top clusters...")
        with db.get_cursor() as cur:
            analyses = run_llm_analysis(cur, limit=15)
            stats["analyses_generated"] = analyses

        # ── Step 7: Weak signals (rule-based detection) ──
        log.info("Step 7: Detecting weak signals...")
        with db.get_cursor() as cur:
            signals = detect_weak_signals(cur)
            for signal in signals[:3]:
                notify_weak_signal(signal["title"], signal["growth_score"])

        # ── Step 8: Grok deep signal analysis (1x/day, premium) ──
        log.info("Step 8: Running Grok deep signal analysis...")
        with db.get_cursor() as cur:
            run_weak_signal_analysis(cur)

        # ── Step 9: Podcast ──
        if should_skip_legacy_podcast():
            log.info("Step 9: Legacy intelligence podcast skipped")
        else:
            log.info("Step 9: Generating podcast...")
            with db.get_cursor() as cur:
                generate_podcast(cur)

        # ── Step 10: Finalize + Notify ──
        with db.get_cursor() as cur:
            db.complete_pipeline_run(cur, run_id, stats)

        notify_pipeline_complete(stats)

        log.info("=" * 60)
        log.info("Core intelligence pipeline complete: %s", stats)
        log.info("=" * 60)

        def enrich_ner():
            with db.get_cursor() as cur:
                articles_ner = db.fetch_articles_for_ner(cur)
                if articles_ner:
                    run_ner(cur, articles_ner)

        def enrich_classification():
            with db.get_cursor() as cur:
                articles_cls = db.fetch_articles_for_classification(cur)
                if articles_cls:
                    run_classification(cur, articles_cls)

        def enrich_keywords():
            with db.get_cursor() as cur:
                articles_kw = db.fetch_articles_for_keywords(cur)
                if articles_kw:
                    run_keyword_extraction(cur, articles_kw)

        def enrich_sentiment():
            with db.get_cursor() as cur:
                articles_sent = db.fetch_articles_for_sentiment(cur)
                if articles_sent:
                    run_sentiment_analysis(cur, articles_sent)

        if should_skip_hf_ml():
            log.info("Step 11: Optional Hugging Face enrichments skipped")
        else:
            log.info("Step 11: Optional Hugging Face enrichments...")
            run_optional_enrichment("NER", enrich_ner)
            run_optional_enrichment("Classification", enrich_classification)
            run_optional_enrichment("Keyword extraction", enrich_keywords)
            run_optional_enrichment("Sentiment", enrich_sentiment)

    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        try:
            with db.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'failed', completed_at = NOW(),
                        errors = %s::jsonb,
                        duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::int
                    WHERE id = %s
                    """,
                    (f'["{e}"]', run_id),
                )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    run()
