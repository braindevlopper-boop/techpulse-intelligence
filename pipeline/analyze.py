"""Main intelligence pipeline — entry point for GitHub Actions.

Steps:
  1. NER — extract entities from articles
  2. Classification — zero-shot categorization
  3. KeyBERT — extract keywords
  4. Sentiment — analyze article/comment sentiment
  5. Clustering — group articles by similarity (pgvector)
  6. Scoring — compute importance, growth, novelty
  7. LLM Analysis — analyze top clusters with Gemini/OpenAI
  8. Weak signals — detect emerging topics
  9. Podcast — generate daily audio (Edge TTS)
  10. Notifications — push via FCM
"""

import logging
import os
import sys

from . import db
from .ner_extractor import run_ner
from .classifier import run_classification
from .keyword_extractor import run_keyword_extraction
from .sentiment_analyzer import run_sentiment_analysis
from .clusterer import run_clustering
from .cluster_merger import run_cluster_merging
from .scorer import run_scoring
from .llm_analyzer import run_llm_analysis, run_weak_signal_analysis
from .signal_detector import detect_weak_signals
from .podcast_generator import generate_podcast
from .notifier import notify_pipeline_complete, notify_weak_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("intelligence")


def should_recluster_all() -> bool:
    return os.getenv("TECHPULSE_RECLUSTER_ALL", "").strip().lower() in {"1", "true", "yes", "on"}


def run():
    log.info("=" * 60)
    log.info("TechPulse Intelligence Pipeline — Starting")
    log.info("=" * 60)

    with db.get_cursor() as cur:
        run_id = db.insert_pipeline_run(cur, "intelligence")

    stats = {
        "articles_fetched": 0,
        "articles_embedded": 0,
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

        # ── Step 1: Clustering ──
        # Keep the core veille product path before optional HF enrichments.
        log.info("Step 1: Clustering articles...")
        with db.get_cursor() as cur:
            repair_stats = db.repair_cluster_integrity(cur)
            log.info("Cluster integrity before clustering: %s", repair_stats)
            created, updated = run_clustering(cur)
            stats["clusters_created"] = created
            stats["clusters_updated"] = updated

        # ── Step 2: LLM cluster merging (Pass 2) ──
        log.info("Step 2: Merging similar clusters (LLM)...")
        with db.get_cursor() as cur:
            merged = run_cluster_merging(cur)
            repair_stats = db.repair_cluster_integrity(cur)
            log.info("Merged %d cluster groups", merged)
            log.info("Cluster integrity after merging: %s", repair_stats)

        # ── Step 3: Scoring ──
        log.info("Step 3: Scoring clusters...")
        with db.get_cursor() as cur:
            run_scoring(cur)

        # ── Step 4: LLM Analysis ──
        log.info("Step 4: Running LLM analysis on top clusters...")
        with db.get_cursor() as cur:
            analyses = run_llm_analysis(cur, limit=15)
            stats["analyses_generated"] = analyses

        # ── Step 5: Weak signals (rule-based detection) ──
        log.info("Step 5: Detecting weak signals...")
        with db.get_cursor() as cur:
            signals = detect_weak_signals(cur)
            for signal in signals[:3]:
                notify_weak_signal(signal["title"], signal["growth_score"])

        # ── Step 6: Grok deep signal analysis (1x/day, premium) ──
        log.info("Step 6: Running Grok deep signal analysis...")
        with db.get_cursor() as cur:
            run_weak_signal_analysis(cur)

        # ── Step 7: Podcast ──
        log.info("Step 7: Generating podcast...")
        with db.get_cursor() as cur:
            generate_podcast(cur)

        # ── Step 8: NER ──
        log.info("Step 8: Extracting entities (NER)...")
        with db.get_cursor() as cur:
            articles_ner = db.fetch_articles_for_ner(cur)
            if articles_ner:
                run_ner(cur, articles_ner)

        # ── Step 9: Classification ──
        log.info("Step 9: Classifying articles (zero-shot)...")
        with db.get_cursor() as cur:
            articles_cls = db.fetch_articles_for_classification(cur)
            if articles_cls:
                run_classification(cur, articles_cls)

        # ── Step 10: KeyBERT ──
        log.info("Step 10: Extracting keywords (KeyBERT)...")
        with db.get_cursor() as cur:
            articles_kw = db.fetch_articles_for_keywords(cur)
            if articles_kw:
                run_keyword_extraction(cur, articles_kw)

        # ── Step 11: Sentiment ──
        log.info("Step 11: Analyzing sentiment...")
        with db.get_cursor() as cur:
            articles_sent = db.fetch_articles_for_sentiment(cur)
            if articles_sent:
                run_sentiment_analysis(cur, articles_sent)

        # ── Step 12: Finalize + Notify ──
        with db.get_cursor() as cur:
            db.complete_pipeline_run(cur, run_id, stats)

        notify_pipeline_complete(stats)

        log.info("=" * 60)
        log.info("Intelligence pipeline complete: %s", stats)
        log.info("=" * 60)

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
