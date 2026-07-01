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
  9b. Serendipity — arXiv science nuggets
  9c. Podcast moments — extract quotes, predictions, concepts from transcripts
  10. Optional HF enrichments — NER, classification, keywords, sentiment
"""

import json
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
from .scorer import run_scoring
from .llm_analyzer import CLUSTER_ANALYSIS_PROMPT, WEAK_SIGNAL_PROMPT, run_llm_analysis, run_weak_signal_analysis
from .signal_detector import detect_weak_signals
from .podcast_generator import generate_podcast, resolve_topics
from .serendipity_generator import run_serendipity
from .podcast_moments_extractor import run_podcast_moments_extraction
from .prediction_tracker import extract_predictions_from_cluster_analysis, extract_predictions_from_podcast_moments
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


def on_demand_podcast_topic_refs() -> list[tuple[str, str]]:
    """Parse TECHPULSE_PODCAST_TOPICS as "type:id,type:id" (type = cluster|serendipity)."""
    raw = os.getenv("TECHPULSE_PODCAST_TOPICS", "").strip()
    refs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        ttype, _, tid = chunk.partition(":")
        if ttype in ("cluster", "serendipity") and tid:
            refs.append((ttype, tid))
    return refs


def seed_prompt_registry(cur) -> None:
    from .article_intelligence import ARTICLE_INTELLIGENCE_MODEL, ARTICLE_INTELLIGENCE_PROMPT
    from .prompt_profiles import DOMAIN_PROFILES, build_impact_fields_section, build_stakeholders_hint, build_quality_hint, get_profile

    # Seed le prompt "general" (fallback)
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

    # Seed un prompt par domaine (ai, macroeconomics, space, energy...)
    # Chaque domaine a ses propres champs d'impact, stakeholders et quality hints
    for domain_key, profile in DOMAIN_PROFILES.items():
        if domain_key == "general":
            continue  # déjà seedé ci-dessus

        domain_label = profile["label"]
        impact_fields = build_impact_fields_section(domain_key)
        stakeholders_hint = build_stakeholders_hint(domain_key)
        quality_hint = build_quality_hint(domain_key)

        # Article intelligence — variante par domaine
        domain_article_prompt = ARTICLE_INTELLIGENCE_PROMPT.replace(
            "{domain_label}", domain_label
        ).replace(
            "{impact_fields}", impact_fields
        ).replace(
            "{stakeholders_hint}", stakeholders_hint
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

        # Cluster analysis — variante par domaine
        impact_lines = []
        for field_name, field_desc in profile["impact_fields"]:
            impact_lines.append(f'- "{field_name}": {field_desc}, or null when not relevant')
        impact_fields_str = "\n".join(impact_lines)

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

    podcast_topic_refs = on_demand_podcast_topic_refs()
    if podcast_topic_refs:
        target_minutes = int(os.getenv("TECHPULSE_PODCAST_MINUTES", "10") or "10")
        log.info("On-demand podcast requested for %d topic(s), target %dmin",
                  len(podcast_topic_refs), target_minutes)
        with db.get_cursor() as cur:
            topics = resolve_topics(cur, podcast_topic_refs)
            podcast_id = generate_podcast(
                cur, podcast_type="on_demand",
                topics=topics, target_minutes=target_minutes,
            )
            log.info("On-demand podcast: %s", podcast_id or "FAILED")
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

        # ── Step 6: LLM Analysis ──
        log.info("Step 6: Running LLM analysis on top clusters...")
        with db.get_cursor() as cur:
            analyses = run_llm_analysis(cur, limit=10)
            stats["analyses_generated"] = analyses

        # ── Step 6b: Extract predictions from cluster analyses ──
        if os.getenv("TECHPULSE_PREDICTIONS_ENABLED", "1") not in ("0", "false", "False"):
            def predictions_step():
                with db.get_cursor() as cur:
                    cur.execute(
                        """SELECT aa.target_id, aa.content, c.title
                           FROM ai_analyses aa
                           JOIN clusters c ON c.id = aa.target_id
                           WHERE aa.target_type = 'cluster'
                             AND aa.analysis_type = 'full'
                             AND aa.created_at > NOW() - INTERVAL '1 hour'
                        """
                    )
                    rows = cur.fetchall()
                    pred_count = 0
                    for row in rows:
                        content = row["content"] if isinstance(row["content"], dict) else json.loads(row["content"] or "{}")
                        pred_count += extract_predictions_from_cluster_analysis(
                            cur, row["target_id"], row["title"], content
                        )
                    stats["predictions_extracted"] = pred_count
                    if pred_count:
                        log.info("[Predictions] Extracted %d predictions from cluster analyses", pred_count)
            run_optional_enrichment("Step 6b: Predictions extraction", predictions_step)

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

        # ── Step 9b: Sérendipité scientifique (pépites arXiv vulgarisées) ──
        if os.getenv("SERENDIPITY_ENABLED", "1") not in ("0", "false", "False"):
            def serendipity_step():
                with db.get_cursor() as cur:
                    run_serendipity(cur)
            run_optional_enrichment("Step 9b: Serendipity", serendipity_step)

        # ── Step 9c: Podcast moments extraction (citations, prédictions, concepts) ──
        if os.getenv("TECHPULSE_PODCAST_MOMENTS_ENABLED", "1") not in ("0", "false", "False"):
            def podcast_moments_step():
                with db.get_cursor() as cur:
                    moments_count = run_podcast_moments_extraction(cur)
                    stats["podcast_moments_extracted"] = moments_count

                    # Extract predictions from podcast moments
                    cur.execute(
                        """SELECT aa.target_id, aa.content, a.title, a.source_name
                           FROM ai_analyses aa
                           JOIN articles a ON a.id = aa.target_id
                           WHERE aa.target_type = 'article'
                             AND aa.analysis_type = 'podcast_moments'
                             AND aa.created_at > NOW() - INTERVAL '1 hour'
                        """
                    )
                    rows = cur.fetchall()
                    pred_count = 0
                    for row in rows:
                        content = row["content"] if isinstance(row["content"], dict) else json.loads(row["content"] or "{}")
                        pred_count += extract_predictions_from_podcast_moments(
                            cur, row["targetId"], row["title"], row["sourceName"] or "", content
                        )
                    if pred_count:
                        stats["predictions_from_podcasts"] = pred_count
                        log.info("[Predictions] Extracted %d predictions from podcast moments", pred_count)
            run_optional_enrichment("Step 9c: Podcast moments", podcast_moments_step)

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
