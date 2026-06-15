"""Database connection and helpers for the intelligence pipeline."""

import os
import uuid
import json
import re
from contextlib import contextmanager
from datetime import date, datetime

import psycopg2
import psycopg2.extras


def get_connection():
    return psycopg2.connect(os.environ["NEON_DATABASE_URL"], sslmode="require")


@contextmanager
def get_cursor():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def gen_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Prompt registry ──

def ensure_prompt_registry(cur):
    """Create prompt registry tables used to version and evaluate prompts."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_templates (
          id               TEXT PRIMARY KEY,
          task             TEXT NOT NULL,
          theme            TEXT NOT NULL DEFAULT 'general',
          version          INTEGER NOT NULL DEFAULT 1,
          status           TEXT NOT NULL DEFAULT 'draft',
          template         TEXT NOT NULL,
          variables        JSONB NOT NULL DEFAULT '[]'::jsonb,
          model_provider   TEXT,
          model_name       TEXT,
          parent_id        TEXT,
          quality_score    INTEGER,
          evaluator_score  INTEGER,
          evaluator_notes  JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_by       TEXT NOT NULL DEFAULT 'system',
          created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          UNIQUE(task, theme, version)
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_templates_active_unique
        ON prompt_templates(task, theme)
        WHERE status = 'active'
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prompt_templates_task_theme_status
        ON prompt_templates(task, theme, status)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_evaluations (
          id                  TEXT PRIMARY KEY,
          prompt_template_id  TEXT NOT NULL REFERENCES prompt_templates(id) ON DELETE CASCADE,
          evaluator_provider  TEXT NOT NULL,
          evaluator_model     TEXT NOT NULL,
          score               INTEGER NOT NULL,
          recommendation      TEXT NOT NULL,
          notes               JSONB NOT NULL DEFAULT '{}'::jsonb,
          sample_count        INTEGER NOT NULL DEFAULT 0,
          created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prompt_evaluations_prompt_created
        ON prompt_evaluations(prompt_template_id, created_at DESC)
        """
    )


def seed_prompt_template(
    cur,
    *,
    task: str,
    theme: str,
    template: str,
    variables: list[str],
    model_provider: str | None = None,
    model_name: str | None = None,
):
    """Insert the code fallback as version 1 if no DB prompt exists yet."""
    ensure_prompt_registry(cur)
    cur.execute(
        """
        INSERT INTO prompt_templates (
          id, task, theme, version, status, template, variables,
          model_provider, model_name, created_by
        )
        VALUES (%s, %s, %s, 1, 'active', %s, %s::jsonb, %s, %s, 'code_seed')
        ON CONFLICT (task, theme, version) DO NOTHING
        """,
        (
            gen_id(),
            task,
            theme,
            template,
            json.dumps(variables),
            model_provider,
            model_name,
        ),
    )


def fetch_active_prompt_template(cur, task: str, theme: str = "general") -> dict | None:
    ensure_prompt_registry(cur)
    cur.execute(
        """
        SELECT id, task, theme, version, template, variables,
               model_provider, model_name, quality_score, evaluator_score,
               evaluator_notes, created_by, updated_at
        FROM prompt_templates
        WHERE task = %s
          AND status = 'active'
          AND theme IN (%s, 'general')
        ORDER BY CASE WHEN theme = %s THEN 0 ELSE 1 END, version DESC
        LIMIT 1
        """,
        (task, theme, theme),
    )
    return cur.fetchone()


def insert_prompt_candidate(
    cur,
    *,
    task: str,
    theme: str,
    template: str,
    variables: list[str],
    parent_id: str | None,
    model_provider: str,
    model_name: str,
    evaluator_notes: dict | None = None,
) -> str:
    ensure_prompt_registry(cur)
    cur.execute(
        """
        SELECT COALESCE(MAX(version), 0) + 1 AS next_version
        FROM prompt_templates
        WHERE task = %s AND theme = %s
        """,
        (task, theme),
    )
    version = cur.fetchone()["next_version"]
    prompt_id = gen_id()
    cur.execute(
        """
        INSERT INTO prompt_templates (
          id, task, theme, version, status, template, variables,
          model_provider, model_name, parent_id, evaluator_notes, created_by
        )
        VALUES (%s, %s, %s, %s, 'candidate', %s, %s::jsonb, %s, %s, %s, %s::jsonb, 'llm')
        """,
        (
            prompt_id,
            task,
            theme,
            version,
            template,
            json.dumps(variables),
            model_provider,
            model_name,
            parent_id,
            json.dumps(evaluator_notes or {}),
        ),
    )
    return prompt_id


def insert_prompt_evaluation(
    cur,
    *,
    prompt_template_id: str,
    evaluator_provider: str,
    evaluator_model: str,
    score: int,
    recommendation: str,
    notes: dict,
    sample_count: int = 0,
):
    ensure_prompt_registry(cur)
    cur.execute(
        """
        INSERT INTO prompt_evaluations (
          id, prompt_template_id, evaluator_provider, evaluator_model,
          score, recommendation, notes, sample_count
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        """,
        (
            gen_id(),
            prompt_template_id,
            evaluator_provider,
            evaluator_model,
            max(0, min(100, int(score))),
            recommendation,
            json.dumps(notes or {}),
            sample_count,
        ),
    )


def _safe_iso_date(value) -> str | None:
    """Normalize exact ISO dates and drop ambiguous LLM dates."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw or raw.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return None
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return None


# ── Article queries ──

def fetch_processed_articles(cur, limit: int = 500) -> list[dict]:
    """Articles with embeddings but not yet clustered."""
    cur.execute(
        """
        SELECT a.id, a.title, a.description, a.full_text, a.source_type, a.source_name,
               a.published_at, a.external_score, a.embedding::text as embedding_str,
               a.category, a.sentiment,
               ai.canonical_title, ai.primary_domain, ai.topic, ai.subtopics,
               ai.event_fingerprint, ai.entities, ai.keywords, ai.tags,
               ai.quality_score, ai.relevance_score, ai.should_cluster,
               ai.cluster_hint
        FROM articles a
        LEFT JOIN article_intelligence ai ON ai.article_id = a.id
        WHERE a.status = 'processed'
          AND a.embedding IS NOT NULL
          AND COALESCE(ai.should_cluster, true) = true
        ORDER BY a.fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def fetch_articles_for_llm_intelligence(cur, limit: int = 40) -> list[dict]:
    """Embedded articles that still need structured LLM metadata."""
    cur.execute(
        """
        SELECT a.id, a.title, a.description, a.full_text, a.source_name,
               a.source_type, a.published_at, a.external_score, a.url
        FROM articles a
        LEFT JOIN article_intelligence ai ON ai.article_id = a.id
        WHERE a.embedding IS NOT NULL
          AND a.status IN ('processed', 'clustered', 'analyzed')
          AND ai.article_id IS NULL
          AND COALESCE(a.llm_enrichment_status, 'pending') <> 'failed'
        ORDER BY a.fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def fetch_articles_for_ner(cur, limit: int = 300) -> list[dict]:
    """Articles that need entity extraction."""
    cur.execute(
        """
        SELECT a.id, a.title, a.description, a.full_text
        FROM articles a
        LEFT JOIN article_entities ae ON ae.article_id = a.id
        WHERE a.status IN ('processed', 'clustered')
          AND ae.id IS NULL
        ORDER BY a.fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def fetch_articles_for_classification(cur, limit: int = 300) -> list[dict]:
    """Articles without a category."""
    cur.execute(
        """
        SELECT id, title, description
        FROM articles
        WHERE category IS NULL
          AND status IN ('processed', 'clustered')
        ORDER BY fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def fetch_articles_for_keywords(cur, limit: int = 100) -> list[dict]:
    """Recent processed articles used for keyword discovery."""
    cur.execute(
        """
        SELECT id, title, description, full_text
        FROM articles
        WHERE status IN ('processed', 'clustered', 'analyzed')
        ORDER BY fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def fetch_articles_for_sentiment(cur, limit: int = 300) -> list[dict]:
    """Articles without sentiment analysis."""
    cur.execute(
        """
        SELECT id, title, description
        FROM articles
        WHERE sentiment IS NULL
          AND status IN ('processed', 'clustered')
        ORDER BY fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def reset_clusters_for_rebuild(cur) -> dict:
    """Drop cluster-derived data and mark embedded articles for reclustering."""
    cur.execute("DELETE FROM trend_snapshots WHERE cluster_id IS NOT NULL")
    trend_snapshots = cur.rowcount

    cur.execute("DELETE FROM timeline_events")
    timeline_events = cur.rowcount

    cur.execute(
        """
        DELETE FROM ai_analyses
        WHERE target_type = 'cluster'
           OR (target_type = 'daily_digest' AND target_id = 'weak_signals')
        """
    )
    analyses = cur.rowcount

    cur.execute("DELETE FROM cluster_articles")
    cluster_articles = cur.rowcount

    cur.execute("DELETE FROM clusters")
    clusters = cur.rowcount

    cur.execute(
        """
        UPDATE articles
        SET cluster_id = NULL,
            status = 'processed'
        WHERE embedding IS NOT NULL
        """
    )
    articles_reset = cur.rowcount

    return {
        "trend_snapshots": trend_snapshots,
        "timeline_events": timeline_events,
        "analyses": analyses,
        "cluster_articles": cluster_articles,
        "clusters": clusters,
        "articles_reset": articles_reset,
    }


def update_article_category(cur, article_id: str, category: str, confidence: float):
    cur.execute(
        "UPDATE articles SET category = %s, category_confidence = %s WHERE id = %s",
        (category, confidence, article_id),
    )


def update_article_sentiment(cur, article_id: str, sentiment: str, score: float):
    cur.execute(
        "UPDATE articles SET sentiment = %s, sentiment_score = %s WHERE id = %s",
        (sentiment, score, article_id),
    )


def mark_article_llm_failed(cur, article_id: str, error: str):
    cur.execute(
        """
        UPDATE articles
        SET llm_enrichment_status = 'failed',
            last_error = %s,
            retry_count = COALESCE(retry_count, 0) + 1,
            last_processed_at = NOW()
        WHERE id = %s
        """,
        (error[:500], article_id),
    )


def update_article_cluster(cur, article_id: str, cluster_id: str):
    cur.execute(
        "DELETE FROM cluster_articles WHERE article_id = %s",
        (article_id,),
    )
    cur.execute(
        """
        UPDATE articles
        SET cluster_id = %s,
            status = 'clustered',
            pipeline_status = 'clustered',
            clustering_status = 'clustered',
            last_error = NULL,
            last_processed_at = NOW()
        WHERE id = %s
        """,
        (cluster_id, article_id),
    )


# ── Cluster queries ──

def fetch_active_clusters(cur) -> list[dict]:
    """Get all active clusters with their centroids."""
    cur.execute(
        """
        SELECT c.id, c.title, c.centroid::text as centroid_str,
               c.article_count, c.source_diversity, c.importance_score,
               (
                 SELECT fa.embedding::text
                 FROM cluster_articles fca
                 JOIN articles fa ON fa.id = fca.article_id
                 WHERE fca.cluster_id = c.id AND fa.embedding IS NOT NULL
                 ORDER BY (fca.role = 'primary') DESC,
                          fca.similarity_score DESC NULLS LAST,
                          fca.created_at ASC
                 LIMIT 1
               ) AS founder_embedding_str,
               COALESCE(
                 ARRAY_AGG(DISTINCT a.source_name) FILTER (WHERE a.source_name IS NOT NULL),
                 ARRAY[]::text[]
               ) AS source_names,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.primary_domain) FILTER (WHERE ai.primary_domain IS NOT NULL),
                 ARRAY[]::text[]
               ) AS primary_domains,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.topic) FILTER (WHERE ai.topic IS NOT NULL),
                 ARRAY[]::text[]
               ) AS topics,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.event_fingerprint) FILTER (WHERE ai.event_fingerprint IS NOT NULL),
                 ARRAY[]::text[]
               ) AS event_fingerprints,
               COALESCE(
                 ARRAY_AGG(DISTINCT ai.cluster_hint) FILTER (WHERE ai.cluster_hint IS NOT NULL),
                 ARRAY[]::text[]
               ) AS cluster_hints
        FROM clusters c
        LEFT JOIN articles a ON a.cluster_id = c.id
        LEFT JOIN article_intelligence ai ON ai.article_id = a.id
        WHERE c.status IN ('active', 'growing')
        GROUP BY c.id
        ORDER BY c.last_updated_at DESC
        """
    )
    return cur.fetchall()


def repair_cluster_integrity(cur) -> dict:
    """Repair cluster/article links before scoring or reclustering.

    The article table has a single cluster_id, so cluster_articles must follow
    the same rule. This prevents old links from previous runs from inflating
    cluster counts or leaving empty clusters active.
    """
    cur.execute(
        """
        WITH ranked AS (
          SELECT ca.id,
                 ROW_NUMBER() OVER (
                   PARTITION BY ca.article_id
                   ORDER BY (ca.cluster_id = a.cluster_id) DESC,
                            ca.similarity_score DESC NULLS LAST,
                            ca.created_at DESC
                 ) AS rn
          FROM cluster_articles ca
          JOIN articles a ON a.id = ca.article_id
        )
        DELETE FROM cluster_articles ca
        USING ranked r
        WHERE ca.id = r.id AND r.rn > 1
        """
    )
    duplicate_links = cur.rowcount

    cur.execute(
        """
        DELETE FROM cluster_articles ca
        USING articles a
        WHERE ca.article_id = a.id
          AND a.cluster_id IS NOT NULL
          AND ca.cluster_id <> a.cluster_id
        """
    )
    stale_links = cur.rowcount

    cur.execute(
        """
        INSERT INTO cluster_articles (id, cluster_id, article_id, similarity_score, role)
        SELECT md5(a.cluster_id || ':' || a.id), a.cluster_id, a.id, 1.0, 'primary'
        FROM articles a
        WHERE a.cluster_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1
            FROM cluster_articles ca
            WHERE ca.article_id = a.id AND ca.cluster_id = a.cluster_id
          )
        ON CONFLICT (cluster_id, article_id) DO NOTHING
        """
    )
    missing_links = cur.rowcount

    cur.execute(
        """
        DELETE FROM clusters c
        WHERE c.status IN ('active', 'growing', 'peak')
          AND NOT EXISTS (
            SELECT 1 FROM cluster_articles ca WHERE ca.cluster_id = c.id
          )
        """
    )
    empty_clusters = cur.rowcount

    cur.execute(
        """
        WITH stats AS (
          SELECT ca.cluster_id,
                 COUNT(*) AS article_count,
                 COUNT(DISTINCT a.source_name) FILTER (WHERE a.source_name IS NOT NULL) AS source_diversity,
                 MAX(a.published_at) AS latest_article_at
          FROM cluster_articles ca
          JOIN articles a ON a.id = ca.article_id
          GROUP BY ca.cluster_id
        )
        UPDATE clusters c
        SET article_count = stats.article_count,
            source_diversity = COALESCE(stats.source_diversity, 0),
            last_updated_at = COALESCE(stats.latest_article_at, c.last_updated_at, NOW())
        FROM stats
        WHERE c.id = stats.cluster_id
        """
    )
    updated_clusters = cur.rowcount

    return {
        "duplicate_links": duplicate_links,
        "stale_links": stale_links,
        "missing_links": missing_links,
        "empty_clusters": empty_clusters,
        "updated_clusters": updated_clusters,
    }


def create_cluster(cur, cluster_id: str, title: str, centroid: list[float]):
    centroid_str = "[" + ",".join(str(x) for x in centroid) + "]"
    cur.execute(
        """
        INSERT INTO clusters (id, title, centroid, article_count, source_diversity,
                              first_seen_at, last_updated_at)
        VALUES (%s, %s, %s::vector, 1, 1, NOW(), NOW())
        """,
        (cluster_id, title, centroid_str),
    )


def update_cluster_centroid(cur, cluster_id: str, centroid: list[float],
                            article_count: int, source_diversity: int):
    centroid_str = "[" + ",".join(str(x) for x in centroid) + "]"
    cur.execute(
        """
        UPDATE clusters
        SET centroid = %s::vector,
            article_count = %s,
            source_diversity = %s,
            last_updated_at = NOW()
        WHERE id = %s
        """,
        (centroid_str, article_count, source_diversity, cluster_id),
    )


def insert_cluster_article(cur, cluster_id: str, article_id: str,
                           similarity: float, role: str = "supporting"):
    cur.execute(
        """
        INSERT INTO cluster_articles (id, cluster_id, article_id, similarity_score, role)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (cluster_id, article_id) DO NOTHING
        """,
        (gen_id(), cluster_id, article_id, similarity, role),
    )


def update_cluster_scores(cur, cluster_id: str, importance: int, growth: int, novelty: int):
    cur.execute(
        """
        UPDATE clusters
        SET importance_score = %s, growth_score = %s, novelty_score = %s
        WHERE id = %s
        """,
        (importance, growth, novelty, cluster_id),
    )


def fetch_top_clusters(cur, limit: int = 20) -> list[dict]:
    """Top clusters by combined score for LLM analysis."""
    cur.execute(
        """
        SELECT c.id, c.title, c.article_count, c.source_diversity,
               c.importance_score, c.growth_score, c.novelty_score,
               c.first_seen_at, c.last_updated_at
        FROM clusters c
        WHERE c.status IN ('active', 'growing')
          AND c.article_count >= 2
        ORDER BY (
          c.importance_score
          + LEAST(c.growth_score, 20) * 2
          + c.novelty_score * 2
        ) DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def fetch_cluster_articles(cur, cluster_id: str) -> list[dict]:
    """Get all articles in a cluster."""
    cur.execute(
        """
        SELECT a.id, a.title, a.description, a.source_name, a.source_type,
               a.published_at, a.url, a.full_text,
               ca.similarity_score, ca.role
        FROM articles a
        JOIN cluster_articles ca ON ca.article_id = a.id
        WHERE ca.cluster_id = %s
        ORDER BY ca.similarity_score DESC
        """,
        (cluster_id,),
    )
    return cur.fetchall()


# ── Entity queries ──

def find_entity_by_name(cur, normalized_name: str) -> dict | None:
    cur.execute(
        "SELECT id FROM entities WHERE normalized_name = %s",
        (normalized_name,),
    )
    row = cur.fetchone()
    return row


def upsert_entity(cur, name: str, entity_type: str) -> str:
    normalized = name.lower().strip()
    existing = find_entity_by_name(cur, normalized)
    if existing:
        cur.execute(
            """
            UPDATE entities
            SET mentions_count = mentions_count + 1, last_seen_at = NOW()
            WHERE id = %s
            """,
            (existing["id"],),
        )
        return existing["id"]

    entity_id = gen_id()
    cur.execute(
        """
        INSERT INTO entities (id, name, normalized_name, type, mentions_count,
                              first_seen_at, last_seen_at)
        VALUES (%s, %s, %s, %s, 1, NOW(), NOW())
        """,
        (entity_id, name, normalized, entity_type),
    )
    return entity_id


def insert_article_entity(cur, article_id: str, entity_id: str,
                          role: str, confidence: float, source: str = "ner"):
    cur.execute(
        """
        INSERT INTO article_entities (id, article_id, entity_id, role, confidence, source)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (article_id, entity_id) DO NOTHING
        """,
        (gen_id(), article_id, entity_id, role, confidence, source),
    )


def has_entity_relationships_table(cur) -> bool:
    cur.execute("SELECT to_regclass('public.entity_relationships') AS table_name")
    row = cur.fetchone()
    return bool(row and row.get("table_name"))


def fetch_entity_relationship_candidates(cur, limit: int = 800) -> list[dict]:
    """Build entity-pair candidates from cooccurrences inside active clusters."""
    cur.execute(
        """
        WITH cluster_entities AS (
          SELECT
            c.id AS cluster_id,
            c.title AS cluster_title,
            c.article_count,
            c.source_diversity,
            c.first_seen_at,
            c.last_updated_at,
            e.id AS entity_id,
            e.name AS entity_name,
            e.type AS entity_type,
            COUNT(DISTINCT ae.article_id) AS entity_article_count
          FROM clusters c
          JOIN cluster_articles ca ON ca.cluster_id = c.id
          JOIN article_entities ae ON ae.article_id = ca.article_id
          JOIN entities e ON e.id = ae.entity_id
          WHERE c.status IN ('active', 'growing', 'peak')
            AND e.normalized_name NOT IN (
              'ai', 'us', 'u.s.', 'u. s.', 'usa', 'uk', 'reuters', 'bloomberg',
              'bloomberg tech', 'bloomberg technology', 'hacker news',
              'techcrunch', 'the verge', 'ars technica', 'cnbc',
              'youtube', 'internet', 'technology', 'tech', 'data', 'software'
            )
          GROUP BY c.id, e.id
        ),
        pair_clusters AS (
          SELECT
            CASE WHEN ce1.entity_id < ce2.entity_id THEN ce1.entity_id ELSE ce2.entity_id END AS source_entity_id,
            CASE WHEN ce1.entity_id < ce2.entity_id THEN ce2.entity_id ELSE ce1.entity_id END AS target_entity_id,
            CASE WHEN ce1.entity_id < ce2.entity_id THEN ce1.entity_type ELSE ce2.entity_type END AS source_entity_type,
            CASE WHEN ce1.entity_id < ce2.entity_id THEN ce2.entity_type ELSE ce1.entity_type END AS target_entity_type,
            ce1.cluster_id,
            ce1.cluster_title,
            ce1.article_count,
            ce1.source_diversity,
            ce1.first_seen_at,
            ce1.last_updated_at,
            ce1.entity_article_count + ce2.entity_article_count AS entity_article_mentions
          FROM cluster_entities ce1
          JOIN cluster_entities ce2
            ON ce1.cluster_id = ce2.cluster_id
           AND ce1.entity_id < ce2.entity_id
        )
        SELECT
          source_entity_id,
          target_entity_id,
          source_entity_type,
          target_entity_type,
          COUNT(DISTINCT cluster_id) AS evidence_count,
          ARRAY_AGG(DISTINCT cluster_id) AS evidence_cluster_ids,
          COALESCE(SUM(entity_article_mentions), 0) AS article_signal,
          COALESCE(SUM(source_diversity), 0) AS source_signal,
          MIN(first_seen_at) AS first_seen_at,
          MAX(last_updated_at) AS last_seen_at,
          JSONB_AGG(
            JSONB_BUILD_OBJECT(
              'cluster_id', cluster_id,
              'title', cluster_title,
              'article_count', article_count,
              'source_diversity', source_diversity,
              'last_updated_at', last_updated_at
            )
            ORDER BY last_updated_at DESC NULLS LAST
          ) AS evidence_clusters
        FROM pair_clusters
        GROUP BY source_entity_id, target_entity_id, source_entity_type, target_entity_type
        ORDER BY
          COUNT(DISTINCT cluster_id) DESC,
          COALESCE(SUM(source_diversity), 0) DESC,
          COALESCE(SUM(entity_article_mentions), 0) DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def replace_entity_relationships(cur, relationships: list[dict]) -> int:
    """Replace graph relationships with the latest cluster-derived snapshot."""
    cur.execute("DELETE FROM entity_relationships")
    if not relationships:
        return 0

    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO entity_relationships (
          id, source_entity_id, target_entity_id, relation_type,
          strength_score, evidence_count, evidence_cluster_ids,
          evidence_article_ids, evidence_summary, first_seen_at, last_seen_at,
          updated_at
        )
        VALUES (
          %(id)s, %(source_entity_id)s, %(target_entity_id)s, %(relation_type)s,
          %(strength_score)s, %(evidence_count)s, %(evidence_cluster_ids)s::jsonb,
          %(evidence_article_ids)s::jsonb, %(evidence_summary)s::jsonb,
          %(first_seen_at)s, %(last_seen_at)s, NOW()
        )
        ON CONFLICT (source_entity_id, target_entity_id, relation_type) DO UPDATE SET
          strength_score = EXCLUDED.strength_score,
          evidence_count = EXCLUDED.evidence_count,
          evidence_cluster_ids = EXCLUDED.evidence_cluster_ids,
          evidence_article_ids = EXCLUDED.evidence_article_ids,
          evidence_summary = EXCLUDED.evidence_summary,
          first_seen_at = EXCLUDED.first_seen_at,
          last_seen_at = EXCLUDED.last_seen_at,
          updated_at = NOW()
        """,
        [
            {
                **relationship,
                "id": relationship.get("id") or gen_id(),
                "evidence_cluster_ids": json.dumps(relationship.get("evidence_cluster_ids") or []),
                "evidence_article_ids": json.dumps(relationship.get("evidence_article_ids") or []),
                "evidence_summary": json.dumps(relationship.get("evidence_summary") or {}, default=str),
            }
            for relationship in relationships
        ],
        page_size=100,
    )
    return len(relationships)


def upsert_article_intelligence(cur, article_id: str, provider: str, model: str,
                                content: dict):
    cur.execute(
        """
        INSERT INTO article_intelligence (
          id, article_id, model_provider, model_name, language,
          canonical_title, summary, article_type, primary_domain, topic,
          subtopics, event_fingerprint, event_date, entities, companies,
          people, products, sectors, countries, keywords, tags, sentiment,
          sentiment_score, tech_impact, business_impact, finance_impact,
          market_impact, quality_score, relevance_score, novelty_score,
          time_sensitivity, should_cluster, cluster_hint, confidence, raw,
          updated_at
        )
        VALUES (
          %(id)s, %(article_id)s, %(provider)s, %(model)s, %(language)s,
          %(canonical_title)s, %(summary)s, %(article_type)s, %(primary_domain)s,
          %(topic)s, %(subtopics)s::jsonb, %(event_fingerprint)s, %(event_date)s::date,
          %(entities)s::jsonb, %(companies)s::jsonb, %(people)s::jsonb,
          %(products)s::jsonb, %(sectors)s::jsonb, %(countries)s::jsonb,
          %(keywords)s::jsonb, %(tags)s::jsonb, %(sentiment)s, %(sentiment_score)s,
          %(tech_impact)s, %(business_impact)s, %(finance_impact)s,
          %(market_impact)s, %(quality_score)s, %(relevance_score)s,
          %(novelty_score)s, %(time_sensitivity)s, %(should_cluster)s,
          %(cluster_hint)s, %(confidence)s, %(raw)s::jsonb, NOW()
        )
        ON CONFLICT (article_id) DO UPDATE SET
          model_provider = EXCLUDED.model_provider,
          model_name = EXCLUDED.model_name,
          language = EXCLUDED.language,
          canonical_title = EXCLUDED.canonical_title,
          summary = EXCLUDED.summary,
          article_type = EXCLUDED.article_type,
          primary_domain = EXCLUDED.primary_domain,
          topic = EXCLUDED.topic,
          subtopics = EXCLUDED.subtopics,
          event_fingerprint = EXCLUDED.event_fingerprint,
          event_date = EXCLUDED.event_date,
          entities = EXCLUDED.entities,
          companies = EXCLUDED.companies,
          people = EXCLUDED.people,
          products = EXCLUDED.products,
          sectors = EXCLUDED.sectors,
          countries = EXCLUDED.countries,
          keywords = EXCLUDED.keywords,
          tags = EXCLUDED.tags,
          sentiment = EXCLUDED.sentiment,
          sentiment_score = EXCLUDED.sentiment_score,
          tech_impact = EXCLUDED.tech_impact,
          business_impact = EXCLUDED.business_impact,
          finance_impact = EXCLUDED.finance_impact,
          market_impact = EXCLUDED.market_impact,
          quality_score = EXCLUDED.quality_score,
          relevance_score = EXCLUDED.relevance_score,
          novelty_score = EXCLUDED.novelty_score,
          time_sensitivity = EXCLUDED.time_sensitivity,
          should_cluster = EXCLUDED.should_cluster,
          cluster_hint = EXCLUDED.cluster_hint,
          confidence = EXCLUDED.confidence,
          raw = EXCLUDED.raw,
          updated_at = NOW()
        """,
        {
            "id": gen_id(),
            "article_id": article_id,
            "provider": provider,
            "model": model,
            "language": content.get("language"),
            "canonical_title": content.get("canonical_title"),
            "summary": content.get("summary"),
            "article_type": content.get("article_type"),
            "primary_domain": content.get("primary_domain"),
            "topic": content.get("topic"),
            "subtopics": json.dumps(content.get("subtopics") or []),
            "event_fingerprint": content.get("event_fingerprint"),
            "event_date": content.get("event_date"),
            "entities": json.dumps(content.get("entities") or []),
            "companies": json.dumps(content.get("companies") or []),
            "people": json.dumps(content.get("people") or []),
            "products": json.dumps(content.get("products") or []),
            "sectors": json.dumps(content.get("sectors") or []),
            "countries": json.dumps(content.get("countries") or []),
            "keywords": json.dumps(content.get("keywords") or []),
            "tags": json.dumps(content.get("tags") or []),
            "sentiment": content.get("sentiment"),
            "sentiment_score": content.get("sentiment_score"),
            "tech_impact": content.get("tech_impact"),
            "business_impact": content.get("business_impact"),
            "finance_impact": content.get("finance_impact"),
            "market_impact": content.get("market_impact"),
            "quality_score": content.get("quality_score"),
            "relevance_score": content.get("relevance_score"),
            "novelty_score": content.get("novelty_score"),
            "time_sensitivity": content.get("time_sensitivity"),
            "should_cluster": content.get("should_cluster", True),
            "cluster_hint": content.get("cluster_hint"),
            "confidence": content.get("confidence"),
            "raw": json.dumps(content),
        },
    )

    cur.execute(
        """
        UPDATE articles
        SET category = COALESCE(%s, category),
            category_confidence = COALESCE(%s, category_confidence),
            sentiment = COALESCE(%s, sentiment),
            sentiment_score = COALESCE(%s, sentiment_score),
            language = COALESCE(%s, language),
            llm_enrichment_status = 'completed',
            llm_enriched_at = NOW(),
            llm_enrichment_model = %s,
            last_error = NULL,
            last_processed_at = NOW()
        WHERE id = %s
        """,
        (
            content.get("primary_domain"),
            content.get("confidence"),
            content.get("sentiment"),
            content.get("sentiment_score"),
            content.get("language"),
            f"{provider}:{model}",
            article_id,
        ),
    )


# ── Keywords ──

def upsert_keyword(cur, keyword: str, category: str, source: str = "keybert",
                   reason: str | None = None):
    cur.execute(
        """
        INSERT INTO keywords (id, keyword, category, source, discovery_reason,
                              first_seen_at, last_used_at, usage_count)
        VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), 1)
        ON CONFLICT (keyword) DO UPDATE
        SET usage_count = keywords.usage_count + 1,
            last_used_at = NOW()
        """,
        (gen_id(), keyword.lower().strip(), category, source, reason),
    )


# ── AI Analyses ──

def insert_analysis(cur, target_type: str, target_id: str,
                    provider: str, model: str, analysis_type: str,
                    content: dict, tokens: int = 0, cost: float = 0.0):
    cur.execute(
        """
        INSERT INTO ai_analyses (id, target_type, target_id, model_provider,
                                 model_name, analysis_type, content,
                                 tokens_used, cost_estimate)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
        """,
        (gen_id(), target_type, target_id, provider, model,
         analysis_type, json.dumps(content), tokens, cost),
    )
    if target_type == "cluster":
        cur.execute(
            """
            UPDATE articles a
            SET status = CASE WHEN a.status = 'clustered' THEN 'analyzed' ELSE a.status END,
                pipeline_status = 'analyzed',
                analysis_status = 'analyzed',
                last_error = NULL,
                last_processed_at = NOW()
            FROM cluster_articles ca
            WHERE ca.article_id = a.id
              AND ca.cluster_id = %s
            """,
            (target_id,),
        )


# ── Timeline events ──

def insert_timeline_event(cur, cluster_id: str, title: str,
                          description: str | None, event_date: str | None,
                          importance: int = 0, source_article_id: str | None = None):
    """Insert a timeline event for a cluster. Skip duplicates by title."""
    safe_event_date = _safe_iso_date(event_date)
    cur.execute(
        """
        INSERT INTO timeline_events (id, cluster_id, title, description,
                                     event_date, importance, source_article_id)
        VALUES (%s, %s, %s, %s, %s::timestamptz, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (gen_id(), cluster_id, title, description,
         safe_event_date, importance, source_article_id),
    )


# ── Podcasts ──

def insert_podcast(cur, title: str, description: str, script: str,
                   audio_url: str, duration: int, cluster_ids: list[str],
                   podcast_type: str = "daily") -> str:
    podcast_id = gen_id()
    cur.execute(
        """
        INSERT INTO podcasts (id, title, description, podcast_type, script,
                              audio_url, duration_seconds, cluster_ids, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'ready')
        """,
        (podcast_id, title, description, podcast_type, script,
         audio_url, duration, json.dumps(cluster_ids)),
    )
    return podcast_id


# ── Trend snapshots ──

def insert_trend_snapshot(cur, entity_id: str | None, keyword_id: str | None,
                          cluster_id: str | None, mention_count: int,
                          source_count: int, growth_rate: float):
    cur.execute(
        """
        INSERT INTO trend_snapshots (id, entity_id, keyword_id, cluster_id,
                                     snapshot_date, mention_count, source_count,
                                     growth_rate)
        VALUES (%s, %s, %s, %s, CURRENT_DATE, %s, %s, %s)
        """,
        (gen_id(), entity_id, keyword_id, cluster_id,
         mention_count, source_count, growth_rate),
    )


# ── Pipeline runs ──

def insert_pipeline_run(cur, pipeline_type: str) -> str:
    run_id = gen_id()
    cur.execute(
        """
        INSERT INTO pipeline_runs (id, pipeline_type, status, started_at)
        VALUES (%s, %s, 'running', NOW())
        """,
        (run_id, pipeline_type),
    )
    return run_id


def complete_pipeline_run(cur, run_id: str, stats: dict):
    cur.execute(
        """
        UPDATE pipeline_runs
        SET status = 'completed',
            completed_at = NOW(),
            articles_fetched = %(articles_fetched)s,
            articles_embedded = %(articles_embedded)s,
            articles_enriched = %(articles_enriched)s,
            clusters_created = %(clusters_created)s,
            clusters_updated = %(clusters_updated)s,
            analyses_generated = %(analyses_generated)s,
            duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::int
        WHERE id = %(run_id)s
        """,
        {**stats, "run_id": run_id},
    )


# ── Sérendipité scientifique ──

def fetch_recent_serendipity_arxiv_ids(cur, limit: int = 500) -> set[str]:
    """Identifiants arXiv déjà transformés en cartes (pour éviter les doublons)."""
    cur.execute(
        "SELECT arxiv_id FROM serendipity_cards "
        "WHERE arxiv_id IS NOT NULL ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    return {row["arxiv_id"] for row in cur.fetchall()}


def fetch_recent_serendipity_source_urls(cur, limit: int = 500) -> set[str]:
    """Sources déjà transformées en cartes, quelle que soit leur origine."""
    cur.execute(
        "SELECT source_url FROM serendipity_cards "
        "WHERE source_url IS NOT NULL ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    return {row["source_url"] for row in cur.fetchall()}


def fetch_serendipity_candidates(cur, limit: int = 80) -> list[dict]:
    """Recent TechPulse intelligence items suitable for inspiration cards."""
    cur.execute(
        """
        SELECT
          a.id AS article_id,
          a.url AS source_url,
          a.title,
          a.description,
          a.source_name,
          a.source_type,
          a.category,
          a.published_at,
          a.fetched_at,
          ai.primary_domain,
          ai.topic,
          ai.subtopics,
          ai.entities,
          ai.keywords,
          ai.tags,
          ai.quality_score,
          ai.relevance_score,
          ai.cluster_hint,
          c.id AS cluster_id,
          c.title AS cluster_title,
          c.importance_score,
          c.novelty_score,
          c.growth_score,
          ca.role AS cluster_role
        FROM articles a
        LEFT JOIN article_intelligence ai ON ai.article_id = a.id
        LEFT JOIN cluster_articles ca ON ca.article_id = a.id
        LEFT JOIN clusters c ON c.id = ca.cluster_id
        WHERE a.url IS NOT NULL
          AND a.title IS NOT NULL
          AND COALESCE(a.published_at, a.fetched_at) >= NOW() - INTERVAL '21 days'
          AND a.status IN ('processed', 'clustered', 'analyzed')
          AND (
            ai.article_id IS NOT NULL
            OR c.id IS NOT NULL
            OR LOWER(COALESCE(a.source_type, '')) = 'arxiv'
          )
          AND CONCAT_WS(' ', a.title, a.description, ai.topic, ai.primary_domain, c.title) !~*
              '(grand theft auto|final fantasy|video games?|summer game fest|multiplayer sequel|remake trilogy|entertainment/games)'
          AND CONCAT_WS(' ', a.title, a.description, ai.topic, ai.primary_domain, ai.keywords::text, ai.tags::text, c.title) ~*
              '(science|research|paper|arxiv|nature|physics|quantum|astro|space|cosmo|neuro|brain|bio|biotech|medicine|medical|genom|crispr|protein|drug|materials?|battery|energy|fusion|nuclear|robot|semiconductor|chip|climate|mathematics|algorithm|ai model|artificial intelligence|nasa|mit)'
        ORDER BY
          COALESCE(ai.relevance_score, 0) DESC,
          COALESCE(c.novelty_score, 0) DESC,
          COALESCE(a.published_at, a.fetched_at) DESC
        LIMIT %s
        """,
        (limit,),
    )
    return cur.fetchall()


def insert_serendipity_card(cur, card: dict) -> str | None:
    """Insère une carte. Les doublons arXiv sont ignorés par contrainte DB."""
    sid = gen_id()
    cur.execute(
        """
        INSERT INTO serendipity_cards (
          id, arxiv_id, source_url, domain, arxiv_category,
          title_choc, enigme, personnage, concept, so_what,
          paper_title, authors, published_at, model_provider, model_name
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO NOTHING
        RETURNING id
        """,
        (
            sid, card.get("arxiv_id"), card.get("source_url"), card.get("domain"),
            card.get("arxiv_category"), card["title_choc"], card.get("enigme"),
            card.get("personnage"), card.get("concept"), card.get("so_what"),
            card.get("paper_title"), json.dumps(card.get("authors", [])),
            card.get("published_at"), card.get("model_provider"), card.get("model_name"),
        ),
    )
    row = cur.fetchone()
    return row["id"] if row else None
