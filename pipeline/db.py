"""Database connection and helpers for the intelligence pipeline."""

import os
import uuid
import json
from contextlib import contextmanager

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


# ── Article queries ──

def fetch_processed_articles(cur, limit: int = 500) -> list[dict]:
    """Articles with embeddings but not yet clustered."""
    cur.execute(
        """
        SELECT id, title, description, full_text, source_type, source_name,
               published_at, external_score, embedding::text as embedding_str
        FROM articles
        WHERE status = 'processed' AND embedding IS NOT NULL
        ORDER BY fetched_at DESC
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


def update_article_cluster(cur, article_id: str, cluster_id: str):
    cur.execute(
        "UPDATE articles SET cluster_id = %s, status = 'clustered' WHERE id = %s",
        (cluster_id, article_id),
    )


# ── Cluster queries ──

def fetch_active_clusters(cur) -> list[dict]:
    """Get all active clusters with their centroids."""
    cur.execute(
        """
        SELECT id, title, centroid::text as centroid_str,
               article_count, source_diversity, importance_score
        FROM clusters
        WHERE status IN ('active', 'growing')
        ORDER BY last_updated_at DESC
        """
    )
    return cur.fetchall()


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
        ORDER BY (c.importance_score + c.growth_score * 10 + c.novelty_score * 15) DESC
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
            clusters_created = %(clusters_created)s,
            clusters_updated = %(clusters_updated)s,
            analyses_generated = %(analyses_generated)s,
            duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::int
        WHERE id = %(run_id)s
        """,
        {**stats, "run_id": run_id},
    )
