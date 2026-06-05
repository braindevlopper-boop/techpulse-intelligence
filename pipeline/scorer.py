"""Cluster scoring — compute importance, growth, novelty."""

import logging
from datetime import datetime, timezone, timedelta

from . import db

log = logging.getLogger(__name__)


def run_scoring(cur) -> int:
    """Score all active clusters and update trend snapshots."""
    clusters = db.fetch_active_clusters(cur)
    scored = 0
    now = datetime.now(timezone.utc)

    for cluster in clusters:
        cluster_id = cluster["id"]
        articles = db.fetch_cluster_articles(cur, cluster_id)

        if not articles:
            continue

        # ── Importance score ──
        article_count = len(articles)
        source_names = set(a["source_name"] for a in articles if a.get("source_name"))
        source_diversity = len(source_names)
        avg_external = sum(a.get("external_score") or 0 for a in articles) / max(article_count, 1)

        importance = (
            article_count * 3
            + source_diversity * 5
            + min(avg_external, 100) * 2
        )

        # ── Growth score ──
        recent_articles = [
            a for a in articles
            if a.get("published_at") and
            (now - a["published_at"].replace(tzinfo=timezone.utc if a["published_at"].tzinfo is None else a["published_at"].tzinfo)) < timedelta(hours=24)
        ]
        older_articles = [
            a for a in articles
            if a.get("published_at") and
            (now - a["published_at"].replace(tzinfo=timezone.utc if a["published_at"].tzinfo is None else a["published_at"].tzinfo)) >= timedelta(hours=24)
        ]

        if older_articles:
            growth = len(recent_articles) / max(len(older_articles), 1)
        else:
            growth = min(len(recent_articles) * 2, 3)

        # ── Novelty score ──
        first_seen = cluster.get("first_seen_at")
        if first_seen:
            if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=timezone.utc)
            age = now - first_seen
            if age < timedelta(hours=24):
                novelty = 15
            elif age < timedelta(hours=72):
                novelty = 8
            else:
                novelty = 0
        else:
            novelty = 15

        db.update_cluster_scores(
            cur, cluster_id,
            importance=int(importance),
            growth=int(growth * 10),
            novelty=int(novelty),
        )

        # ── Trend snapshot ──
        db.insert_trend_snapshot(
            cur,
            entity_id=None,
            keyword_id=None,
            cluster_id=cluster_id,
            mention_count=article_count,
            source_count=source_diversity,
            growth_rate=round(growth, 2),
        )

        scored += 1

    log.info("Scored %d clusters", scored)
    return scored
