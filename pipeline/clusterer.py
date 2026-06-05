"""Clustering engine using pgvector similarity in Neon."""

import logging
import json
import numpy as np

from . import db

log = logging.getLogger(__name__)

SAME_EVENT_THRESHOLD = 0.88
SAME_THEME_THRESHOLD = 0.78
NEW_TOPIC_THRESHOLD = 0.70


def parse_embedding(embedding_str: str) -> list[float]:
    """Parse a pgvector text representation back to a list of floats."""
    clean = embedding_str.strip("[]")
    return [float(x) for x in clean.split(",")]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    arr = np.array(embeddings)
    centroid = arr.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


def run_clustering(cur) -> tuple[int, int]:
    """Cluster processed articles using pgvector similarity.

    Returns (clusters_created, clusters_updated).
    """
    articles = db.fetch_processed_articles(cur)
    if not articles:
        log.info("No articles to cluster")
        return 0, 0

    clusters = db.fetch_active_clusters(cur)
    log.info("Clustering %d articles against %d existing clusters", len(articles), len(clusters))

    cluster_data = {}
    for c in clusters:
        if c["centroid_str"]:
            cluster_data[c["id"]] = {
                "centroid": parse_embedding(c["centroid_str"]),
                "article_count": c["article_count"],
                "source_types": set(),
            }

    created = 0
    updated = 0

    for article in articles:
        if not article["embedding_str"]:
            continue

        emb = parse_embedding(article["embedding_str"])
        best_cluster_id = None
        best_similarity = 0.0

        for cid, cdata in cluster_data.items():
            sim = cosine_similarity(emb, cdata["centroid"])
            if sim > best_similarity:
                best_similarity = sim
                best_cluster_id = cid

        if best_similarity >= SAME_THEME_THRESHOLD and best_cluster_id:
            role = "primary" if best_similarity >= SAME_EVENT_THRESHOLD else "supporting"
            db.update_article_cluster(cur, article["id"], best_cluster_id)
            db.insert_cluster_article(cur, best_cluster_id, article["id"], best_similarity, role)

            cdata = cluster_data[best_cluster_id]
            cdata["article_count"] += 1
            cdata["source_types"].add(article["source_type"])

            alpha = 1.0 / cdata["article_count"]
            new_centroid = [
                (1 - alpha) * c + alpha * e
                for c, e in zip(cdata["centroid"], emb)
            ]
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = [x / norm for x in new_centroid]
            cdata["centroid"] = new_centroid

            db.update_cluster_centroid(
                cur, best_cluster_id, new_centroid,
                cdata["article_count"], len(cdata["source_types"]),
            )
            updated += 1

        elif best_similarity < NEW_TOPIC_THRESHOLD or not best_cluster_id:
            new_id = db.gen_id()
            db.create_cluster(cur, new_id, article["title"][:200], emb)
            db.update_article_cluster(cur, article["id"], new_id)
            db.insert_cluster_article(cur, new_id, article["id"], 1.0, "primary")

            cluster_data[new_id] = {
                "centroid": emb,
                "article_count": 1,
                "source_types": {article["source_type"]},
            }
            created += 1

        else:
            new_id = db.gen_id()
            db.create_cluster(cur, new_id, article["title"][:200], emb)
            db.update_article_cluster(cur, article["id"], new_id)
            db.insert_cluster_article(cur, new_id, article["id"], 1.0, "primary")

            cluster_data[new_id] = {
                "centroid": emb,
                "article_count": 1,
                "source_types": {article["source_type"]},
            }
            created += 1

    log.info("Clustering done: %d created, %d updated", created, updated)
    return created, updated
