"""Clustering engine using pgvector similarity in Neon."""

import logging
import os
import re
import numpy as np

from . import db

log = logging.getLogger(__name__)

SAME_EVENT_THRESHOLD = float(os.getenv("TECHPULSE_SAME_EVENT_THRESHOLD", "0.84"))
SAME_THEME_THRESHOLD = float(os.getenv("TECHPULSE_SAME_THEME_THRESHOLD", "0.68"))
NEW_TOPIC_THRESHOLD = 0.60
MAX_CLUSTER_SIZE = 12

TITLE_STOPWORDS = {
    "about", "after", "again", "ahead", "amid", "and", "are", "back", "been",
    "but", "can", "for", "from", "has", "have", "how", "into", "its", "new",
    "now", "off", "our", "over", "says", "the", "their", "this", "through",
    "under", "was", "what", "when", "where", "will", "with", "your",
    "tech", "technology", "artificial", "intelligence", "models", "model",
    "company", "companies", "market", "markets", "business", "future",
}

BRAND_TOKENS = {
    "adyen", "alphabet", "amazon", "anthropic", "apple", "aws", "bloomberg",
    "deepseek", "google", "meta", "microsoft", "nasa", "nvidia", "openai",
    "oracle", "spacex", "stripe", "tesla",
}


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


def title_tokens(title: str | None) -> set[str]:
    """Extract distinctive title tokens used as a guard for broad themes."""
    if not title:
        return set()
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.'&-]{1,}", title.lower()):
        clean = token.strip(".'&-").removesuffix("'s")
        if len(clean) < 3 or clean in TITLE_STOPWORDS:
            continue
        tokens.add(clean)
    return tokens


def _list_tokens(values) -> set[str]:
    tokens = set()
    if not values:
        return tokens
    for value in values:
        if isinstance(value, dict):
            value = value.get("name")
        tokens.update(title_tokens(str(value) if value else ""))
    return tokens


def article_signal_tokens(article: dict) -> set[str]:
    tokens = set()
    tokens.update(title_tokens(article.get("canonical_title") or article.get("title")))
    tokens.update(title_tokens(article.get("primary_domain")))
    tokens.update(title_tokens(article.get("topic")))
    tokens.update(title_tokens(article.get("event_fingerprint")))
    tokens.update(title_tokens(article.get("cluster_hint")))
    tokens.update(_list_tokens(article.get("subtopics")))
    tokens.update(_list_tokens(article.get("entities")))
    tokens.update(_list_tokens(article.get("keywords")))
    tokens.update(_list_tokens(article.get("tags")))
    return tokens


def lexical_anchor_score(article_tokens: set[str], cluster_tokens: set[str]) -> int:
    score = 0
    for token in article_tokens & cluster_tokens:
        score += 1 if token in BRAND_TOKENS else 2
    return score


def passes_lexical_guard(article: dict, cluster_tokens: set[str],
                         similarity: float, founder_similarity: float) -> bool:
    """Avoid merging broad semantic neighbors that do not share title anchors."""
    if similarity >= SAME_EVENT_THRESHOLD or founder_similarity >= SAME_EVENT_THRESHOLD:
        return True

    article_tokens = article_signal_tokens(article)
    anchor_score = lexical_anchor_score(article_tokens, cluster_tokens)
    if anchor_score >= 2:
        return True

    has_brand_overlap = bool((article_tokens & cluster_tokens) & BRAND_TOKENS)
    return has_brand_overlap and max(similarity, founder_similarity) >= 0.74


def run_clustering(cur) -> tuple[int, int]:
    """Cluster processed articles using pgvector similarity.

    Anti-snowball measures:
      - Clusters are capped at MAX_CLUSTER_SIZE articles
      - New articles must be similar to BOTH the centroid AND the
        founding article (prevents centroid drift)
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
                "title": c["title"],
                "tokens": title_tokens(c["title"])
                | _list_tokens(c.get("primary_domains"))
                | _list_tokens(c.get("topics"))
                | _list_tokens(c.get("event_fingerprints"))
                | _list_tokens(c.get("cluster_hints")),
                "event_fingerprints": set(c.get("event_fingerprints") or []),
                "primary_domains": set(c.get("primary_domains") or []),
                "centroid": parse_embedding(c["centroid_str"]),
                "founder_embedding": parse_embedding(c.get("founder_embedding_str") or c["centroid_str"]),
                "article_count": c["article_count"],
                "source_names": set(c.get("source_names") or []),
            }

    created = 0
    updated = 0

    for article in articles:
        if not article["embedding_str"]:
            continue

        emb = parse_embedding(article["embedding_str"])
        best_cluster_id = None
        best_similarity = 0.0
        best_accept_threshold = SAME_THEME_THRESHOLD

        for cid, cdata in cluster_data.items():
            # Skip full clusters
            if cdata["article_count"] >= MAX_CLUSTER_SIZE:
                continue

            sim = cosine_similarity(emb, cdata["centroid"])
            if sim > best_similarity:
                # Also check similarity with the founding article
                founder_sim = cosine_similarity(emb, cdata["founder_embedding"])
                article_fingerprint = article.get("event_fingerprint")
                same_event = bool(
                    article_fingerprint
                    and article_fingerprint in cdata.get("event_fingerprints", set())
                )
                domain_overlap = bool(
                    article.get("primary_domain")
                    and article["primary_domain"] in cdata.get("primary_domains", set())
                )
                dynamic_threshold = SAME_THEME_THRESHOLD
                if same_event:
                    dynamic_threshold = min(dynamic_threshold, 0.62)
                elif domain_overlap and article.get("topic"):
                    dynamic_threshold = min(dynamic_threshold, 0.66)

                guard_ok = same_event or passes_lexical_guard(
                    article, cdata.get("tokens", set()), sim, founder_sim
                )
                if founder_sim >= dynamic_threshold and guard_ok:
                    best_similarity = sim
                    best_cluster_id = cid
                    best_accept_threshold = dynamic_threshold

        if best_similarity >= best_accept_threshold and best_cluster_id:
            role = "primary" if best_similarity >= SAME_EVENT_THRESHOLD else "supporting"
            db.update_article_cluster(cur, article["id"], best_cluster_id)
            db.insert_cluster_article(cur, best_cluster_id, article["id"], float(best_similarity), role)

            cdata = cluster_data[best_cluster_id]
            cdata["article_count"] += 1
            cdata["source_names"].add(article["source_name"])
            cdata["tokens"].update(article_signal_tokens(article))
            if article.get("event_fingerprint"):
                cdata["event_fingerprints"].add(article["event_fingerprint"])
            if article.get("primary_domain"):
                cdata["primary_domains"].add(article["primary_domain"])

            # Update centroid (slow drift — weighted toward founder)
            count = cdata["article_count"]
            alpha = 1.0 / count
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
                cdata["article_count"], len(cdata["source_names"]),
            )
            updated += 1

        else:
            # Create new cluster — this article becomes the founder
            new_id = db.gen_id()
            title = article.get("cluster_hint") or article.get("canonical_title") or article["title"]
            db.create_cluster(cur, new_id, title[:200], emb)
            db.update_article_cluster(cur, article["id"], new_id)
            db.insert_cluster_article(cur, new_id, article["id"], 1.0, "primary")

            cluster_data[new_id] = {
                "title": title[:200],
                "tokens": article_signal_tokens(article),
                "event_fingerprints": {article["event_fingerprint"]} if article.get("event_fingerprint") else set(),
                "primary_domains": {article["primary_domain"]} if article.get("primary_domain") else set(),
                "centroid": emb,
                "founder_embedding": emb,
                "article_count": 1,
                "source_names": {article["source_name"]},
            }
            created += 1

    log.info("Clustering done: %d created, %d updated", created, updated)
    return created, updated
