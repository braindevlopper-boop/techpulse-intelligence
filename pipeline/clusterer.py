"""Clustering engine — simple dedup over D1 articles, no embeddings.

Three signals decide if a new article joins an existing (recent) cluster:
  1. Canonical URL match (same normalized URL → same cluster, always).
  2. Title similarity (Jaccard over normalized tokens) against the cluster's
     founder title, within the time window.
  3. Keyword/theme overlap as a secondary signal alongside title similarity.

The BRAND_TOKENS/lexical-guard idea from the embedding-era clusterer was
already pure text matching, so it's reused as-is here as an anti-false-positive
guard: a merge only goes through if the title/keyword similarity is either
strong on its own, or moderate with a shared brand/anchor token.
"""

import logging
import os
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from datetime import datetime, timezone

from . import db

log = logging.getLogger(__name__)

CLUSTER_WINDOW_HOURS = int(os.getenv("TECHPULSE_CLUSTER_WINDOW_HOURS", "72"))
TITLE_SIM_THRESHOLD = float(os.getenv("TECHPULSE_TITLE_SIM_THRESHOLD", "0.62"))
KEYWORD_SIM_THRESHOLD = float(os.getenv("TECHPULSE_KEYWORD_SIM_THRESHOLD", "0.5"))
MAX_CLUSTER_SIZE = 12

TITLE_STOPWORDS = {
    # English
    "about", "after", "again", "ahead", "amid", "and", "are", "back", "been",
    "but", "can", "for", "from", "has", "have", "how", "into", "its", "new",
    "now", "off", "our", "over", "says", "the", "their", "this", "through",
    "under", "was", "what", "when", "where", "will", "with", "your",
    "tech", "technology", "artificial", "intelligence", "models", "model",
    "company", "companies", "market", "markets", "business", "future",
    "ai", "global", "latest", "news", "developments", "development",
    "industry", "industries", "financial", "finance", "policy", "policies",
    "governance", "regulation", "regulatory", "challenge", "challenges",
    "announcement", "announcements", "update", "updates",
    "biggest", "calendar", "coverage", "live", "recap", "showcase",
    "storylines", "trailer", "trailers",
    # French
    "les", "des", "sur", "pour", "avec", "dans", "une", "un", "sont", "est",
    "par", "cette", "plus", "vers", "aux", "son", "sa", "ses", "que", "qui",
    "nouveau", "nouvelle", "actualite", "actualites", "marche", "marches",
    "entreprise", "entreprises",
}

BRAND_TOKENS = {
    "adyen", "alphabet", "amazon", "anthropic", "apple", "aws", "bloomberg",
    "deepseek", "google", "meta", "microsoft", "nasa", "nvidia", "openai",
    "oracle", "spacex", "stripe", "tesla",
}

TRACKING_QUERY_PREFIXES = ("utm_", "ref", "fbclid", "gclid", "mc_", "icid", "cmp")


def canonical_url(url: str | None) -> str | None:
    """Normalize a URL for exact dedup: lowercase host, strip tracking params
    and trailing slash, drop fragment."""
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None

    scheme = "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parts.path.rstrip("/") or ""

    kept_query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not any(k.lower().startswith(p) for p in TRACKING_QUERY_PREFIXES)
    ]
    query = urlencode(sorted(kept_query))

    return urlunsplit((scheme, netloc, path, query, ""))


def title_tokens(title: str | None) -> set[str]:
    """Normalize a title into a set of distinctive tokens (used for both
    Jaccard similarity and the brand/anchor guard)."""
    if not title:
        return set()
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.'&-]{1,}", title.lower()):
        clean = token.strip(".'&-").removesuffix("'s")
        if clean.isdigit() and len(clean) == 4:
            continue
        if len(clean) < 3 or clean in TITLE_STOPWORDS:
            continue
        tokens.add(clean)
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _parse_keywords(raw) -> set[str]:
    """keywords_json comes back as a JSON string or already-parsed list."""
    if not raw:
        return set()
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return set()
    if not isinstance(raw, list):
        return set()
    tokens = set()
    for kw in raw:
        if isinstance(kw, dict):
            kw = kw.get("name") or kw.get("keyword")
        if not kw:
            continue
        tokens.update(title_tokens(str(kw)))
    return tokens


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def article_theme(article: dict) -> str:
    return (article.get("classified_theme") or article.get("theme") or "").strip().lower()


def lexical_anchor_score(article_tokens: set[str], cluster_tokens: set[str]) -> int:
    score = 0
    for token in article_tokens & cluster_tokens:
        score += 2 if token in BRAND_TOKENS else 1
    return score


def passes_lexical_guard(article_tokens: set[str], cluster_tokens: set[str],
                         title_sim: float, keyword_sim: float) -> bool:
    """Anti-false-positive guard, reused as-is from the embedding-era clusterer
    (it was already pure text matching, no embeddings involved)."""
    anchor_score = lexical_anchor_score(article_tokens, cluster_tokens)
    best_sim = max(title_sim, keyword_sim)

    if anchor_score >= 2 and best_sim >= 0.7:
        return True

    has_brand_overlap = bool((article_tokens & cluster_tokens) & BRAND_TOKENS)
    if has_brand_overlap and anchor_score >= 3 and best_sim >= 0.55:
        return True

    if anchor_score >= 3 and best_sim >= 0.6:
        return True

    return anchor_score >= 4 and best_sim >= 0.5


class ClusterCandidate:
    """In-memory view of a cluster being built/extended during this run."""

    __slots__ = ("id", "title", "theme", "dedup_title", "founder_hash",
                 "title_tokens", "keyword_tokens", "canonical_urls",
                 "article_hashes", "latest_at")

    def __init__(self, id_: str, title: str, theme: str, dedup_title: str,
                 founder_hash: str):
        self.id = id_
        self.title = title
        self.theme = theme
        self.dedup_title = dedup_title
        self.founder_hash = founder_hash
        self.title_tokens = title_tokens(dedup_title)
        self.keyword_tokens: set[str] = set()
        self.canonical_urls: set[str] = set()
        self.article_hashes: list[str] = []
        self.latest_at: datetime | None = None

    def add(self, article: dict, url_norm: str | None):
        self.article_hashes.append(article["hash"])
        self.keyword_tokens.update(_parse_keywords(article.get("keywords_json")))
        if url_norm:
            self.canonical_urls.add(url_norm)
        pub = _parse_time(article.get("published_at")) or _parse_time(article.get("fetched_at"))
        if pub and (self.latest_at is None or pub > self.latest_at):
            self.latest_at = pub


def run_clustering(articles: list[dict]) -> tuple[int, int]:
    """Dedup+cluster recent articles using URL/title/keyword signals.

    `articles` should already be scoped to the clustering window (see
    db.fetch_processed_articles(hours=CLUSTER_WINDOW_HOURS)).

    Returns (clusters_created, clusters_updated_with_new_articles).
    """
    if not articles:
        log.info("No articles to cluster")
        return 0, 0

    log.info("Clustering %d articles (window=%dh)", len(articles), CLUSTER_WINDOW_HOURS)

    # Sort oldest → newest so earlier articles become founders of their cluster.
    def sort_key(a):
        return _parse_time(a.get("published_at")) or _parse_time(a.get("fetched_at")) or datetime.min.replace(tzinfo=timezone.utc)

    articles = sorted(articles, key=sort_key)

    # Articles already carry cluster_id from previous runs (denormalized by
    # the Worker). Re-hydrate those as existing ClusterCandidates first, using
    # their real id — only articles with cluster_id=None are up for grouping.
    # Without this, every run would mint brand-new random cluster ids and
    # duplicate every cluster instead of growing them across the 3x/day cron.
    by_url: dict[str, ClusterCandidate] = {}
    clusters_by_id: dict[str, ClusterCandidate] = {}
    unclustered: list[dict] = []
    created = 0
    updated = 0

    for article in articles:
        existing_cluster_id = article.get("cluster_id")
        if not existing_cluster_id:
            unclustered.append(article)
            continue

        url_norm = canonical_url(article.get("url"))
        cand = clusters_by_id.get(existing_cluster_id)
        if cand is None:
            title = article.get("title") or ""
            cand = ClusterCandidate(
                id_=existing_cluster_id,
                title=title[:200],
                theme=article_theme(article),
                dedup_title=title[:200],
                founder_hash=article["hash"],
            )
            clusters_by_id[existing_cluster_id] = cand
        cand.add(article, url_norm)
        if url_norm:
            by_url[url_norm] = cand

    clusters: list[ClusterCandidate] = list(clusters_by_id.values())

    for article in unclustered:
        url_norm = canonical_url(article.get("url"))
        article_kw = _parse_keywords(article.get("keywords_json"))
        article_title_tokens = title_tokens(article.get("title"))
        theme = article_theme(article)

        # 1. Exact canonical URL match — always the same cluster.
        if url_norm and url_norm in by_url:
            target = by_url[url_norm]
            target.add(article, url_norm)
            updated += 1
            continue

        # 2/3. Title similarity + keyword/theme overlap against active clusters
        # in the time window, guarded by the lexical anchor check.
        best_cluster = None
        best_score = 0.0

        for cand in clusters:
            if len(cand.article_hashes) >= MAX_CLUSTER_SIZE:
                continue

            title_sim = jaccard(article_title_tokens, cand.title_tokens)
            keyword_sim = jaccard(article_kw, cand.keyword_tokens)
            same_theme = bool(theme and theme == cand.theme)

            qualifies = title_sim >= TITLE_SIM_THRESHOLD or (
                same_theme and keyword_sim >= KEYWORD_SIM_THRESHOLD
            )
            if not qualifies:
                continue

            guard_ok = passes_lexical_guard(
                article_title_tokens | article_kw,
                cand.title_tokens | cand.keyword_tokens,
                title_sim, keyword_sim,
            )
            if not guard_ok:
                continue

            score = max(title_sim, keyword_sim if same_theme else 0.0)
            if score > best_score:
                best_score = score
                best_cluster = cand

        if best_cluster:
            best_cluster.add(article, url_norm)
            if url_norm:
                by_url[url_norm] = best_cluster
            updated += 1
        else:
            new_id = db.gen_id()
            dedup_title = article.get("title") or ""
            cand = ClusterCandidate(
                id_=new_id,
                title=dedup_title[:200],
                theme=theme,
                dedup_title=dedup_title[:200],
                founder_hash=article["hash"],
            )
            cand.add(article, url_norm)
            clusters.append(cand)
            if url_norm:
                by_url[url_norm] = cand
            created += 1

    payload = []
    for cand in clusters:
        payload.append({
            "id": cand.id,
            "title": cand.title,
            "theme": cand.theme or None,
            "dedup_title": cand.dedup_title,
            "keywords_json": list(cand.keyword_tokens)[:20],
            "article_hashes": cand.article_hashes,
            "founder_hash": cand.founder_hash,
            "status": "active",
        })

    if payload:
        result = db.push_clusters(payload)
        log.info("Pushed %d clusters to Worker (%s)", len(payload), result)

    log.info("Clustering done: %d new clusters, %d articles merged into existing/new clusters", created, updated)
    return created, updated
