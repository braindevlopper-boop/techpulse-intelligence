"""Structured LLM parsing for individual articles.

This replaces the old HF-first enrichment path for classification, entities,
keywords and sentiment. The output is stored before clustering so the clusterer
can use business/topic/entity signals in addition to embeddings.
"""

import logging
import os
import re
from datetime import date

from . import db
from .llm_analyzer import analyze_with_deepseek, analyze_with_gemini, analyze_with_openai

log = logging.getLogger(__name__)

ARTICLE_INTELLIGENCE_MODEL = os.getenv("TECHPULSE_ARTICLE_LLM_MODEL", "deepseek-v4-flash")
ARTICLE_INTELLIGENCE_LIMIT = int(os.getenv("TECHPULSE_ARTICLE_LLM_LIMIT", "40"))

ARTICLE_INTELLIGENCE_PROMPT = """Tu es l'analyste d'ingestion de TechPulse.

Objectif: transformer un article brut en métadonnées structurées fiables pour une application de veille technologique, financière et économique.

Article:
- Source: {source_name} ({source_type})
- Date: {published_at}
- URL: {url}
- Titre: {title}
- Description: {description}
- Texte:
{text}

Réponds uniquement avec un JSON valide, sans markdown.

Schéma attendu:
{{
  "language": "fr" | "en" | "other",
  "canonical_title": "titre nettoyé et factuel, sans HTML, clickbait ni source",
  "summary": "résumé en français en 1-2 phrases",
  "article_type": "news" | "analysis" | "opinion" | "research" | "press_release" | "market_note" | "tutorial" | "social_discussion" | "other",
  "primary_domain": "ai" | "software" | "semiconductors" | "cloud" | "cybersecurity" | "fintech" | "crypto" | "markets" | "macroeconomics" | "energy" | "space" | "defense" | "regulation" | "startups" | "consumer_tech" | "gaming" | "other",
  "topic": "thème court et normalisé, ex: spacex ipo, openai aws, ai capex, nvidia chips",
  "subtopics": ["0 à 5 sous-thèmes"],
  "event_fingerprint": "clé stable en anglais, lowercase, 3-8 mots, pour regrouper le même événement exact",
  "event_date": "YYYY-MM-DD ou null",
  "companies": ["entreprises citées"],
  "people": ["personnes citées"],
  "products": ["produits, modèles, technologies"],
  "sectors": ["secteurs"],
  "countries": ["pays ou zones"],
  "entities": [
    {{"name": "nom", "type": "company|person|product|technology|country|sector|concept", "role": "main|mentioned", "confidence": 0.0}}
  ],
  "keywords": ["5 à 10 mots-clés normalisés"],
  "tags": ["3 à 8 tags courts"],
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "sentiment_score": -1.0,
  "tech_impact": "impact technique en français, ou null",
  "business_impact": "impact business en français, ou null",
  "finance_impact": "impact marché/finance en français, ou null",
  "market_impact": "low" | "medium" | "high" | "unknown",
  "quality_score": 0,
  "relevance_score": 0,
  "novelty_score": 0,
  "time_sensitivity": "low" | "medium" | "high",
  "should_cluster": true,
  "cluster_hint": "titre court du cluster idéal",
  "confidence": 0.0
}}

Règles:
- Les scores quality/relevance/novelty sont entre 0 et 100.
- sentiment_score est entre -1 et 1.
- should_cluster=false seulement pour contenu hors sujet, trop vide, doublon technique évident ou page non informative.
- event_fingerprint doit regrouper seulement le même événement, pas un thème large.
- Si une information est absente, mets null ou [].
"""


def _clean_text(value: str | None, limit: int) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit]


def _build_prompt(article: dict) -> str:
    published = article.get("published_at")
    published_at = str(published)[:10] if published else "unknown"
    text = _clean_text(article.get("full_text") or article.get("description"), 3500)
    return ARTICLE_INTELLIGENCE_PROMPT.format(
        source_name=article.get("source_name") or "unknown",
        source_type=article.get("source_type") or "unknown",
        published_at=published_at,
        url=article.get("url") or "",
        title=_clean_text(article.get("title"), 300),
        description=_clean_text(article.get("description"), 800),
        text=text,
    )


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _as_int(value, default: int = 0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0, min_value: float = -1.0,
              max_value: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _date_or_none(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return None


def normalize_result(result: dict, article: dict) -> dict:
    canonical_title = result.get("canonical_title") or article.get("title")
    event_fingerprint = result.get("event_fingerprint")
    if event_fingerprint:
        event_fingerprint = re.sub(r"[^a-z0-9]+", "-", str(event_fingerprint).lower()).strip("-")

    normalized = {
        "language": result.get("language") or article.get("language") or "unknown",
        "canonical_title": _clean_text(canonical_title, 240),
        "summary": _clean_text(result.get("summary"), 800),
        "article_type": result.get("article_type") or "other",
        "primary_domain": result.get("primary_domain") or "other",
        "topic": _clean_text(result.get("topic"), 120),
        "subtopics": _as_list(result.get("subtopics"))[:5],
        "event_fingerprint": event_fingerprint,
        "event_date": _date_or_none(result.get("event_date")),
        "companies": _as_list(result.get("companies"))[:12],
        "people": _as_list(result.get("people"))[:12],
        "products": _as_list(result.get("products"))[:12],
        "sectors": _as_list(result.get("sectors"))[:12],
        "countries": _as_list(result.get("countries"))[:12],
        "entities": _as_list(result.get("entities"))[:20],
        "keywords": _as_list(result.get("keywords"))[:12],
        "tags": _as_list(result.get("tags"))[:10],
        "sentiment": result.get("sentiment") or "neutral",
        "sentiment_score": _as_float(result.get("sentiment_score")),
        "tech_impact": _clean_text(result.get("tech_impact"), 500),
        "business_impact": _clean_text(result.get("business_impact"), 500),
        "finance_impact": _clean_text(result.get("finance_impact"), 500),
        "market_impact": result.get("market_impact") or "unknown",
        "quality_score": _as_int(result.get("quality_score"), 50),
        "relevance_score": _as_int(result.get("relevance_score"), 50),
        "novelty_score": _as_int(result.get("novelty_score"), 50),
        "time_sensitivity": result.get("time_sensitivity") or "medium",
        "should_cluster": bool(result.get("should_cluster", True)),
        "cluster_hint": _clean_text(result.get("cluster_hint"), 180),
        "confidence": _as_float(result.get("confidence"), default=0.5, min_value=0.0, max_value=1.0),
    }
    if not normalized["cluster_hint"]:
        normalized["cluster_hint"] = normalized["canonical_title"]
    return normalized


def analyze_article(article: dict) -> tuple[dict | None, str, str]:
    prompt = _build_prompt(article)
    result = analyze_with_deepseek(prompt, model=ARTICLE_INTELLIGENCE_MODEL)
    if result:
        return normalize_result(result, article), "deepseek", ARTICLE_INTELLIGENCE_MODEL

    result = analyze_with_gemini(prompt)
    if result:
        return normalize_result(result, article), "gemini", "gemini-3.1-flash-lite"

    result = analyze_with_openai(prompt)
    if result:
        return normalize_result(result, article), "openai", "gpt-4o-mini"

    return None, "none", "none"


def _persist_entities_and_keywords(cur, article_id: str, content: dict):
    typed_entities = []
    for ent in content.get("entities") or []:
        if isinstance(ent, dict) and ent.get("name"):
            typed_entities.append(ent)

    for key, entity_type in (
        ("companies", "company"),
        ("people", "person"),
        ("products", "product"),
        ("sectors", "sector"),
        ("countries", "country"),
    ):
        for name in content.get(key) or []:
            typed_entities.append({
                "name": name,
                "type": entity_type,
                "role": "mentioned",
                "confidence": content.get("confidence") or 0.6,
            })

    seen = set()
    for ent in typed_entities:
        name = str(ent.get("name", "")).strip()
        if len(name) < 2:
            continue
        normalized = name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        entity_id = db.upsert_entity(cur, name, ent.get("type") or "concept")
        db.insert_article_entity(
            cur,
            article_id,
            entity_id,
            role=ent.get("role") or "mentioned",
            confidence=_as_float(ent.get("confidence"), default=0.6, min_value=0.0, max_value=1.0),
            source="article_llm",
        )

    for keyword in [*content.get("keywords", []), *content.get("tags", [])]:
        keyword_text = str(keyword).strip()
        if len(keyword_text) < 3:
            continue
        db.upsert_keyword(
            cur,
            keyword_text,
            category=content.get("primary_domain") or "concept",
            source="article_llm",
            reason=f"from article intelligence: {content.get('canonical_title', '')[:50]}",
        )


def run_article_intelligence(cur, limit: int = ARTICLE_INTELLIGENCE_LIMIT) -> int:
    articles = db.fetch_articles_for_llm_intelligence(cur, limit=limit)
    if not articles:
        log.info("No articles need LLM intelligence")
        return 0

    enriched = 0
    for article in articles:
        try:
            content, provider, model = analyze_article(article)
            if not content:
                db.mark_article_llm_failed(cur, article["id"], "article intelligence returned no JSON")
                continue

            db.upsert_article_intelligence(cur, article["id"], provider, model, content)
            _persist_entities_and_keywords(cur, article["id"], content)
            enriched += 1
            log.info("Article intelligence [%s]: %s", provider, content["canonical_title"][:80])
        except Exception as exc:
            log.error("Article intelligence failed for %s: %s", article["id"], exc, exc_info=True)
            db.mark_article_llm_failed(cur, article["id"], str(exc))

    log.info("Article intelligence enriched %d/%d articles", enriched, len(articles))
    return enriched
