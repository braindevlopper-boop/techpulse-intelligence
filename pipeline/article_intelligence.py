"""Lightweight structured LLM enrichment for individual articles.

D1 only stores 5 enrichment fields per article (impact_fr, reliability,
why_interesting_fr, score_interest, keywords_json) via
POST /pipeline/articles/enrich — the old Neon article_intelligence table
(entities/companies/people/products/sectors/countries/tags/event_fingerprint)
has no D1 equivalent, so the prompt and output were both trimmed down to
match. Clustering (clusterer.py) no longer needs any of the dropped fields:
it works from url/title/keywords_json/theme instead.
"""

import logging
import os
import re

from . import db
from .llm_analyzer import analyze_with_deepseek, analyze_with_gemini, analyze_with_openai
from .prompt_registry import render_prompt
from .prompt_profiles import detect_domain, build_quality_hint, get_profile

log = logging.getLogger(__name__)

ARTICLE_INTELLIGENCE_MODEL = os.getenv("TECHPULSE_ARTICLE_LLM_MODEL", "deepseek-v4-flash")
ARTICLE_INTELLIGENCE_LIMIT = int(os.getenv("TECHPULSE_ARTICLE_LLM_LIMIT", "60"))

ARTICLE_INTELLIGENCE_PROMPT = """Tu es l'analyste d'ingestion de TechPulse.

Objectif: transformer un article brut en métadonnées légères et fiables pour une application de veille technologique, financière et économique.

Article:
- Source: {source_name}
- Date: {published_at}
- Titre: {title}
- Texte:
{text}

Domaine détecté: {domain_label}

Réponds uniquement avec un JSON valide, sans markdown.

Schéma attendu:
{{
  "impact_fr": "1-2 phrases en français : pourquoi/comment cet article compte concrètement",
  "why_interesting_fr": "1-2 phrases en français : ce qui rend cet article notable ou différenciant",
  "reliability": "peer-reviewed" | "preprint" | "communique" | "analyse" | "presse" | "rumeur",
  "score_interest": 0,
  "keywords_json": ["5 à 8 mots-clés normalisés"]
}}

Règles:
- score_interest est un entier entre 0 et 100 (intérêt/pertinence pour un lecteur stratégique).
- {quality_hint}
- reliability reflète la nature de la source et des affirmations :
  • peer-reviewed : papier publié dans une revue à comité de lecture
  • preprint : arXiv, bioRxiv, working paper non encore publié
  • communique : communiqué officiel d'entreprise, institution, gouvernement
  • analyse : analyse, opinion argumentée, éditorial, podcast expert
  • presse : article de presse classique, news recap
  • rumeur : fuite non confirmée, rumeur, spéculation
"""


def _clean_text(value: str | None, limit: int) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit]


def _build_prompt(article: dict, cur=None) -> str:
    published = article.get("published_at")
    published_at = str(published)[:10] if published else "unknown"
    text = _clean_text(article.get("content"), 3500)
    title = _clean_text(article.get("title"), 300)
    theme = article.get("classified_theme") or article.get("theme") or ""

    domain = detect_domain(title=title, description="", theme=theme)
    profile = get_profile(domain)

    values = {
        "source_name": article.get("source_name") or "unknown",
        "published_at": published_at,
        "title": title,
        "text": text,
        "domain_label": profile["label"],
        "quality_hint": build_quality_hint(domain),
    }
    if cur is None:
        return ARTICLE_INTELLIGENCE_PROMPT.format(**values)

    rendered = render_prompt(
        cur,
        task="article_intelligence",
        theme=domain,
        fallback_template=ARTICLE_INTELLIGENCE_PROMPT,
        values=values,
        model_provider="deepseek",
        model_name=ARTICLE_INTELLIGENCE_MODEL,
    )
    if rendered.source == "db":
        log.info("Prompt article_intelligence: %s v%s (domain=%s)", rendered.theme, rendered.version, domain)
    return rendered.text


def _as_int(value, default: int = 0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def normalize_result(result: dict) -> dict:
    return {
        "impact_fr": _clean_text(result.get("impact_fr"), 500),
        "why_interesting_fr": _clean_text(result.get("why_interesting_fr"), 500),
        "reliability": result.get("reliability") or "presse",
        "score_interest": _as_int(result.get("score_interest"), 50),
        "keywords_json": [str(k)[:60] for k in _as_list(result.get("keywords_json"))[:8]],
    }


def analyze_article(article: dict, cur=None) -> tuple[dict | None, str, str]:
    prompt = _build_prompt(article, cur=cur)
    result = analyze_with_deepseek(prompt, model=ARTICLE_INTELLIGENCE_MODEL)
    if result:
        return normalize_result(result), "deepseek", ARTICLE_INTELLIGENCE_MODEL

    result = analyze_with_gemini(prompt)
    if result:
        return normalize_result(result), "gemini", "gemini-3.1-flash-lite"

    result = analyze_with_openai(prompt)
    if result:
        return normalize_result(result), "openai", "gpt-4o-mini"

    return None, "none", "none"


def run_article_intelligence(cur, articles: list[dict], limit: int = ARTICLE_INTELLIGENCE_LIMIT) -> int:
    """Enrich up to `limit` articles and push results to D1.

    `articles` comes from the same D1 fetch used by clustering (there is no
    dedicated "needs enrichment" stage on the Worker) — the enrich route is an
    upsert, so re-enriching an already-enriched article is harmless, just
    slightly wasteful; the limit keeps LLM cost bounded per run.
    """
    if not articles:
        log.info("No articles need LLM intelligence")
        return 0

    batch = articles[:limit]
    enriched = 0
    to_push = []

    for article in batch:
        try:
            content, provider, model = analyze_article(article, cur=cur)
            if not content:
                log.warning("Article intelligence returned no JSON for %s", article.get("hash"))
                continue

            to_push.append({
                "hash": article["hash"],
                "impact_fr": content["impact_fr"],
                "reliability": content["reliability"],
                "why_interesting_fr": content["why_interesting_fr"],
                "score_interest": content["score_interest"],
                "keywords_json": content["keywords_json"],
            })
            enriched += 1
            log.info("Article intelligence [%s]: %s", provider, (article.get("title") or "")[:80])
        except Exception as exc:
            log.error("Article intelligence failed for %s: %s", article.get("hash"), exc, exc_info=True)

    if to_push:
        pushed = db.push_article_enrichment(to_push)
        log.info("Pushed %s enrichments to Worker", pushed)

    log.info("Article intelligence enriched %d/%d articles", enriched, len(batch))
    return enriched
