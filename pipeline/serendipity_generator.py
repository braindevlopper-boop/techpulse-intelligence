"""Sérendipité — pépites inattendues à partir de l'intelligence TechPulse.

Le but n'est pas de créer une rubrique arXiv. Le générateur part d'articles et
clusters déjà enrichis par TechPulse, puis demande au LLM de choisir quelques
signaux scientifiques ou technologiques qui ouvrent une porte inattendue.
"""

import json
import logging
import os
import random
import re
from datetime import date, datetime
from typing import Any

from . import db
from .llm_analyzer import analyze_with_deepseek, analyze_with_gemini

log = logging.getLogger(__name__)

DOMAIN_LABELS = {
    "ai": "IA & recherche",
    "artificial_intelligence": "IA & recherche",
    "space": "espace",
    "energy": "énergie",
    "biotech": "biotech",
    "medicine": "médecine",
    "science": "science",
    "semiconductors": "semi-conducteurs",
    "climate": "climat",
    "robotics": "robotique",
}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, (datetime, date)):
        text = value.isoformat()
    else:
        text = str(value)
    return " ".join(text.split())[:limit]


def _arxiv_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"arxiv\.org/abs/([^?#/]+)", url, re.I)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1), flags=re.I)


def _domain_for(candidate: dict) -> str:
    for key in (
        candidate.get("primary_domain"),
        candidate.get("category"),
        candidate.get("source_type"),
    ):
        if not key:
            continue
        normalized = str(key).strip().lower().replace("-", "_")
        if normalized in DOMAIN_LABELS:
            return DOMAIN_LABELS[normalized]
        if normalized.startswith("arxiv"):
            return "science"
    topic = _as_text(candidate.get("topic"), 80).lower()
    if any(word in topic for word in ("bio", "drug", "protein", "medicine", "crispr")):
        return "bio & santé"
    if any(word in topic for word in ("space", "astro", "satellite", "cosmo")):
        return "espace"
    if any(word in topic for word in ("quantum", "physics", "material")):
        return "physique"
    return "inspiration"


def _candidate_payload(candidate: dict, index: int) -> dict:
    entities = _as_list(candidate.get("entities"))[:6]
    keywords = _as_list(candidate.get("keywords"))[:8]
    subtopics = _as_list(candidate.get("subtopics"))[:5]
    return {
        "source_id": f"c{index}",
        "title": _as_text(candidate.get("title"), 220),
        "summary": _as_text(candidate.get("description"), 500),
        "source": _as_text(candidate.get("source_name"), 80),
        "url": candidate.get("source_url"),
        "domain": _domain_for(candidate),
        "topic": _as_text(candidate.get("topic"), 160),
        "subtopics": subtopics,
        "entities": entities,
        "keywords": keywords,
        "cluster": _as_text(candidate.get("cluster_title"), 180),
        "cluster_hint": _as_text(candidate.get("cluster_hint"), 220),
        "published_at": _as_text(candidate.get("published_at") or candidate.get("fetched_at"), 40),
        "scores": {
            "relevance": candidate.get("relevance_score") or 0,
            "quality": candidate.get("quality_score") or 0,
            "importance": candidate.get("importance_score") or 0,
            "novelty": candidate.get("novelty_score") or 0,
            "growth": candidate.get("growth_score") or 0,
        },
    }


def _build_prompt(candidates: list[dict], target: int) -> str:
    payload = json.dumps(candidates, ensure_ascii=False)
    return f"""Tu es l'éditeur du flux "Sérendipité" de TechPulse.

Objectif : choisir {target} pépites inattendues parmi ces signaux déjà enrichis par
TechPulse. Ce ne sont PAS des news à résumer. Ce sont des portes d'entrée vers une
idée que l'utilisateur n'aurait probablement pas cherchée : science profonde,
recherche appliquée, médecine, espace, énergie, semi-conducteurs, IA ou robotique.

Règles :
- Utilise uniquement les faits présents dans les candidats.
- Choisis des sujets multidisciplinaires et surprenants.
- Ne sélectionne pas plus d'une carte issue d'arXiv si d'autres sources sont disponibles.
- Favorise la diversité des sources : preprints bio/médecine/chimie, revues scientifiques,
  institutions de recherche, analyses science crédibles, Grok science, et clusters TechPulse.
- Évite les doublons et les sujets purement business/produit.
- Évite les actualités trop légères : il faut une idée, une méthode, un résultat ou une
  implication scientifique concrète.
- Garde un ton captivant, clair, sans jargon gratuit.
- Chaque carte doit référencer un "source_id" fourni.

Candidats JSON :
{payload}

Réponds uniquement par ce JSON strict :
{{
  "cards": [
    {{
      "source_id": "c0",
      "title_choc": "titre français accrocheur, max 75 caractères",
      "enigme": "pourquoi c'est étonnant, 2 phrases simples",
      "personnage": "acteur, chercheur, équipe ou organisation clé, 1 phrase",
      "concept": "explication vulgarisée en 3-4 phrases",
      "so_what": "le 'et alors ?' : ce que cela pourrait changer, 2 phrases",
      "domain": "domaine court en français"
    }}
  ]
}}"""


def _generate_cards(candidates: list[dict], target: int) -> tuple[list[dict], str, str]:
    prompt = _build_prompt(candidates, target)
    result = analyze_with_gemini(prompt)
    provider, model = "gemini", "gemini-3.1-flash-lite"
    if not result or not isinstance(result.get("cards"), list):
        result = analyze_with_deepseek(prompt)
        provider, model = "deepseek", "deepseek-v4-flash"
    if not result or not isinstance(result.get("cards"), list):
        return [], provider, model
    return result["cards"], provider, model


def _to_card(raw: dict, source_by_id: dict[str, dict], provider: str, model: str) -> dict | None:
    source_id = str(raw.get("source_id") or "")
    source = source_by_id.get(source_id)
    if not source:
        return None

    title = _as_text(raw.get("title_choc"), 200)
    if not title:
        return None

    source_url = source.get("url")
    return {
        "arxiv_id": _arxiv_id_from_url(source_url),
        "source_url": source_url,
        "domain": _as_text(raw.get("domain") or source.get("domain") or "inspiration", 80),
        "arxiv_category": None,
        "title_choc": title,
        "enigme": _as_text(raw.get("enigme"), 700) or None,
        "personnage": _as_text(raw.get("personnage"), 400) or None,
        "concept": _as_text(raw.get("concept"), 1000) or None,
        "so_what": _as_text(raw.get("so_what"), 700) or None,
        "paper_title": source.get("title"),
        "authors": [source.get("source")] if source.get("source") else [],
        "published_at": source.get("published_at") or None,
        "model_provider": provider,
        "model_name": model,
    }


def run_serendipity(cur, count: int | None = None) -> int:
    """Génère quelques cartes d'inspiration. Best-effort, ne lève pas."""
    target = count or int(os.getenv("SERENDIPITY_COUNT", "5"))
    existing_urls = db.fetch_recent_serendipity_source_urls(cur)
    existing_arxiv_ids = db.fetch_recent_serendipity_arxiv_ids(cur)

    raw_candidates = [
        item for item in db.fetch_serendipity_candidates(cur, limit=120)
        if item.get("source_url") not in existing_urls
        and (_arxiv_id_from_url(item.get("source_url")) not in existing_arxiv_ids)
    ]
    if not raw_candidates:
        log.info("Serendipity: no TechPulse candidates available")
        return 0

    random.shuffle(raw_candidates)
    payload = [_candidate_payload(candidate, index) for index, candidate in enumerate(raw_candidates[:45])]
    source_by_id = {candidate["source_id"]: candidate for candidate in payload}

    generated, provider, model = _generate_cards(payload, target)
    inserted = 0
    for raw in generated[:target]:
        card = _to_card(raw, source_by_id, provider, model)
        if not card:
            continue
        try:
            if card.get("source_url") in existing_urls:
                continue
            if db.insert_serendipity_card(cur, card):
                inserted += 1
                existing_urls.add(card.get("source_url"))
                if card.get("arxiv_id"):
                    existing_arxiv_ids.add(card["arxiv_id"])
                log.info("Serendipity card: [%s] %s", card["domain"], card["title_choc"])
        except Exception as exc:
            log.warning("Serendipity insert failed for %s: %s", card.get("source_url"), exc)

    log.info("Serendipity: %d cards generated from TechPulse intelligence", inserted)
    return inserted
