"""Sérendipité scientifique — pépites quotidiennes ancrées sur de vrais papiers arXiv.

Workflow (sans hallucination de sources) :
  1. Choisir au hasard quelques domaines scientifiques "inspirationnels".
  2. Récupérer des papiers RÉCENTS via l'API publique arXiv (gratuite, sans clé).
  3. Demander au LLM de VULGARISER le titre+résumé réels (interdit d'inventer).
  4. Stocker une carte par papier (avec arxiv_id + lien = citation).

Le LLM ne choisit pas le sujet et n'invente pas de chercheur : il reformule une
source réelle. C'est ce qui distingue cette implémentation de la version "seed
aléatoire" qui ferait halluciner des noms/papiers.
"""

import logging
import os
import random
import time
import xml.etree.ElementTree as ET

import httpx

from . import db
from .llm_analyzer import analyze_with_gemini

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

# Catégories arXiv -> domaine vulgarisé (FR). Choisies pour l'effet "inspirationnel"
# (science de demain), complémentaire de la veille tech/finance "rationnelle".
DOMAINS: dict[str, str] = {
    "astro-ph.HE": "astrophysique",
    "astro-ph.CO": "cosmologie",
    "gr-qc": "gravitation et relativité",
    "quant-ph": "physique quantique",
    "cond-mat.quant-gas": "matière quantique",
    "physics.bio-ph": "biophysique",
    "q-bio.NC": "neurosciences",
    "q-bio.GN": "génomique",
    "nlin.AO": "systèmes complexes",
    "math.DS": "systèmes dynamiques",
}

_ATOM = "{http://www.w3.org/2005/Atom}"


def _arxiv_id_from_url(url: str) -> str:
    """http://arxiv.org/abs/2401.12345v2 -> 2401.12345 (sans version, pour la dédup)."""
    tail = url.rstrip("/").split("/abs/")[-1]
    return tail.split("v")[0] if "v" in tail else tail


def fetch_arxiv_recent(category: str, max_results: int = 15) -> list[dict]:
    """Papiers récents d'une catégorie arXiv, triés par date de soumission."""
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }
    try:
        resp = httpx.get(ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.warning("arXiv fetch failed for %s: %s", category, e)
        return []

    papers: list[dict] = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        log.warning("arXiv XML parse failed for %s: %s", category, e)
        return []

    for entry in root.findall(f"{_ATOM}entry"):
        id_url = (entry.findtext(f"{_ATOM}id") or "").strip()
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        published = (entry.findtext(f"{_ATOM}published") or "").strip() or None
        authors = [
            (a.findtext(f"{_ATOM}name") or "").strip()
            for a in entry.findall(f"{_ATOM}author")
        ]
        if not id_url or not title or not summary:
            continue
        papers.append({
            "arxiv_id": _arxiv_id_from_url(id_url),
            "source_url": id_url,
            "title": title,
            "summary": summary,
            "authors": [a for a in authors if a][:6],
            "published_at": published,
        })
    return papers


def _build_prompt(paper: dict, domain: str) -> str:
    authors = ", ".join(paper["authors"]) or "auteurs non listés"
    return f"""Tu es un vulgarisateur scientifique pour une app mobile.
Voici un VRAI papier de recherche (domaine : {domain}).

Titre : {paper['title']}
Auteurs : {authors}
Résumé : {paper['summary'][:1800]}

Transforme-le en une "pépite" captivante, SANS RIEN INVENTER : n'ajoute aucun fait
absent du résumé, ne change pas les auteurs, reste fidèle. Vulgarise sans déformer.

Réponds uniquement par ce JSON :
{{
  "title_choc": "titre français accrocheur, max 70 caractères",
  "enigme": "pourquoi c'est fascinant, 2 phrases simples",
  "personnage": "qui est derrière (auteurs/équipe), 1 phrase",
  "concept": "explication vulgarisée du mécanisme, sans jargon, 3-4 phrases",
  "so_what": "en quoi ça change notre compréhension ou le futur, 2 phrases"
}}"""


def _vulgarize(paper: dict, domain: str, category: str) -> dict | None:
    result = analyze_with_gemini(_build_prompt(paper, domain))
    if not result or not result.get("title_choc"):
        return None
    return {
        "arxiv_id": paper["arxiv_id"],
        "source_url": paper["source_url"],
        "domain": domain,
        "arxiv_category": category,
        "title_choc": result["title_choc"][:200],
        "enigme": result.get("enigme"),
        "personnage": result.get("personnage"),
        "concept": result.get("concept"),
        "so_what": result.get("so_what"),
        "paper_title": paper["title"][:500],
        "authors": paper["authors"],
        "published_at": paper["published_at"],
        "model_provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
    }


def run_serendipity(cur, count: int | None = None) -> int:
    """Génère jusqu'à `count` cartes du jour. Best-effort, ne lève pas."""
    target = count or int(os.getenv("SERENDIPITY_COUNT", "3"))
    existing = db.fetch_recent_serendipity_arxiv_ids(cur)

    categories = list(DOMAINS.keys())
    random.shuffle(categories)

    inserted = 0
    for category in categories:
        if inserted >= target:
            break
        domain = DOMAINS[category]
        papers = [p for p in fetch_arxiv_recent(category) if p["arxiv_id"] not in existing]
        time.sleep(3)  # politesse API arXiv (~1 req / 3 s)
        if not papers:
            continue

        paper = random.choice(papers[:8])  # un papier récent au hasard
        existing.add(paper["arxiv_id"])

        card = _vulgarize(paper, domain, category)
        if not card:
            continue
        try:
            if db.insert_serendipity_card(cur, card):
                inserted += 1
                log.info("Serendipity card: [%s] %s", domain, card["title_choc"])
        except Exception as e:
            log.warning("Serendipity insert failed for %s: %s", paper["arxiv_id"], e)

    log.info("Serendipity: %d cards generated", inserted)
    return inserted
