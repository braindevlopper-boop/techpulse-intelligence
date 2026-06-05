"""Named Entity Recognition using bert-base-NER."""

import logging
from transformers import pipeline as hf_pipeline

from . import db

log = logging.getLogger(__name__)

_ner = None

NER_TYPE_MAP = {
    "ORG": "company",
    "PER": "person",
    "LOC": "country",
    "MISC": "technology",
}


def _get_ner():
    global _ner
    if _ner is None:
        log.info("Loading NER model...")
        _ner = hf_pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1,
        )
        log.info("NER model loaded")
    return _ner


def extract_entities(text: str) -> list[dict]:
    """Extract entities from text, returning unique ones with confidence."""
    ner = _get_ner()
    raw = ner(text[:1000])

    seen = {}
    for ent in raw:
        name = ent["word"].strip()
        if len(name) < 2 or name.startswith("##"):
            continue

        key = name.lower()
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = {
                "name": name,
                "type": NER_TYPE_MAP.get(ent["entity_group"], "concept"),
                "score": round(ent["score"], 3),
            }

    return list(seen.values())


def run_ner(cur, articles: list[dict]) -> int:
    """Run NER on articles and store entities in DB."""
    total_entities = 0
    total_articles = len(articles)

    for i, article in enumerate(articles, 1):
        text = f"{article['title']} {article.get('description', '')} {(article.get('full_text') or '')[:500]}"
        entities = extract_entities(text)

        for ent in entities:
            if ent["score"] < 0.7:
                continue

            entity_id = db.upsert_entity(cur, ent["name"], ent["type"])
            db.insert_article_entity(
                cur, article["id"], entity_id,
                role="mentioned", confidence=float(ent["score"]), source="ner",
            )
            total_entities += 1

        if i % 20 == 0 or i == total_articles:
            log.info("NER: %d/%d (%.0f%%)", i, total_articles, i / total_articles * 100)

    log.info("Extracted %d entity links from %d articles", total_entities, total_articles)
    return total_entities
