"""Extract key moments from podcast transcripts using LLM.

Podcasts contain unique value that articles don't:
- Quotable statements from named guests
- Predictions with time horizons
- Disagreements and debates
- Concept explanations (feeds "Comprendre l'IA")

This module reads podcast transcripts from Neon, sends them to DeepSeek
with a dedicated prompt, and stores the extracted moments in ai_analyses
(analysis_type='podcast_moments') for the app to surface in Serendipity
and the article detail view.
"""

import json
import logging
import os

from . import db
from .llm_analyzer import analyze_with_deepseek, parse_llm_json

log = logging.getLogger(__name__)

# Max episodes to analyze per run (cost control — DeepSeek V4 Flash is cheap
# but transcripts are long ~6000 tokens each)
MAX_EPISODES_PER_RUN = 5
MIN_TRANSCRIPT_LENGTH = 500  # skip very short episodes (ads, trailers)
MAX_TRANSCRIPT_CHARS = 15000  # cap input to LLM


PODCAST_MOMENTS_PROMPT = """You are an expert editorial analyst for TechPulse, a strategic tech/AI/finance watch product.
You are given a podcast transcript (possibly diarized with speaker labels like speaker_0, speaker_1).

Your job: extract the HIGH-SIGNAL moments that a strategic reader cannot get from a headline.

Podcast title: {title}
Source: {source_name}
Language hint: {language}

Transcript (may be truncated):
{transcript}

Extract the following, as STRICT JSON (no markdown around it):

{{
  "tldr": "one sentence summary of what this episode is about",
  "key_quotes": [
    {{
      "quote": "the exact words spoken (paraphrase only if needed for readability, keep it faithful)",
      "speaker": "speaker label or inferred name/role if obvious from context, else null",
      "context": "1 sentence explaining why this quote matters (in French)",
      "topic": "short tag: e.g. 'AI', 'semiconductors', 'macro', 'space', 'regulation'"
    }}
  ],
  "predictions": [
    {{
      "prediction": "what the speaker predicted will happen (in French)",
      "horizon": "short-term" | "medium-term" | "long-term" | "unknown",
      "speaker": "who made the prediction, or null",
      "confidence": "stated" | "implied" | "speculative"
    }}
  ],
  "concepts_explained": [
    {{
      "concept": "the term or idea being explained (in original language)",
      "explanation": "the explanation as given in the podcast, condensed to 1-2 sentences (in French)",
      "domain": "AI" | "math" | "finance" | "science" | "space" | "energy" | "other"
    }}
  ],
  "disagreements": [
    {{
      "topic": "what the disagreement was about (in French)",
      "position_a": "one side's view (in French)",
      "position_b": "the other side's view (in French)",
      "speakers": ["speaker labels if identifiable"]
    }}
  ],
  "cross_domain_links": [
    {{
      "from_domain": "e.g. science",
      "to_domain": "e.g. economy",
      "link": "how this episode connects the two domains (in French, 1 sentence)"
    }}
  ],
  "entities_mentioned": ["list of companies, people, products, or technologies named in the episode"],
  "epistemic_status": "peer-reviewed" | "expert_opinion" | "communique" | "analysis" | "presse" | "rumeur"
}}

Rules:
- Extract ONLY what is actually in the transcript. Never invent quotes, predictions, or concepts.
- If a category is empty, return an empty array. Do NOT pad with filler.
- key_quotes: 2-6 items. Only quotes that are quotable, surprising, or strategically useful.
- predictions: 0-4 items. Only explicit predictions about the future, not general observations.
- concepts_explained: 0-5 items. Only when the speaker actually EXPLAINS a concept, not just mentions it.
- disagreements: 0-3 items. Only real debates, not mild nuances.
- cross_domain_links: 0-2 items. Only when the episode genuinely bridges two domains.
- Context and explanations must be in FRENCH. Quotes stay in original language.
- epistemic_status reflects the nature of the claims: expert_opinion for interviews, analysis for commentary, presse for news recaps.

Respond ONLY with valid JSON."""


def run_podcast_moments_extraction(cur) -> int:
    """Extract key moments from recent podcast transcripts.

    Returns the number of episodes analyzed.
    """
    # Fetch podcast articles with transcripts that haven't been analyzed yet
    cur.execute(
        """
        SELECT a.id, a.title, a.source_name, a.full_text, a.language, a.published_at
        FROM articles a
        WHERE a.source_type = 'podcast'
          AND a.full_text IS NOT NULL
          AND LENGTH(a.full_text) >= %s
          AND NOT EXISTS (
            SELECT 1 FROM ai_analyses aa
            WHERE aa.target_type = 'article'
              AND aa.target_id = a.id
              AND aa.analysis_type = 'podcast_moments'
          )
        ORDER BY a.published_at DESC NULLS LAST
        LIMIT %s
        """,
        (MIN_TRANSCRIPT_LENGTH, MAX_EPISODES_PER_RUN),
    )

    episodes = cur.fetchall()
    if not episodes:
        log.info("[PodcastMoments] No new podcast transcripts to analyze")
        return 0

    log.info("[PodcastMoments] Analyzing %d podcast episode(s)", len(episodes))

    analyzed = 0
    for ep in episodes:
        article_id = ep["id"]
        title = ep["title"]
        source_name = ep["source_name"]
        transcript = ep["full_text"] or ""
        language = ep["language"] or "en"

        # Truncate transcript to stay within token budget
        truncated = transcript[:MAX_TRANSCRIPT_CHARS]
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            truncated += "\n[...transcript truncated...]"

        prompt = PODCAST_MOMENTS_PROMPT.format(
            title=title,
            source_name=source_name,
            language=language,
            transcript=truncated,
        )

        # Use DeepSeek V4 Flash (cheapest, good at structured extraction)
        result = analyze_with_deepseek(prompt, model="deepseek-v4-flash")

        if not result:
            log.warning("[PodcastMoments] LLM returned nothing for '%s'", title[:60])
            continue

        # Validate structure
        if not isinstance(result, dict):
            log.warning("[PodcastMoments] Invalid response for '%s'", title[:60])
            continue

        # Enrich with metadata for storage
        result["_article_title"] = title
        result["_source_name"] = source_name
        result["_article_id"] = article_id

        # Store in ai_analyses
        db.insert_analysis(
            cur,
            target_type="article",
            target_id=article_id,
            provider="deepseek",
            model="deepseek-v4-flash",
            analysis_type="podcast_moments",
            content=result,
        )

        analyzed += 1
        quote_count = len(result.get("key_quotes", []))
        pred_count = len(result.get("predictions", []))
        concept_count = len(result.get("concepts_explained", []))
        log.info(
            "[PodcastMoments] '%s' → %d quotes, %d predictions, %d concepts",
            title[:60],
            quote_count,
            pred_count,
            concept_count,
        )

    log.info("[PodcastMoments] Analyzed %d / %d episodes", analyzed, len(episodes))
    return analyzed
