"""LLM analysis on top clusters — DeepSeek V4, Gemini 3.1, OpenAI, Grok 4.3.

Strategy (optimized for cost, June 2026):
  - DeepSeek V4 Flash  → default for all cluster analyses ($0.14/M input)
  - Gemini 3.1 Flash-Lite → podcast scripts, summaries ($0.25/M input)
  - GPT-4o-mini         → final UX syntheses, quiz ($0.15/M input)
  - Grok 4.3            → 1-2 deep signal analyses/day only ($1.25/M input)
"""

import json
import logging
import os
import re
from datetime import date, datetime

import httpx

from . import db
from .prompt_registry import render_prompt

log = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROK_URL = "https://api.x.ai/v1/chat/completions"


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_candidate(text: str) -> str:
    stripped = _strip_json_fence(text)
    if stripped.startswith("{") or stripped.startswith("["):
        balanced = _extract_first_balanced_json(stripped)
        return balanced or stripped

    starts = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0]
    if not starts:
        return stripped

    start = min(starts)
    balanced = _extract_first_balanced_json(stripped[start:])
    return balanced or stripped[start:]


def _extract_first_balanced_json(text: str) -> str | None:
    if not text:
        return None

    opener = text[0]
    if opener not in "{[":
        return None

    stack = [opener]
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}

    for index, char in enumerate(text[1:], 1):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or pairs[stack[-1]] != char:
                return None
            stack.pop()
            if not stack:
                return text[:index + 1]

    return None


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_llm_json(text: str, provider: str) -> dict | None:
    """Parse provider JSON with small repairs for common LLM formatting issues."""
    candidate = _extract_json_candidate(text)

    for attempt in (candidate, _remove_trailing_commas(candidate)):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
            log.warning("%s JSON ignored: root value is %s", provider, type(parsed).__name__)
            return None
        except json.JSONDecodeError:
            continue

    preview = candidate[:500].replace("\n", "\\n")
    log.error("%s JSON parse failed. Preview: %s", provider, preview)
    return None


def _safe_iso_date(value: object) -> str | None:
    """Accept only real ISO dates. Ambiguous LLM dates must become NULL."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw or raw.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return None
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return None


def _safe_importance(value: object, default: int = 5) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 10))


# ── Prompts ──────────────────────────────────────────────────────────────────

CLUSTER_ANALYSIS_PROMPT = """You are a strategic intelligence editor. Analyze this cluster of related articles.

Cluster title: {title}
Number of sources: {source_count}
Sources: {source_names}

Articles:
{articles_text}

Produce a JSON response with these fields:
- "summary": 2-3 sentence summary in French
- "why_it_matters": why this matters for the relevant audience in this story (in French)
- "tech_impact": technical/product/developer consequences in French, or null when the story is not technical
- "business_impact": company/industry/operator consequences in French, or null when not relevant
- "finance_impact": market/financial/investor consequences in French, or null when not relevant
- "risk_level": "low" | "medium" | "high"
- "key_takeaways": array of 3 key points (in French)
- "suggested_keywords": array of 3-5 keywords to track
- "pedagogical_analysis": a deep educational analysis object in French with:
  - "executive_explanation": 5-7 sentences that explain the story clearly without jargon
  - "core_mechanism": the underlying mechanism, cause, constraint, incentive, or technical/market dynamic
  - "second_order_effects": array of 3-5 non-obvious consequences
  - "stakeholder_impacts": array of objects with "stakeholder" and "impact"; choose only stakeholders that truly appear in or are directly affected by the story. Examples: states, regulators, consumers, patients, researchers, companies, developers, investors, suppliers, workers, military actors. Do not include developers or investors by default.
  - "risks": array of 3-5 concrete risks, uncertainties, or failure modes
  - "opportunities": array of 3-5 concrete opportunities or strategic options
  - "what_to_watch": array of 4-6 concrete indicators, keywords, events, filings, product launches, pricing changes, or regulatory moves to monitor next
  - "common_misreadings": array of 2-4 ways readers could misunderstand or over-interpret the story
  - "bottom_line": one strong paragraph explaining what a serious TechPulse reader should remember
- "timeline_events": array of key events, each with: "date" (strict ISO YYYY-MM-DD only when the exact day is known, otherwise null), "title" (short event description in French), "importance" (1-10). Extract 2-5 events from the articles showing how this story evolved.

Quality bar:
- Do not paraphrase the summary under a different heading.
- Do not force a tech/developer/investor framing. For geopolitics, energy, science, regulation, health, or society stories, use the actual affected actors and set irrelevant impact fields to null.
- Each impact field must add a distinct causal angle. If two fields would say the same thing, keep the most relevant field and set the other to null.
- Use concrete facts, names, numbers, constraints, and relationships from the articles.
- If the source material is thin or uncertain, say exactly what is uncertain.
- Avoid generic phrases like "this could be important for innovation" unless you explain the causal path.
- The pedagogical analysis must be useful to a strategic reader who wants to understand the system, not just the headline.
- Never invent partial dates such as "2026-06-??", "2026-06", "June 2026", or "unknown". Use null when the exact date is not available.

Respond ONLY with valid JSON, no markdown."""


WEAK_SIGNAL_PROMPT = """You are an expert analyst specializing in detecting weak signals and emerging trends in tech, AI, and finance.

Here are clusters detected today with their growth scores:
{clusters_text}

Find the 5-10 signals that most people would NOT have noticed.

For each signal, produce a JSON object with:
- "signal": what the signal is (in French)
- "why_important": why it could matter (in French)
- "strength": "weak" | "moderate" | "strong"
- "tech_impact": technical implications (in French)
- "economic_impact": economic/market implications (in French)
- "keywords_to_track": array of 3 keywords to monitor

Respond with a JSON object: {{"signals": [...]}}"""


# ── LLM Clients ──────────────────────────────────────────────────────────────

def analyze_with_deepseek(prompt: str, model: str = "deepseek-v4-flash") -> dict | None:
    """Call DeepSeek V4 API. Default: V4 Flash ($0.14/M in, $0.28/M out)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set")
        return None

    try:
        resp = httpx.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 5500,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return parse_llm_json(text, "DeepSeek")
    except Exception as e:
        log.error("DeepSeek error: %s", e)
        return None


def analyze_with_gemini(prompt: str) -> dict | None:
    """Call Gemini 3.1 Flash-Lite API ($0.25/M in, $1.50/M out)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning("GEMINI_API_KEY not set")
        return None

    try:
        resp = httpx.post(
            f"{GEMINI_URL}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 5500,
                    "responseMimeType": "application/json",
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_llm_json(text, "Gemini")
    except Exception as e:
        log.error("Gemini error: %s", e)
        return None


def analyze_with_openai(prompt: str, model: str = "gpt-4o-mini") -> dict | None:
    """Call OpenAI API ($0.15/M in, $0.60/M out)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.warning("OPENAI_API_KEY not set")
        return None

    try:
        resp = httpx.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 5500,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return parse_llm_json(text, "OpenAI")
    except Exception as e:
        log.error("OpenAI error: %s", e)
        return None


def analyze_with_grok(prompt: str, model: str = "grok-4.3") -> dict | None:
    """Call Grok 4.3 API ($1.25/M in, $2.50/M out). Use sparingly."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        log.warning("XAI_API_KEY not set")
        return None

    try:
        resp = httpx.post(
            GROK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 5500,
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return parse_llm_json(text, "Grok")
    except Exception as e:
        log.error("Grok error: %s", e)
        return None


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_cluster_prompt(cur, cluster: dict, articles: list[dict]) -> str:
    """Build the analysis prompt for a cluster."""
    articles_text = ""
    for i, a in enumerate(articles[:8], 1):
        desc = a.get("description") or ""
        full_text = (a.get("full_text") or "").strip()
        excerpt = full_text[:1200] if full_text else desc[:500]
        pub = a.get("published_at", "")
        date_str = str(pub)[:10] if pub else "unknown"
        articles_text += f"\n{i}. [{a['source_name']}] ({date_str}) {a['title']}\n   {desc[:200]}\n"
        if excerpt:
            articles_text += f"   Excerpt: {excerpt}\n"

    source_names = sorted(set(a["source_name"] for a in articles if a.get("source_name")))

    rendered = render_prompt(
        cur,
        task="cluster_analysis",
        theme="general",
        fallback_template=CLUSTER_ANALYSIS_PROMPT,
        values={
            "title": cluster["title"],
            "source_count": len(articles),
            "source_names": ", ".join(source_names),
            "articles_text": articles_text,
        },
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
    )
    if rendered.source == "db":
        log.info("Prompt cluster_analysis: %s v%s", rendered.theme, rendered.version)
    return rendered.text


def build_weak_signal_prompt(cur, clusters: list[dict]) -> str:
    """Build prompt for Grok weak signal detection."""
    clusters_text = ""
    for i, c in enumerate(clusters, 1):
        clusters_text += (
            f"\n{i}. {c['title']}"
            f"\n   Articles: {c['article_count']} | Sources: {c['source_diversity']}"
            f"\n   Growth: {c['growth_score']} | Novelty: {c['novelty_score']}\n"
        )
    rendered = render_prompt(
        cur,
        task="weak_signal_analysis",
        theme="general",
        fallback_template=WEAK_SIGNAL_PROMPT,
        values={"clusters_text": clusters_text},
        model_provider="grok",
        model_name="grok-4.3",
    )
    if rendered.source == "db":
        log.info("Prompt weak_signal_analysis: %s v%s", rendered.theme, rendered.version)
    return rendered.text


# ── Orchestration ────────────────────────────────────────────────────────────

def _pick_provider(cluster: dict, rank: int):
    """Choose the right LLM based on cluster characteristics and rank.

    Strategy:
      - Rank 1-2 (top signals):   Grok 4.3 (deep analysis, premium)
      - Tech/code clusters:       GPT-4o-mini (good at structured tech output)
      - Everything else:          DeepSeek V4 Flash (cheapest, fast, good quality)
      - Fallback chain:           DeepSeek → Gemini → OpenAI
    """
    title_lower = (cluster.get("title") or "").lower()

    is_top_signal = rank <= 2 and cluster.get("growth_score", 0) > 30
    is_tech = any(
        kw in title_lower
        for kw in ["code", "api", "sdk", "framework", "developer", "github",
                    "programming", "devops", "kubernetes", "docker"]
    )

    if is_top_signal:
        return "grok"
    elif is_tech:
        return "openai"
    else:
        return "deepseek"


def _call_provider(provider: str, prompt: str) -> tuple[dict | None, str, str]:
    """Call the chosen provider with fallback chain.

    Returns (result, provider_name, model_name).
    """
    if provider == "grok":
        result = analyze_with_grok(prompt)
        if result:
            return result, "grok", "grok-4.3"
        # Fallback to DeepSeek V4 Pro for deep analysis
        result = analyze_with_deepseek(prompt, model="deepseek-v4-pro")
        if result:
            return result, "deepseek", "deepseek-v4-pro"

    if provider == "openai":
        result = analyze_with_openai(prompt)
        if result:
            return result, "openai", "gpt-4o-mini"

    if provider == "deepseek":
        result = analyze_with_deepseek(prompt)
        if result:
            return result, "deepseek", "deepseek-v4-flash"

    # Preferred fallback: OpenAI tends to preserve structured JSON well.
    result = analyze_with_openai(prompt)
    if result:
        return result, "openai", "gpt-4o-mini"

    # Last resort: Gemini, if configured and authorized.
    result = analyze_with_gemini(prompt)
    if result:
        return result, "gemini", "gemini-3.1-flash-lite"

    return None, "", ""


def run_llm_analysis(cur, limit: int = 15) -> int:
    """Analyze top clusters with LLMs.

    Cost breakdown for 15 clusters:
      - 2 via Grok 4.3:      ~$0.008/day
      - 3 via GPT-4o-mini:   ~$0.002/day
      - 10 via DeepSeek V4:  ~$0.004/day
      Total: ~$0.014/day = ~$0.42/month
    """
    top_clusters = db.fetch_top_clusters(cur, limit=limit)
    analyzed = 0
    force = os.environ.get("TECHPULSE_FORCE_LLM_ANALYSIS", "").lower() == "true"

    for rank, cluster in enumerate(top_clusters, 1):
        # Skip if already analyzed recently
        if not force:
            cur.execute(
                """
                SELECT id FROM ai_analyses
                WHERE target_type = 'cluster' AND target_id = %s
                  AND created_at > NOW() - INTERVAL '12 hours'
                """,
                (cluster["id"],),
            )
            if cur.fetchone():
                continue

        articles = db.fetch_cluster_articles(cur, cluster["id"])
        if len(articles) < 2:
            continue

        prompt = build_cluster_prompt(cur, cluster, articles)
        provider = _pick_provider(cluster, rank)
        result, used_provider, used_model = _call_provider(provider, prompt)

        if result:
            db.insert_analysis(
                cur,
                target_type="cluster",
                target_id=cluster["id"],
                provider=used_provider,
                model=used_model,
                analysis_type="full",
                content=result,
            )

            if result.get("suggested_keywords"):
                for kw in result["suggested_keywords"]:
                    db.upsert_keyword(
                        cur, kw, category="trend", source="llm",
                        reason=f"from cluster: {cluster['title'][:50]}",
                    )

            # Insert timeline events from LLM response
            timeline_count = 0
            for event in result.get("timeline_events", []):
                if not isinstance(event, dict) or not event.get("title"):
                    continue
                db.insert_timeline_event(
                    cur,
                    cluster_id=cluster["id"],
                    title=event["title"],
                    description=None,
                    event_date=_safe_iso_date(event.get("date")),
                    importance=_safe_importance(event.get("importance")),
                )
                timeline_count += 1

            analyzed += 1
            log.info("Analyzed [%s] cluster #%d: %s (%d timeline events)",
                     used_provider, rank, cluster["title"][:50], timeline_count)

    log.info("LLM analysis: %d clusters analyzed", analyzed)
    return analyzed


def run_weak_signal_analysis(cur) -> dict | None:
    """Run Grok 4.3 deep analysis on all clusters to find hidden signals.

    Called once per day — uses the most expensive model for maximum insight.
    """
    clusters = db.fetch_top_clusters(cur, limit=30)
    if len(clusters) < 5:
        log.info("Not enough clusters for weak signal analysis")
        return None

    prompt = build_weak_signal_prompt(cur, clusters)
    log.info("Running Grok 4.3 weak signal analysis on %d clusters...", len(clusters))

    result = analyze_with_grok(prompt)
    if not result:
        log.warning("Grok failed, falling back to DeepSeek V4 Pro")
        result = analyze_with_deepseek(prompt, model="deepseek-v4-pro")

    if result:
        db.insert_analysis(
            cur,
            target_type="daily_digest",
            target_id="weak_signals",
            provider="grok" if result else "deepseek",
            model="grok-4.3",
            analysis_type="weak_signal",
            content=result,
        )
        log.info("Weak signal analysis complete: %d signals found",
                 len(result.get("signals", [])))

    return result
