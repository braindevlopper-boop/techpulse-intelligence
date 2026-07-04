"""LLM analysis on top clusters.

Cluster analysis is routed through OpenRouter (one key, many models) with a
per-tier model so the important clusters get a premium model and the long tail
runs cheap/free. If OPENROUTER_API_KEY is unset or a call fails, we fall back to
the legacy direct providers below, so nothing breaks during the migration.

Tier models are configured via env (see OPENROUTER_MODEL_* below). The other
pipeline modules (podcast, article_intelligence, cluster_merger, …) still use the
direct provider functions for now and will be migrated to OpenRouter next.
"""

import json
import logging
import os
import re
from datetime import date, datetime

import httpx
from . import db
from .prompt_registry import render_prompt
from .prompt_profiles import (
    detect_domain,
    build_impact_fields_section,
    build_stakeholders_hint,
    build_quality_hint,
    get_profile,
)

log = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROK_URL = "https://api.x.ai/v1/chat/completions"

# ── OpenRouter (unified LLM gateway) ──────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cluster-analysis model tiers (OpenRouter slugs, verified June 2026). Override
# via env. A wrong/unavailable slug degrades gracefully: the OpenRouter call
# fails and we fall back to the legacy direct providers.
#   premium: deepseek/deepseek-v4-pro     ($0.435/$0.87)  — near-frontier quality, ~€1/mo @ top 5/day
#   tail:    deepseek/deepseek-v4-flash   ($0.098/$0.196) — reliable JSON, near-free
# Alts for premium (higher French/prose finesse, higher cost): z-ai/glm-5.2 (~€3.4/mo),
# google/gemini-3.5-flash (~€6/mo), anthropic/claude-sonnet-4.6 (~€10/mo).
# `or` (not getenv default) so an empty env value from an unset GitHub repo
# variable still falls back to the default instead of becoming "".
OPENROUTER_MODEL_PREMIUM = os.getenv("OPENROUTER_MODEL_PREMIUM") or "deepseek/deepseek-v4-pro"
OPENROUTER_MODEL_TAIL = os.getenv("OPENROUTER_MODEL_TAIL") or "deepseek/deepseek-v4-flash"
# Auto-premium on top clusters is OFF by default (0): the feed cluster analyses
# run on the cheap tail model, and premium is reserved for the on-demand
# per-article deep-dive. Set >0 to re-enable auto-premium on the top N clusters.
OPENROUTER_PREMIUM_TOP_N = int(os.getenv("OPENROUTER_PREMIUM_TOP_N") or "0")


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
Domaine détecté: {domain_label}

Articles:
{articles_text}

Produce a JSON response with these fields:
- "summary": 2-3 sentence summary in French
- "why_it_matters": why this matters for the relevant audience in this story (in French)
{impact_fields}
- "risk_level": "low" | "medium" | "high"
- "key_takeaways": array of 3 key points (in French)
- "suggested_keywords": array of 3-5 keywords to track
- "pedagogical_analysis": a deep educational analysis object in French with:
  - "executive_explanation": 5-7 sentences that explain the story clearly without jargon
  - "core_mechanism": the underlying mechanism, cause, constraint, incentive, or technical/market dynamic
  - "second_order_effects": array of 3-5 non-obvious consequences
  - "stakeholder_impacts": array of objects with "stakeholder" and "impact"; {stakeholders_hint}. Do not include developers or investors by default.
  - "risks": array of 3-5 concrete risks, uncertainties, or failure modes
  - "opportunities": array of 3-5 concrete opportunities or strategic options
  - "what_to_watch": array of 4-6 concrete indicators, keywords, events, filings, product launches, pricing changes, or regulatory moves to monitor next
  - "common_misreadings": array of 2-4 ways readers could misunderstand or over-interpret the story
  - "bottom_line": one strong paragraph explaining what a serious TechPulse reader should remember
- "timeline_events": array of key events, each with: "date" (strict ISO YYYY-MM-DD only when the exact day is known, otherwise null), "title" (short event description in French, max ~12 words), "importance" (1-10). Extract only the 2-3 MOST significant events (not every event) showing how this story evolved.
- "epistemic": the overall epistemic status of this cluster, choose one:
  • "peer-reviewed": published research with peer review
  • "preprint": arXiv/bioRxiv working paper, not yet published
  • "communique": official statement from a company, institution, or government
  • "analyse": analysis, argued opinion, editorial, expert podcast
  • "presse": standard press article, news recap
  • "rumeur": unconfirmed leak, rumor, speculation
- "predictions": array of 0-3 predictions made in or implied by the articles, each with:
  - "prediction": what is predicted to happen (in French)
  - "horizon": "short-term" | "medium-term" | "long-term" | "unknown"
  - "confidence": "stated" | "implied" | "speculative"
  Only include real predictions about the future, not general observations. If no predictions, return empty array.
- "counter_analysis": a deliberate contradictory reading to fight hype, in French, with:
  - "counter_thesis": one-sentence opposing thesis
  - "arguments_against": array of 2-4 arguments that contradict or seriously temper the main thesis
  - "what_would_change_the_view": array of 2-3 concrete signals that would flip the analysis
  If the story is trivial or purely factual with no debatable thesis, return null.

Quality bar:
- Do not paraphrase the summary under a different heading.
- {quality_hint}
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

# Retry léger pour les erreurs transitoires (429 rate limit, 503 unavailable)
def _retryable_post(provider: str, **kwargs) -> httpx.Response:
    import time
    timeout = kwargs.pop("timeout", 60)
    for attempt in range(3):
        try:
            resp = httpx.post(**kwargs, timeout=timeout)
            if resp.status_code in (429, 503, 502):
                wait = (attempt + 1) * 3
                log.warning("%s returned %d, retrying in %ds (attempt %d)", provider, resp.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt < 2:
                wait = (attempt + 1) * 2
                log.warning("%s error: %s, retrying in %ds", provider, exc, wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"{provider}: exhausted retries after 3 attempts")

def analyze_with_deepseek(prompt: str, model: str = "deepseek-v4-flash") -> dict | None:
    """Call DeepSeek V4 API. Default: V4 Flash ($0.14/M in, $0.28/M out)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set")
        return None

    try:
        resp = _retryable_post("DeepSeek",
            url=DEEPSEEK_URL,
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
        resp = _retryable_post("Gemini",
            url=f"{GEMINI_URL}?key={api_key}",
            headers={},
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
        resp = _retryable_post("OpenAI",
            url=OPENAI_URL,
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
        resp = _retryable_post("Grok",
            url=GROK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 5500,
            },
            timeout=90,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return parse_llm_json(text, "Grok")
    except Exception as e:
        log.error("Grok error: %s", e)
        return None


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_cluster_prompt(cur, cluster: dict, articles: list[dict]) -> str:
    """Build the analysis prompt for a cluster, adapted to the cluster's domain."""
    articles_text = ""
    for i, a in enumerate(articles[:8], 1):
        content_text = (a.get("content") or "").strip()
        excerpt = content_text[:1200]
        pub = a.get("published_at", "")
        date_str = str(pub)[:10] if pub else "unknown"
        articles_text += f"\n{i}. [{a.get('source_name', '')}] ({date_str}) {a.get('title', '')}\n"
        if excerpt:
            articles_text += f"   Excerpt: {excerpt}\n"

    source_names = sorted(set(a["source_name"] for a in articles if a.get("source_name")))

    # Détecter le domaine du cluster
    cluster_title = cluster.get("title") or ""
    cluster_theme = cluster.get("theme") or ""
    domain = detect_domain(
        title=cluster_title,
        description="",
        theme=cluster_theme,
    )
    profile = get_profile(domain)

    # Construire les champs d'impact adaptatifs au format JSON
    impact_lines = []
    for field_name, field_desc in profile["impact_fields"]:
        impact_lines.append(f'- "{field_name}": {field_desc}, or null when not relevant')
    impact_fields_str = "\n".join(impact_lines)

    rendered = render_prompt(
        cur,
        task="cluster_analysis",
        theme=domain,
        fallback_template=CLUSTER_ANALYSIS_PROMPT,
        values={
            "title": cluster["title"],
            "source_count": len(articles),
            "source_names": ", ".join(source_names),
            "articles_text": articles_text,
            "domain_label": profile["label"],
            "impact_fields": impact_fields_str,
            "stakeholders_hint": build_stakeholders_hint(domain),
            "quality_hint": build_quality_hint(domain),
        },
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
    )
    if rendered.source == "db":
        log.info("Prompt cluster_analysis: %s v%s (domain=%s)", rendered.theme, rendered.version, domain)
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


def analyze_with_openrouter(prompt: str, model: str) -> dict | None:
    """Call any model through OpenRouter (OpenAI-compatible gateway)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        resp = _retryable_post(f"OpenRouter:{model}",
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://techpulse.app",
                "X-Title": "TechPulse",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 5500,
                "response_format": {"type": "json_object"},
            },
            timeout=90,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return parse_llm_json(text, f"OpenRouter:{model}")
    except Exception as e:
        log.error("OpenRouter error (%s): %s", model, e)
        return None


def _pick_cluster_model(cluster: dict, rank: int) -> str:
    """Cheap tail model for feed cluster analyses by default. Premium is reserved
    for the on-demand per-article deep-dive; set OPENROUTER_PREMIUM_TOP_N>0 to
    re-enable auto-premium on the top / high-signal clusters."""
    if OPENROUTER_PREMIUM_TOP_N <= 0:
        return OPENROUTER_MODEL_TAIL
    growth = cluster.get("growth_score") or 0
    importance = cluster.get("importance_score") or 0
    if rank <= OPENROUTER_PREMIUM_TOP_N or growth > 30 or importance > 60:
        return OPENROUTER_MODEL_PREMIUM
    return OPENROUTER_MODEL_TAIL


def _analyze_cluster(cluster: dict, rank: int, prompt: str) -> tuple[dict | None, str, str]:
    """Analyze a cluster via OpenRouter (tiered model), with a safe fallback.

    If OpenRouter is configured and succeeds → use it. Otherwise fall back to the
    legacy direct-provider routing so the pipeline keeps working unchanged.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        model = _pick_cluster_model(cluster, rank)
        result = analyze_with_openrouter(prompt, model)
        if result:
            return result, "openrouter", model
        log.warning(
            "OpenRouter failed (model=%s, rank=%d) — falling back to direct providers",
            model, rank,
        )

    provider = _pick_provider(cluster, rank)
    return _call_provider(provider, prompt)


def _map_analysis_for_push(cluster_id: str, result: dict, provider: str, model: str) -> dict:
    """Map the rich LLM JSON onto the 7 fields accepted by
    POST /pipeline/cluster-analyses. Everything else (pedagogical_analysis,
    timeline_events, predictions, counter_analysis, ...) has no D1 column yet
    and is dropped here; only the summary/impact/reliability/risk fields ship."""
    why_interesting = result.get("why_it_matters") or ""
    reliability = result.get("epistemic") or "presse"
    keywords = result.get("suggested_keywords") or []
    return {
        "cluster_id": cluster_id,
        "summary_fr": (result.get("summary") or "")[:2000],
        "impact_fr": why_interesting[:2000],
        "why_interesting_fr": why_interesting[:2000],
        "reliability": reliability,
        "risk_level": result.get("risk_level") or "medium",
        "keywords_json": keywords[:10],
        "model_used": f"{provider}:{model}",
    }


def run_llm_analysis(clusters: list[dict], limit: int = 15) -> int:
    """Analyze the top clusters (by article count) with LLMs and push results
    to D1 via POST /pipeline/cluster-analyses.

    `clusters` is the post-merge cluster payload from clusterer.py /
    cluster_merger.py (each with article_hashes + hydrated article dicts
    attached by the caller as cluster["_articles"]).

    Cost breakdown for 15 clusters:
      - 2 via Grok 4.3:      ~$0.008/day
      - 3 via GPT-4o-mini:   ~$0.002/day
      - 10 via DeepSeek V4:  ~$0.004/day
      Total: ~$0.014/day = ~$0.42/month
    """
    ranked = sorted(clusters, key=lambda c: len(c.get("article_hashes") or []), reverse=True)[:limit]
    analyzed = 0
    to_push = []

    for rank, cluster in enumerate(ranked, 1):
        articles = cluster.get("_articles") or []
        if len(articles) < 2:
            continue

        prompt = build_cluster_prompt(None, cluster, articles)
        result, used_provider, used_model = _analyze_cluster(cluster, rank, prompt)

        if result:
            to_push.append(_map_analysis_for_push(cluster["id"], result, used_provider, used_model))
            analyzed += 1
            log.info("Analyzed [%s] cluster #%d: %s", used_provider, rank, cluster["title"][:50])

    if to_push:
        pushed = db.push_cluster_analyses(to_push)
        log.info("Pushed %s cluster analyses to Worker", pushed)

    log.info("LLM analysis: %d clusters analyzed", analyzed)
    return analyzed


def run_weak_signal_analysis(clusters: list[dict]) -> dict | None:
    """Weak-signal digest generation is disabled: there is no D1 route/table
    for daily_digest analyses (ai_analyses target_type='daily_digest' had no
    D1 equivalent per the migration contract). Kept as a log-only skip so
    callers don't need to change."""
    log.info("Weak signal analysis skipped: no D1 route for daily_digest analyses")
    return None
