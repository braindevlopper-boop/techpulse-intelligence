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

import httpx

from . import db

log = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROK_URL = "https://api.x.ai/v1/chat/completions"


# ── Prompts ──────────────────────────────────────────────────────────────────

CLUSTER_ANALYSIS_PROMPT = """You are a tech and finance analyst. Analyze this cluster of related articles.

Cluster title: {title}
Number of sources: {source_count}
Source types: {source_types}

Articles:
{articles_text}

Produce a JSON response with these fields:
- "summary": 2-3 sentence summary in French
- "why_it_matters": why this matters for a developer/investor (in French)
- "tech_impact": technical consequences (in French)
- "business_impact": business consequences (in French)
- "finance_impact": market/financial consequences (in French)
- "risk_level": "low" | "medium" | "high"
- "key_takeaways": array of 3 key points (in French)
- "suggested_keywords": array of 3-5 keywords to track

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
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)
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
                    "maxOutputTokens": 2000,
                    "responseMimeType": "application/json",
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
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
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)
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
                "max_tokens": 3000,
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        # Grok doesn't always respect JSON-only — try to extract JSON
        if text.strip().startswith("{"):
            return json.loads(text)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return None
    except Exception as e:
        log.error("Grok error: %s", e)
        return None


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_cluster_prompt(cluster: dict, articles: list[dict]) -> str:
    """Build the analysis prompt for a cluster."""
    articles_text = ""
    for i, a in enumerate(articles[:8], 1):
        desc = a.get("description") or ""
        articles_text += f"\n{i}. [{a['source_name']}] {a['title']}\n   {desc[:200]}\n"

    source_types = list(set(a["source_type"] for a in articles))

    return CLUSTER_ANALYSIS_PROMPT.format(
        title=cluster["title"],
        source_count=len(articles),
        source_types=", ".join(source_types),
        articles_text=articles_text,
    )


def build_weak_signal_prompt(clusters: list[dict]) -> str:
    """Build prompt for Grok weak signal detection."""
    clusters_text = ""
    for i, c in enumerate(clusters, 1):
        clusters_text += (
            f"\n{i}. {c['title']}"
            f"\n   Articles: {c['article_count']} | Sources: {c['source_diversity']}"
            f"\n   Growth: {c['growth_score']} | Novelty: {c['novelty_score']}\n"
        )
    return WEAK_SIGNAL_PROMPT.format(clusters_text=clusters_text)


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

    # Ultimate fallback: Gemini
    result = analyze_with_gemini(prompt)
    if result:
        return result, "gemini", "gemini-3.1-flash-lite"

    # Last resort: OpenAI
    result = analyze_with_openai(prompt)
    if result:
        return result, "openai", "gpt-4o-mini"

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

    for rank, cluster in enumerate(top_clusters, 1):
        # Skip if already analyzed recently
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

        prompt = build_cluster_prompt(cluster, articles)
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

            analyzed += 1
            log.info("Analyzed [%s] cluster #%d: %s", used_provider, rank, cluster["title"][:50])

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

    prompt = build_weak_signal_prompt(clusters)
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
