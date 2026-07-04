"""HTTP client for the techpulse-worker /pipeline/* bridge to Cloudflare D1.

This pipeline no longer talks to Postgres/Neon directly. Every read/write goes
through the Worker's pipeline routes over HTTPS. `get_cursor()` is kept as a
no-op context manager so call sites that did `with db.get_cursor() as cur:`
still work — `cur` is unused (routes carry no transaction/session state).
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager

import httpx

log = logging.getLogger(__name__)

WORKER_URL = (os.environ.get("TECHPULSE_WORKER_URL") or "https://techpulse-worker.bricebrain.workers.dev").rstrip("/")
WORKER_SECRET = os.environ.get("TECHPULSE_WORKER_SECRET")


def gen_id() -> str:
    return uuid.uuid4().hex[:16]


def _headers() -> dict:
    if not WORKER_SECRET:
        raise RuntimeError("TECHPULSE_WORKER_SECRET is required")
    return {
        "Authorization": f"Bearer {WORKER_SECRET}",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, *, params: dict | None = None,
             json_body: dict | None = None, timeout: float = 30.0) -> dict:
    url = f"{WORKER_URL}{path}"
    last_exc = None
    for attempt in range(3):
        try:
            resp = httpx.request(
                method, url, params=params, json=json_body,
                headers=_headers(), timeout=timeout,
            )
            if resp.status_code in (429, 502, 503):
                wait = (attempt + 1) * 2
                log.warning("Worker %s %s returned %d, retrying in %ds", method, path, resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            if not resp.content:
                return {}
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < 2:
                wait = (attempt + 1) * 2
                log.warning("Worker %s %s error: %s, retrying in %ds", method, path, exc, wait)
                time.sleep(wait)
                continue
    raise RuntimeError(f"Worker request failed: {method} {path}: {last_exc}")


def _stringify_json_fields(items: list[dict], fields: tuple[str, ...]) -> list[dict]:
    """The Worker binds these fields directly into a D1 TEXT column — they must
    be JSON-encoded strings, not raw Python lists/dicts, or env.DB bind() fails."""
    out = []
    for item in items:
        item = dict(item)
        for field in fields:
            value = item.get(field)
            if value is not None and not isinstance(value, str):
                item[field] = json.dumps(value)
        out.append(item)
    return out


@contextmanager
def get_cursor():
    """Kept for call-site compatibility (`with db.get_cursor() as cur:`).

    There is no DB connection/transaction anymore — every db.* call makes its
    own HTTP request to the Worker. `cur` is a harmless placeholder object.
    """
    yield None


# ── Pipeline articles (cluster stage) ──

def fetch_processed_articles(cur=None, hours: int = 72, limit: int = 500) -> list[dict]:
    """Articles available for clustering, from the Worker bridge."""
    data = _request("GET", "/pipeline/articles", params={
        "stage": "cluster", "hours": hours, "limit": limit,
    })
    return data.get("articles", [])


# ── Clusters ──

def push_clusters(clusters: list[dict]) -> dict:
    """Upsert clusters + article links. Max 200 per call (chunked here)."""
    clusters = _stringify_json_fields(clusters, ("keywords_json",))
    total = 0
    for i in range(0, len(clusters), 200):
        chunk = clusters[i:i + 200]
        result = _request("POST", "/pipeline/clusters", json_body={"clusters": chunk})
        total += result.get("count", len(chunk))
    return {"count": total}


# ── Cluster analyses ──

def push_cluster_analyses(analyses: list[dict]) -> dict:
    """Upsert cluster_analyses. Max 100 per call (chunked here)."""
    analyses = _stringify_json_fields(analyses, ("keywords_json",))
    total = 0
    for i in range(0, len(analyses), 100):
        chunk = analyses[i:i + 100]
        result = _request("POST", "/pipeline/cluster-analyses", json_body={"analyses": chunk})
        total += result.get("count", len(chunk))
    return {"count": total}


# ── Article enrichment (lightweight, 5 fields) ──

def push_article_enrichment(articles: list[dict]) -> dict:
    """Upsert lightweight per-article enrichment. Max 300 per call (chunked)."""
    articles = _stringify_json_fields(articles, ("keywords_json",))
    total = 0
    for i in range(0, len(articles), 300):
        chunk = articles[i:i + 300]
        result = _request("POST", "/pipeline/articles/enrich", json_body={"articles": chunk})
        total += result.get("count", len(chunk))
    return {"count": total}


# ── Prompt registry ──

def fetch_active_prompt_template(cur, task: str, theme: str = "general") -> dict | None:
    data = _request("GET", "/pipeline/prompts", params={"task": task, "theme": theme})
    prompt = data.get("prompt")
    if not prompt:
        return None
    return {
        "id": prompt.get("id"),
        "task": prompt.get("task"),
        "theme": prompt.get("theme"),
        "version": prompt.get("version"),
        "template": prompt.get("prompt_text"),
    }


def seed_prompt_template(cur, *, task: str, theme: str, template: str,
                         variables: list[str] | None = None,
                         model_provider: str | None = None,
                         model_name: str | None = None) -> None:
    """Seed/insert (ON CONFLICT DO NOTHING server-side)."""
    _request("POST", "/pipeline/prompts", json_body={
        "task": task,
        "theme": theme,
        "version": 1,
        "status": "active",
        "prompt_text": template,
    })


# ── Jobs / logs ──

def insert_pipeline_run(cur, pipeline_type: str) -> str:
    data = _request("POST", "/pipeline/jobs", json_body={"job_type": pipeline_type})
    return data.get("id") or gen_id()


def complete_pipeline_run(cur, run_id: str, stats: dict) -> None:
    _request("PATCH", f"/pipeline/jobs/{run_id}", json_body={
        "status": "completed",
        "stats_json": json.dumps(stats),
    })


def fail_pipeline_run(run_id: str, error: str) -> None:
    try:
        _request("PATCH", f"/pipeline/jobs/{run_id}", json_body={
            "status": "failed",
            "error": error[:2000],
        })
    except Exception as exc:
        log.warning("Could not report job failure to Worker: %s", exc)


def push_logs(logs: list[dict]) -> None:
    """Best-effort log shipping. Never raises."""
    if not logs:
        return
    try:
        _request("POST", "/pipeline/logs", json_body={"logs": logs})
    except Exception as exc:
        log.debug("push_logs failed (best-effort, ignored): %s", exc)
