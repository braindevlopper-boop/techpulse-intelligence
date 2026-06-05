"""Podcast generation — LLM script + Edge TTS + upload to R2."""

import asyncio
import json
import logging
import os
import tempfile

import edge_tts
import httpx

from . import db
from .llm_analyzer import analyze_with_gemini

log = logging.getLogger(__name__)

PODCAST_SCRIPT_PROMPT = """Tu es un présentateur de podcast tech francophone.
Génère un script de podcast de 5 minutes maximum basé sur ces histoires du jour.

Histoires :
{stories}

Règles :
- Commence par une accroche engageante
- Présente chaque histoire en 30-60 secondes
- Ton conversationnel mais informatif
- Termine par une conclusion et un teaser pour demain
- Écris en français naturel, pas trop formel
- Ne mets pas d'indications de mise en scène, juste le texte à lire

Réponds avec un JSON :
{{
  "title": "titre du podcast",
  "script": "le texte complet à lire",
  "description": "description courte (2 phrases)"
}}"""

TTS_VOICE = "fr-FR-VivienneMultilingualNeural"


async def text_to_speech(text: str, output_path: str) -> bool:
    """Convert text to speech using Edge TTS."""
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        log.error("TTS error: %s", e)
        return False


def upload_to_r2(file_path: str, filename: str) -> str | None:
    """Upload audio file to Cloudflare R2 via S3-compatible API."""
    import boto3

    r2_endpoint = os.environ.get("R2_ENDPOINT")
    r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.environ.get("R2_BUCKET", "techpulse-podcasts")

    if not all([r2_endpoint, r2_access_key, r2_secret_key]):
        log.warning("R2 credentials not configured")
        return None

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            region_name="auto",
        )

        s3.upload_file(
            file_path, r2_bucket, filename,
            ExtraArgs={"ContentType": "audio/mpeg"},
        )

        r2_public = os.environ.get("R2_PUBLIC_URL", r2_endpoint)
        return f"{r2_public}/{r2_bucket}/{filename}"
    except Exception as e:
        log.error("R2 upload error: %s", e)
        return None


def generate_podcast(cur, podcast_type: str = "daily") -> str | None:
    """Generate a podcast from top clusters."""
    top_clusters = db.fetch_top_clusters(cur, limit=7)
    if len(top_clusters) < 2:
        log.info("Not enough clusters for podcast")
        return None

    stories = ""
    cluster_ids = []
    for i, c in enumerate(top_clusters[:5], 1):
        articles = db.fetch_cluster_articles(cur, c["id"])
        source_names = list(set(a["source_name"] for a in articles[:5]))
        stories += f"\n{i}. {c['title']}\n   Sources: {', '.join(source_names)}\n   Importance: {c['importance_score']}\n"
        cluster_ids.append(c["id"])

    prompt = PODCAST_SCRIPT_PROMPT.format(stories=stories)
    result = analyze_with_gemini(prompt)

    if not result or "script" not in result:
        log.error("Failed to generate podcast script")
        return None

    script = result["script"]
    title = result.get("title", f"TechPulse — {podcast_type}")
    description = result.get("description", "")

    log.info("Podcast script generated: %s (%d chars)", title, len(script))

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        audio_path = f.name

    try:
        success = asyncio.run(text_to_speech(script, audio_path))
        if not success:
            log.error("TTS failed")
            return None

        file_size = os.path.getsize(audio_path)
        duration = file_size // 16000

        filename = f"podcast-{podcast_type}-{db.gen_id()}.mp3"
        audio_url = upload_to_r2(audio_path, filename)

        if not audio_url:
            log.warning("R2 upload failed, podcast saved locally: %s", audio_path)
            audio_url = f"local://{audio_path}"

        podcast_id = db.insert_podcast(
            cur, title, description, script,
            audio_url, duration, cluster_ids, podcast_type,
        )

        log.info("Podcast created: %s (duration ~%ds)", podcast_id, duration)
        return podcast_id

    finally:
        if os.path.exists(audio_path) and "local://" not in (audio_url or ""):
            os.remove(audio_path)
