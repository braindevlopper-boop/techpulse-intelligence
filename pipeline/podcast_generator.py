"""Podcast generation — two-voice dialogue script + Gemini TTS + upload to R2.

Two paths:
  - Auto (topics=None): daily digest from top clusters, single voice, short.
    Currently disabled by TECHPULSE_SKIP_LEGACY_PODCAST in the workflow.
  - On-demand (topics=[...]): user-chosen candidates (news clusters and/or
    serendipity science cards), ~10min "approfondi" format, two-voice dialogue
    (Analyste vs Contradicteur) that dramatizes the counter_analysis angle.
"""

import base64
import io
import json
import logging
import os
import tempfile
import wave

import httpx

from . import db
from .llm_analyzer import analyze_with_gemini

log = logging.getLogger(__name__)

GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent"

SPEAKER_ANALYST = "Analyste"
SPEAKER_COUNTER = "Contradicteur"

DIALOGUE_SCRIPT_PROMPT = """Tu es le scénariste d'un podcast tech francophone à deux voix : "{speaker_analyst}" (présentateur, expose les faits et pourquoi ils comptent) et "{speaker_counter}" (contradicteur, challenge, tempère, apporte les risques et les angles morts).

Génère un script de podcast APPROFONDI d'environ 10 minutes (dialogue naturel, pas un simple résumé alterné) basé sur ces sujets :

{stories}

Règles :
- Alterne les deux voix pour un vrai dialogue, pas juste "un lit, l'autre commente"
- {speaker_analyst} pose le sujet, les faits, pourquoi c'est important
- {speaker_counter} challenge activement : contre-arguments, incertitudes, ce qui pourrait faire mentir la thèse
- Termine chaque sujet par un point d'accord ou de désaccord assumé entre les deux voix
- Commence par une accroche engageante, termine par une conclusion + teaser
- Ton naturel, conversationnel, jamais scolaire
- Chaque ligne DOIT commencer par "{speaker_analyst}: " ou "{speaker_counter}: " (obligatoire pour la synthèse vocale)
- Ne mets aucune indication de mise en scène, uniquement le texte à lire

Réponds avec un JSON :
{{
  "title": "titre du podcast",
  "script": "le dialogue complet, chaque ligne préfixée par le nom du speaker suivi de deux-points",
  "description": "description courte (2 phrases)"
}}"""


def _pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Gemini TTS returns raw PCM (mono, 24kHz, 16-bit) — wrap it in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


def text_to_speech_gemini(script: str) -> bytes | None:
    """Two-voice TTS via Gemini 2.5 Flash. Returns WAV bytes, or None on failure.

    NOTE: the exact multiSpeakerVoiceConfig shape below is based on Google's
    documented REST conventions but was not exercised against a live API key
    during implementation — check logs on the first real run in case the
    field names need a small correction.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning("GEMINI_API_KEY not set, cannot generate TTS")
        return None

    try:
        resp = httpx.post(
            f"{GEMINI_TTS_URL}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": script}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "multiSpeakerVoiceConfig": {
                            "speakerVoiceConfigs": [
                                {
                                    "speaker": SPEAKER_ANALYST,
                                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
                                },
                                {
                                    "speaker": SPEAKER_COUNTER,
                                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}},
                                },
                            ]
                        }
                    },
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        b64_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        pcm_bytes = base64.b64decode(b64_audio)
        return _pcm_to_wav_bytes(pcm_bytes)
    except Exception as e:
        log.error("Gemini TTS error: %s", e)
        return None


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
            ExtraArgs={"ContentType": "audio/wav"},
        )

        r2_public = os.environ.get("R2_PUBLIC_URL", r2_endpoint)
        return f"{r2_public}/{r2_bucket}/{filename}"
    except Exception as e:
        log.error("R2 upload error: %s", e)
        return None


def _topic_text(topic: dict) -> str:
    if topic["type"] == "cluster":
        return f"{topic['title']}\n   {topic.get('content', '')}"
    return f"[Science] {topic['title']}\n   {topic.get('content', '')}"


def _default_topics(cur) -> list[dict]:
    """Auto-pick top clusters (legacy daily digest behavior)."""
    top_clusters = db.fetch_top_clusters(cur, limit=7)[:5]
    topics = []
    for c in top_clusters:
        articles = db.fetch_cluster_articles(cur, c["id"])
        source_names = list(set(a["source_name"] for a in articles[:5]))
        topics.append({
            "type": "cluster",
            "id": c["id"],
            "title": c["title"],
            "content": f"Sources: {', '.join(source_names)} | Importance: {c['importance_score']}",
        })
    return topics


def resolve_topics(cur, topic_refs: list[tuple[str, str]]) -> list[dict]:
    """Resolve (type, id) pairs into topic dicts, fetching clusters and serendipity
    cards in one query each rather than one-by-one."""
    cluster_ids = [tid for ttype, tid in topic_refs if ttype == "cluster"]
    card_ids = [tid for ttype, tid in topic_refs if ttype == "serendipity"]

    clusters_by_id = {c["id"]: c for c in db.fetch_clusters_by_ids(cur, cluster_ids)}
    cards_by_id = {c["id"]: c for c in db.fetch_serendipity_cards_by_ids(cur, card_ids)}

    topics = []
    for ttype, tid in topic_refs:
        if ttype == "cluster" and tid in clusters_by_id:
            c = clusters_by_id[tid]
            articles = db.fetch_cluster_articles(cur, tid)
            source_names = list(set(a["source_name"] for a in articles[:5]))
            topics.append({
                "type": "cluster",
                "id": tid,
                "title": c["title"],
                "content": f"Sources: {', '.join(source_names)} | Importance: {c['importance_score']}",
            })
        elif ttype == "serendipity" and tid in cards_by_id:
            card = cards_by_id[tid]
            content = " ".join(filter(None, [card.get("enigme"), card.get("concept"), card.get("so_what")]))
            topics.append({
                "type": "serendipity",
                "id": tid,
                "title": card["title_choc"],
                "content": content,
            })
    return topics


def generate_podcast(
    cur,
    podcast_type: str = "daily",
    topics: list[dict] | None = None,
    target_minutes: int = 10,
) -> str | None:
    """Generate a podcast from topics (mixed clusters + serendipity cards).

    Pass explicit `topics` (see resolve_topics) for an on-demand podcast built
    from user-chosen candidates. Leave None for the auto daily digest.
    """
    if topics is None:
        topics = _default_topics(cur)

    if len(topics) < 1:
        log.info("Not enough topics for podcast")
        return None

    stories = "\n".join(f"{i}. {_topic_text(t)}" for i, t in enumerate(topics, 1))

    prompt = DIALOGUE_SCRIPT_PROMPT.format(
        stories=stories,
        speaker_analyst=SPEAKER_ANALYST,
        speaker_counter=SPEAKER_COUNTER,
    )
    result = analyze_with_gemini(prompt)

    if not result or "script" not in result:
        log.error("Failed to generate podcast script")
        return None

    script = result["script"]
    title = result.get("title", f"TechPulse — {podcast_type}")
    description = result.get("description", "")

    log.info("Podcast script generated: %s (%d chars)", title, len(script))

    audio_bytes = text_to_speech_gemini(script)
    if not audio_bytes:
        log.error("TTS failed")
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        file_size = os.path.getsize(audio_path)
        duration = file_size // 48000  # 24kHz * 2 bytes/sample = 48000 bytes/sec (mono)

        filename = f"podcast-{podcast_type}-{db.gen_id()}.wav"
        audio_url = upload_to_r2(audio_path, filename)

        if not audio_url:
            log.warning("R2 upload failed, podcast saved locally: %s", audio_path)
            audio_url = f"local://{audio_path}"

        topic_ids = [t["id"] for t in topics]
        podcast_id = db.insert_podcast(
            cur, title, description, script,
            audio_url, duration, topic_ids, podcast_type,
        )

        log.info("Podcast created: %s (duration ~%ds, %d topics)", podcast_id, duration, len(topics))
        return podcast_id

    finally:
        if os.path.exists(audio_path) and "local://" not in (audio_url or ""):
            os.remove(audio_path)
