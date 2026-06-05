"""Push notifications via Firebase Cloud Messaging."""

import json
import logging
import os

import httpx

log = logging.getLogger(__name__)


def send_fcm_notification(title: str, body: str, data: dict | None = None) -> bool:
    """Send a push notification via FCM HTTP v1 API.

    Requires FIREBASE_SERVER_KEY env var.
    For personal use, we send to a topic 'techpulse'.
    """
    server_key = os.environ.get("FIREBASE_SERVER_KEY")
    if not server_key:
        log.warning("FIREBASE_SERVER_KEY not set, skipping notification")
        return False

    try:
        payload = {
            "to": "/topics/techpulse",
            "notification": {
                "title": title,
                "body": body,
            },
        }
        if data:
            payload["data"] = {k: str(v) for k, v in data.items()}

        resp = httpx.post(
            "https://fcm.googleapis.com/fcm/send",
            headers={
                "Authorization": f"key={server_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )

        if resp.status_code == 200:
            log.info("Notification sent: %s", title)
            return True
        else:
            log.warning("FCM error: %d %s", resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        log.error("Notification error: %s", e)
        return False


def notify_pipeline_complete(stats: dict):
    """Send a notification that the daily briefing is ready."""
    clusters = stats.get("clusters_created", 0) + stats.get("clusters_updated", 0)
    analyses = stats.get("analyses_generated", 0)

    send_fcm_notification(
        title="TechPulse — Briefing prêt",
        body=f"{clusters} histoires, {analyses} analyses",
        data={"type": "briefing_ready"},
    )


def notify_weak_signal(signal_title: str, growth_score: int):
    """Send a notification about a detected weak signal."""
    send_fcm_notification(
        title="Signal faible détecté",
        body=f"{signal_title} (+{growth_score}%)",
        data={"type": "weak_signal"},
    )
