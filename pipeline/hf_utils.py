"""Shared helpers for optional Hugging Face based steps."""

import logging
import os


def hf_steps_disabled() -> bool:
    value = os.getenv("TECHPULSE_SKIP_HF_ML", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def hf_failure_allowed() -> bool:
    value = os.getenv("TECHPULSE_ALLOW_HF_FAILURE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def handle_hf_unavailable(log: logging.Logger, step_name: str, error: Exception) -> bool:
    if not hf_failure_allowed():
        log.error(
            "%s failed: Hugging Face model unavailable (%s). "
            "Set TECHPULSE_ALLOW_HF_FAILURE=true only for degraded emergency runs.",
            step_name,
            error,
        )
        raise error

    log.warning(
        "%s skipped: Hugging Face model unavailable (%s). "
        "The pipeline will continue with clustering, scoring and LLM analysis.",
        step_name,
        error,
    )
    return True
