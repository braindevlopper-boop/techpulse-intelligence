"""Preload Hugging Face models used by the intelligence pipeline.

This keeps model downloads out of the hot path and makes GitHub Actions
benefit from the ~/.cache/huggingface cache before the actual pipeline starts.
"""

import logging
import os
import time

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf-preload")

MODELS = [
    "dslim/bert-base-NER",
    "cross-encoder/nli-deberta-v3-base",
    "BAAI/bge-small-en-v1.5",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]

MODEL_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "merges.txt",
    "vocab.*",
    "tokenizer.*",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/*",
]


def preload_model(model_id: str, token: str | None, attempts: int = 4) -> None:
    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=MODEL_ALLOW_PATTERNS,
            local_files_only=True,
        )
        log.info("%s already available in local cache", model_id)
        return
    except LocalEntryNotFoundError:
        log.info("%s not fully cached, downloading", model_id)

    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=model_id,
                allow_patterns=MODEL_ALLOW_PATTERNS,
                token=token,
                max_workers=1,
                resume_download=True,
            )
            log.info("%s downloaded", model_id)
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            sleep_seconds = min(90, 10 * 2 ** (attempt - 1))
            log.warning(
                "%s download failed on attempt %d/%d: %s. Retrying in %ds",
                model_id,
                attempt,
                attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def main() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        log.info("Using authenticated Hugging Face downloads")
    else:
        log.warning("HF_TOKEN is not set; anonymous downloads may hit rate limits")

    for model_id in MODELS:
        preload_model(model_id, token)

    log.info("Hugging Face model preload complete")


if __name__ == "__main__":
    main()
