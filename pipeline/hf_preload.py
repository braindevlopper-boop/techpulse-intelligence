"""Preload Hugging Face models used by the intelligence pipeline.

The preferred path is:
  1. Restore model cache archives from Cloudflare R2.
  2. Optionally download missing models from Hugging Face.
  3. Upload freshly downloaded cache archives back to R2.

Normal scheduled runs should not depend on Hugging Face availability. They
restore from R2 and run offline; manual refresh runs can update the R2 cache.
"""

import logging
import os
import tarfile
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
from botocore.exceptions import ClientError
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


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def hf_hub_cache_dir() -> Path:
    explicit_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser()

    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def model_cache_dir(model_id: str) -> Path:
    return hf_hub_cache_dir() / f"models--{model_id.replace('/', '--')}"


def r2_bucket_name() -> str | None:
    return os.getenv("HF_R2_BUCKET") or os.getenv("R2_BUCKET")


def r2_prefix() -> str:
    return os.getenv("HF_R2_PREFIX", "hf-model-cache/v1").strip("/")


def r2_key(model_id: str) -> str:
    safe_model = model_id.replace("/", "--")
    return f"{r2_prefix()}/{safe_model}.tar.gz"


def r2_client():
    endpoint = os.getenv("R2_ENDPOINT")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    if not endpoint or not access_key or not secret_key or not r2_bucket_name():
        return None

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe archive member path: {member.name}")
    tar.extractall(destination)


def restore_model_from_r2(model_id: str) -> bool:
    client = r2_client()
    bucket = r2_bucket_name()
    if client is None or not bucket:
        log.warning("R2 model cache not configured")
        return False

    key = r2_key(model_id)
    try:
        with NamedTemporaryFile(suffix=".tar.gz") as archive:
            client.download_file(bucket, key, archive.name)
            with tarfile.open(archive.name, "r:gz") as tar:
                safe_extract(tar, hf_hub_cache_dir())
        log.info("%s restored from R2 cache (%s)", model_id, key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            log.info("%s not found in R2 cache (%s)", model_id, key)
            return False
        log.warning("%s R2 restore failed: %s", model_id, exc)
        return False
    except Exception as exc:
        log.warning("%s R2 restore failed: %s", model_id, exc)
        return False


def upload_model_to_r2(model_id: str) -> bool:
    client = r2_client()
    bucket = r2_bucket_name()
    cache_dir = model_cache_dir(model_id)
    if client is None or not bucket:
        log.warning("R2 model cache not configured; skipping upload")
        return False
    if not cache_dir.exists():
        log.warning("%s cache directory does not exist, cannot upload", model_id)
        return False

    key = r2_key(model_id)
    with NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz", dereference=False) as tar:
            tar.add(cache_dir, arcname=cache_dir.name)
        client.upload_file(archive.name, bucket, key)

    log.info("%s uploaded to R2 cache (%s)", model_id, key)
    return True


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
        log.info("%s not fully cached locally", model_id)

    if restore_model_from_r2(model_id):
        return

    if not env_flag("TECHPULSE_HF_DOWNLOAD_MISSING", default=False):
        log.warning(
            "%s missing from local and R2 cache; skipping Hugging Face download",
            model_id,
        )
        return

    log.info("%s missing from R2 cache, downloading from Hugging Face", model_id)

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
            upload_model_to_r2(model_id)
            return
        except Exception as exc:
            if attempt == attempts:
                log.warning("%s download failed after %d attempts: %s", model_id, attempts, exc)
                return
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
    hf_hub_cache_dir().mkdir(parents=True, exist_ok=True)

    if token:
        log.info("Using authenticated Hugging Face downloads")
    else:
        log.warning("HF_TOKEN is not set; anonymous downloads may hit rate limits")
    if env_flag("TECHPULSE_HF_DOWNLOAD_MISSING", default=False):
        log.info("Missing models may be downloaded from Hugging Face and uploaded to R2")
    else:
        log.info("Hugging Face downloads disabled; using local/R2 cache only")

    for model_id in MODELS:
        preload_model(model_id, token)

    log.info("Hugging Face model preload complete")


if __name__ == "__main__":
    main()
