"""Shared utilities: content hashing (schema §0/§1 provenance — every hash in
the store is sha256 hex), the seed policy (§0 `seed_policy` / §5 `sample_seeds`),
.env + fail-fast environment access (invariant #7), and logging setup."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

# -- hashing ------------------------------------------------------------------


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_hash(realized_prompt: str) -> str:
    """Schema §1 prompt_hash: sha256 of the full realized input string."""
    return sha256_str(realized_prompt)


# -- seeding ------------------------------------------------------------------

SEED_POLICY_ID = "sha256(base_seed:prompt_id:sample_idx) % 2**63"


def derive_sample_seed(base_seed: int, prompt_id: str, sample_idx: int) -> int:
    """Each of the N vLLM samples gets its own derived seed, so any single
    sample is independently reproducible from (base_seed, prompt_id, idx)."""
    payload = f"{base_seed}:{prompt_id}:{sample_idx}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def derive_named_seed(base_seed: int, name: str) -> int:
    """Deterministic seed for a named sub-draw (e.g. few-shot index sampling)."""
    payload = f"{base_seed}:{name}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


# -- environment / secrets ----------------------------------------------------
# Secrets live in the repo-local .env (gitignored) on dev machines and in
# RunPod Secrets on pods; either way they surface as process env vars.

AZURE_ENV_KEYS = ("AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_CONTAINER", "AZURE_STORAGE_SAS_TOKEN")


class MissingEnvironmentError(RuntimeError):
    pass


def read_dotenv(path: str | Path = ".env") -> None:
    """Parse a KEY=VALUE .env file (comments/blank lines ignored, optional quotes
    stripped) into os.environ, never overriding keys already present."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val


def require_env(keys: tuple[str, ...] | list[str], *, context: str) -> dict[str, str]:
    """Return the requested env vars; raise naming every missing key (no silent defaults)."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise MissingEnvironmentError(
            f"{context}: missing required environment variable(s): {', '.join(missing)}. "
            "Set them in .env (dev) or RunPod Secrets (pod)."
        )
    return {k: os.environ[k] for k in keys}


# -- logging ------------------------------------------------------------------

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("ssep")
        if not root.handlers:
            root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)
