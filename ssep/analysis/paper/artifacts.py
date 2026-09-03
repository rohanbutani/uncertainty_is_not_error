"""Atomic, hash-addressed output manifests for paper experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo_root: str | Path = ".") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class InputArtifact:
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def inspect(cls, path: str | Path) -> InputArtifact:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(str(path), path.stat().st_size, sha256_file(path))


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def create_experiment_manifest(
    *,
    experiment_id: str,
    study: str,
    config_path: str | Path,
    input_paths: list[str | Path],
    split_protocol: dict[str, Any],
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Create the provenance payload written before a test-set read."""
    if study not in {"A", "B", "C", "shared"}:
        raise ValueError("study must be A, B, C, or shared")
    artifacts = [InputArtifact.inspect(path) for path in input_paths]
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "study": study,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(repo_root),
        "config": asdict(InputArtifact.inspect(config_path)),
        "inputs": [asdict(item) for item in artifacts],
        "split_protocol": split_protocol,
        "test_read_count": 0,
        "status": "prepared",
    }


def mark_test_read(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a completed manifest, rejecting an accidental second test read."""
    if int(manifest.get("test_read_count", 0)) != 0:
        raise RuntimeError("test set was already read for this experiment manifest")
    updated = dict(manifest)
    updated["test_read_count"] = 1
    updated["status"] = "test_evaluated"
    updated["test_read_utc"] = datetime.now(UTC).isoformat()
    return updated
