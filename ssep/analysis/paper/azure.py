"""Read-only Azure access for paper tables and activation arrays.

Credentials are read from the environment and are deliberately never returned in
provenance records, paths, or exception messages.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from ssep.analysis.paper.registry import PaperRegistry
from ssep.utils import AZURE_ENV_KEYS, read_dotenv, require_env


def azure_store():
    """Construct the project container's authenticated, read-only-use store."""
    from obstore.store import AzureStore

    read_dotenv()
    env = require_env(AZURE_ENV_KEYS, context="paper Azure read")
    return AzureStore(
        env["AZURE_STORAGE_CONTAINER"],
        account_name=env["AZURE_STORAGE_ACCOUNT"],
        sas_key=env["AZURE_STORAGE_SAS_TOKEN"].lstrip("?"),
        credential_type="sas_token",
    )


def remote_bytes(store: Any, key: str) -> bytes:
    import obstore as obs

    return bytes(obs.get(store, key).bytes())


def remote_head(store: Any, key: str) -> dict[str, Any]:
    """Return a JSON-safe, credential-free object provenance record."""
    import obstore as obs

    raw = obs.head(store, key)
    result: dict[str, Any] = {"key": key, "source": "azure"}
    for name in ("size", "e_tag", "version", "last_modified"):
        value = raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)
        if value is not None:
            result[name] = value.isoformat() if hasattr(value, "isoformat") else value
    return result


def run_object_keys(registry: PaperRegistry, alias: str) -> dict[str, str]:
    spec = registry.run(alias)
    model_safe = registry.model.replace("/", "--")
    table_root = f"tables/v1/{model_safe}/{spec.dataset}"
    return {
        "metadata": f"runs/{spec.run_id}/run_metadata.json",
        "prompts": f"{table_root}/prompts.parquet",
        "labels": f"{table_root}/labels.parquet",
        "resid_post_metadata": f"acts/v1/{model_safe}/{spec.dataset}/resid_post/zarr.json",
    }


def load_remote_run_objects(
    registry: PaperRegistry, alias: str, *, store: Any | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Load the comparatively small paper tables directly from Azure."""
    store = store or azure_store()
    keys = run_object_keys(registry, alias)
    prompts = pd.read_parquet(BytesIO(remote_bytes(store, keys["prompts"])))
    labels = pd.read_parquet(BytesIO(remote_bytes(store, keys["labels"])))
    metadata = json.loads(remote_bytes(store, keys["metadata"]))
    provenance = [remote_head(store, keys[name]) for name in ("metadata", "prompts", "labels")]
    return prompts, labels, metadata, provenance


def open_remote_array(
    registry: PaperRegistry, alias: str, array_name: str, *, store: Any | None = None
):
    """Open one Zarr array without downloading the activation bank."""
    import zarr
    from zarr.storage import ObjectStore

    store = store or azure_store()
    model_safe = registry.model.replace("/", "--")
    path = f"acts/v1/{model_safe}/{registry.run(alias).dataset}/{array_name}"
    return zarr.open_array(store=ObjectStore(store, read_only=True), path=path, mode="r")


def download_remote_file(store: Any, key: str, destination: str | Path) -> Path:
    """Download one non-activation artifact atomically (useful for local audits)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(remote_bytes(store, key))
    temporary.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Print a credential-free schema inventory for one registered remote run."""
    from ssep.analysis.paper.registry import load_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--cache-tables",
        action="store_true",
        help="atomically cache metadata/prompts/labels at the registry's local paths",
    )
    args = parser.parse_args(argv)
    registry = load_registry(args.config)
    prompts, labels, metadata, provenance = load_remote_run_objects(registry, args.run)
    if args.cache_tables:
        store = azure_store()
        keys = run_object_keys(registry, args.run)
        download_remote_file(store, keys["metadata"], registry.metadata_path(args.run))
        download_remote_file(
            store, keys["prompts"], registry.tables_dir(args.run) / "prompts.parquet"
        )
        download_remote_file(
            store, keys["labels"], registry.tables_dir(args.run) / "labels.parquet"
        )
    print(
        json.dumps(
            {
                "run": args.run,
                "prompt_rows": len(prompts),
                "prompt_columns": list(prompts.columns),
                "label_rows": len(labels),
                "label_columns": list(labels.columns),
                "metadata_run_id": metadata.get("run_id"),
                "cached_tables": args.cache_tables,
                "objects": provenance,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
