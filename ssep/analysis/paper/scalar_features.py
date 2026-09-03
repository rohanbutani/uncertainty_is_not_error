"""Materialize the first-generated-token entropy baseline aligned to paper rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssep.analysis.paper.artifacts import atomic_json, create_experiment_manifest
from ssep.analysis.paper.data_contract import first_token_entropy, load_run_frame
from ssep.analysis.paper.registry import load_registry


def materialize_token_entropy(
    *, config_path: Path, run_alias: str, rows_path: Path, output_path: Path
) -> Path:
    registry = load_registry(config_path)
    frame, _ = load_run_frame(registry, run_alias)
    rows = pd.read_parquet(rows_path)
    if rows.prompt_id.isna().any() or rows.prompt_id.duplicated().any():
        raise ValueError("row map prompt_id must be complete and unique")
    joined = rows[["prompt_id"]].merge(frame, on="prompt_id", how="left", validate="one_to_one")
    if joined.source_id.isna().any():
        raise ValueError("row map contains prompt IDs absent from registered run")
    values = first_token_entropy(joined).astype(np.float32)[:, None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    np.save(output_path, values)
    manifest = create_experiment_manifest(
        experiment_id=output_path.stem,
        study="shared",
        config_path=config_path,
        input_paths=[
            rows_path,
            registry.metadata_path(run_alias),
            registry.tables_dir(run_alias) / "prompts.parquet",
            registry.tables_dir(run_alias) / "labels.parquet",
        ],
        split_protocol={
            "run_alias": run_alias,
            "representation": "first_generated_token_entropy",
            "alignment": "prompt_id",
            "fit": "none",
        },
    )
    manifest["status"] = "materialized"
    manifest["outputs"] = {"path": str(output_path), "shape": list(values.shape)}
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        materialize_token_entropy(
            config_path=Path(args.config),
            run_alias=args.run,
            rows_path=args.rows,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
