"""Read-only local artifact audit for the three-run paper registry."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ssep.analysis.paper.registry import PaperRegistry, load_registry


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: str
    required: bool
    present: bool
    detail: str = ""


def _zarr_detail(path: Path) -> str:
    try:
        metadata = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
        return f"shape={metadata.get('shape')} dtype={metadata.get('data_type')}"
    except (OSError, json.JSONDecodeError):
        return "invalid or missing zarr.json"


def _parquet_inventory(path: Path) -> tuple[int | None, set[str], str]:
    """Read only the Parquet footer rather than materializing a table."""
    if not path.is_file():
        return None, set(), "missing"
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        # FileMetaData.schema.names is the flattened Parquet leaf schema, so
        # list columns appear only as "element". ArrowSchema.names preserves
        # the top-level analysis columns required by the channel audit.
        columns = set(pq.read_schema(path).names)
        return metadata.num_rows, columns, f"rows={metadata.num_rows} columns={len(columns)}"
    except Exception as exc:
        return None, set(), f"unreadable parquet footer: {type(exc).__name__}: {exc}"


def audit_run(registry: PaperRegistry, alias: str) -> dict[str, Any]:
    spec = registry.run(alias)
    tables, acts = registry.tables_dir(alias), registry.acts_dir(alias)
    prompt_rows, prompt_columns, prompt_detail = _parquet_inventory(tables / "prompts.parquet")
    label_rows, label_columns, label_detail = _parquet_inventory(tables / "labels.parquet")
    checks = [
        ArtifactCheck(
            "run_metadata",
            str(registry.metadata_path(alias)),
            True,
            registry.metadata_path(alias).is_file(),
        ),
        ArtifactCheck(
            "prompts", str(tables / "prompts.parquet"), True, prompt_rows is not None, prompt_detail
        ),
        ArtifactCheck(
            "labels", str(tables / "labels.parquet"), True, label_rows is not None, label_detail
        ),
    ]
    for array in ("resid_post", "token_mask", "sample_hidden_last", "sample_hidden_mean"):
        path = acts / array
        checks.append(
            ArtifactCheck(
                array, str(path), True, (path / "zarr.json").is_file(), _zarr_detail(path)
            )
        )
    for array in ("attn_out", "mlp_out"):
        path = acts / array
        checks.append(
            ArtifactCheck(
                array, str(path), False, (path / "zarr.json").is_file(), _zarr_detail(path)
            )
        )
    metadata_error = None
    if registry.metadata_path(alias).is_file():
        try:
            metadata = json.loads(registry.metadata_path(alias).read_text(encoding="utf-8"))
            if metadata.get("run_id") != spec.run_id:
                metadata_error = "metadata run_id mismatch"
            elif metadata.get("model_name") != registry.model:
                metadata_error = "metadata model_name mismatch"
        except json.JSONDecodeError as exc:
            metadata_error = f"invalid metadata JSON: {exc}"
    row_count_error = None
    for table, rows in (("prompts", prompt_rows), ("labels", label_rows)):
        if rows is not None and rows != spec.expected_rows:
            row_count_error = (
                f"{table} has {rows} rows; registry expects realized count {spec.expected_rows}"
            )
            break
    channels = {
        "semantic_entropy": "se_discrete" in label_columns,
        "greedy_f1": "f1_squad" in label_columns,
        "greedy_judge_binary": {
            "judge_binary_label",
            "judge_binary_parse_fallback",
        }.issubset(label_columns),
        "sample_f1_fallback": "sample_f1" in label_columns,
        "sample_judge_binary": "sample_judge_binary_labels" in label_columns,
        "token_entropy": "greedy_entropy_full" in prompt_columns,
    }
    tables_complete = (
        prompt_rows == spec.expected_rows
        and label_rows == spec.expected_rows
        and registry.metadata_path(alias).is_file()
        and not metadata_error
    )
    activations_complete = all(
        item.present
        for item in checks
        if item.name in {"resid_post", "token_mask", "sample_hidden_last", "sample_hidden_mean"}
    )
    required_complete = tables_complete and activations_complete and not row_count_error
    study_readiness = {
        "A_se_vs_correctness_f1": required_complete
        and channels["semantic_entropy"]
        and channels["greedy_f1"],
        "A_se_vs_correctness_judge": required_complete
        and channels["semantic_entropy"]
        and channels["greedy_judge_binary"],
        "B_offline_fix_rate_f1": tables_complete
        and channels["greedy_f1"]
        and channels["sample_f1_fallback"],
        "B_offline_fix_rate_judge_greedy_f1_samples": tables_complete
        and channels["greedy_judge_binary"]
        and channels["sample_f1_fallback"],
        "B_offline_fix_rate_judge": tables_complete
        and channels["greedy_judge_binary"]
        and channels["sample_judge_binary"],
        "C_offline_routing_f1": tables_complete
        and channels["greedy_f1"]
        and channels["sample_f1_fallback"],
        "C_offline_routing_judge_greedy_f1_samples": tables_complete
        and channels["greedy_judge_binary"]
        and channels["sample_f1_fallback"],
    }
    return {
        "alias": alias,
        "run": asdict(spec),
        "required_complete_local": required_complete,
        "metadata_error": metadata_error,
        "row_count_error": row_count_error,
        "tables_complete_local": tables_complete,
        "activations_complete_local": activations_complete,
        "channels": channels,
        "study_readiness": study_readiness,
        "checks": [asdict(item) for item in checks],
    }


def audit_registry(registry: PaperRegistry, *, repo_root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(repo_root)
    result_families = registry.raw.get("required_result_families", [])
    result_checks = []
    for family in result_families:
        path = repo_root / "results" / str(family)
        result_checks.append({"family": family, "path": str(path), "present": path.is_dir()})
    return {
        "schema_version": 1,
        "paper_id": registry.paper_id,
        "runs": [audit_run(registry, alias) for alias in registry.runs],
        "result_families": result_checks,
        "confirmation_run": registry.evaluation.get("confirmation_run"),
        "confirmation_is_frozen": registry.evaluation.get("confirmation_run") is not None,
    }


def render_summary(report: dict[str, Any]) -> str:
    lines = [f"Paper artifact audit: {report['paper_id']}"]
    for run in report["runs"]:
        present = sum(check["present"] for check in run["checks"] if check["required"])
        required = sum(check["required"] for check in run["checks"])
        state = "READY" if run["required_complete_local"] else "INCOMPLETE"
        lines.append(f"- {run['alias']}: {state} ({present}/{required} required local artifacts)")
        ready = [name for name, value in run["study_readiness"].items() if value]
        lines.append(f"  study-ready: {ready or 'none'}")
        missing_channels = [name for name, value in run["channels"].items() if not value]
        lines.append(f"  missing channels: {missing_channels or 'none'}")
    missing_results = [item["family"] for item in report["result_families"] if not item["present"]]
    lines.append(f"- missing result families: {missing_results or 'none'}")
    lines.append(
        "- untouched confirmation: "
        + (str(report["confirmation_run"]) if report["confirmation_is_frozen"] else "NOT FROZEN")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--json", action="store_true", help="emit the full JSON audit")
    args = parser.parse_args(argv)
    registry = load_registry(args.config)
    report = audit_registry(registry)
    print(json.dumps(report, indent=2) if args.json else render_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
