"""Typed registry for paper runs and experiment-wide invariants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    alias: str
    run_id: str
    run_group: str
    dataset: str
    source_family: str
    prompt_regime: str
    expected_rows: int
    paired_with: str | None = None

    @classmethod
    def from_mapping(cls, alias: str, raw: dict[str, Any]) -> RunSpec:
        required = {
            "run_id",
            "run_group",
            "dataset",
            "source_family",
            "prompt_regime",
            "expected_rows",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"run {alias!r} is missing {sorted(missing)}")
        expected_rows = int(raw["expected_rows"])
        if expected_rows < 1:
            raise ValueError(f"run {alias!r}.expected_rows must be positive")
        return cls(
            alias=alias,
            run_id=str(raw["run_id"]),
            run_group=str(raw["run_group"]),
            dataset=str(raw["dataset"]),
            source_family=str(raw["source_family"]),
            prompt_regime=str(raw["prompt_regime"]),
            expected_rows=expected_rows,
            paired_with=(str(raw["paired_with"]) if raw.get("paired_with") else None),
        )


@dataclass(frozen=True)
class PaperRegistry:
    version: int
    paper_id: str
    model: str
    artifact_root: Path
    output_root: Path
    runs: dict[str, RunSpec]
    evaluation: dict[str, Any]
    raw: dict[str, Any]

    def run(self, alias: str) -> RunSpec:
        try:
            return self.runs[alias]
        except KeyError as exc:
            raise KeyError(f"unknown run alias {alias!r}; choose from {sorted(self.runs)}") from exc

    def tables_dir(self, alias: str) -> Path:
        spec = self.run(alias)
        model_safe = self.model.replace("/", "--")
        return self.artifact_root / "tables" / "v1" / model_safe / spec.dataset

    def acts_dir(self, alias: str) -> Path:
        spec = self.run(alias)
        model_safe = self.model.replace("/", "--")
        return self.artifact_root / "acts" / "v1" / model_safe / spec.dataset

    def metadata_path(self, alias: str) -> Path:
        return self.artifact_root / "runs" / self.run(alias).run_id / "run_metadata.json"

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported paper registry version {self.version}")
        if not self.runs:
            raise ValueError("paper registry must define at least one run")
        run_ids = [run.run_id for run in self.runs.values()]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("run_id values must be unique")
        datasets = [run.dataset for run in self.runs.values()]
        if len(datasets) != len(set(datasets)):
            raise ValueError(
                "dataset storage identities must be unique; acts/tables are not scoped by run_group"
            )
        for alias, run in self.runs.items():
            expected_id = f"{self.model.replace('/', '--')}--{run.run_group}"
            if run.run_id != expected_id:
                raise ValueError(
                    f"run {alias!r} id {run.run_id!r} does not match model/run_group {expected_id!r}"
                )
            if run.paired_with:
                if run.paired_with not in self.runs:
                    raise ValueError(f"run {alias!r} pairs with unknown run {run.paired_with!r}")
                other = self.runs[run.paired_with]
                if other.paired_with != alias:
                    raise ValueError(
                        f"pairing must be symmetric: {alias!r} <-> {run.paired_with!r}"
                    )
                if other.source_family != run.source_family:
                    raise ValueError(
                        f"paired runs {alias!r}/{other.alias!r} have different sources"
                    )
        folds = int(self.evaluation.get("crossfit_folds", 0))
        if folds < 2:
            raise ValueError("evaluation.crossfit_folds must be at least two")
        splits = tuple(self.evaluation.get("canonical_splits", ()))
        if splits != ("discovery", "train", "test"):
            raise ValueError("canonical_splits must be [discovery, train, test] in that order")


def load_registry(path: str | Path) -> PaperRegistry:
    """Load and fully validate a YAML paper registry."""
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paper registry must be a mapping")
    raw_runs = raw.get("runs")
    if not isinstance(raw_runs, dict):
        raise ValueError("paper registry runs must be a mapping")
    registry = PaperRegistry(
        version=int(raw.get("version", 0)),
        paper_id=str(raw.get("paper_id", "")),
        model=str(raw.get("model", "")),
        artifact_root=Path(raw.get("artifact_root", "data/azure")),
        output_root=Path(raw.get("output_root", "results/paper")),
        runs={alias: RunSpec.from_mapping(alias, item) for alias, item in raw_runs.items()},
        evaluation=dict(raw.get("evaluation", {})),
        raw=raw,
    )
    registry.validate()
    return registry
