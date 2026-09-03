"""Build frozen representation baselines from an aligned feature matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssep.analysis.paper.artifacts import atomic_json, create_experiment_manifest


def transform_features(
    *,
    features_path: Path,
    rows_path: Path,
    output_dir: Path,
    method: str,
    dimensions: int | None,
    target_l0: float | None = None,
    seed: int,
    auxiliary_features_path: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = create_experiment_manifest(
        experiment_id=output_dir.name,
        study="shared",
        config_path="configs/paper/three_run_working_paper.yaml",
        input_paths=[features_path, rows_path]
        + ([auxiliary_features_path] if auxiliary_features_path is not None else []),
        split_protocol={
            "method": method,
            "dimensions": dimensions,
            "target_l0": target_l0,
            "seed": seed,
            "auxiliary_features_path": (
                str(auxiliary_features_path) if auxiliary_features_path is not None else None
            ),
            "fit_split": "train" if method == "pca" else None,
        },
    )
    atomic_json(output_dir / "manifest.pre_materialization.json", manifest)
    source = np.load(features_path, mmap_mode="r")
    rows = pd.read_parquet(rows_path)
    if source.ndim != 2 or len(source) != len(rows) or "split" not in rows:
        raise ValueError("features and row map must be aligned and include split")
    if dimensions is not None and dimensions < 1:
        raise ValueError("dimensions must be positive")
    if method in {"pca", "dense_row_topk"} and dimensions is not None:
        if dimensions > source.shape[1]:
            raise ValueError(f"{method} dimensions cannot exceed the source feature width")

    if method == "pca":
        from sklearn.decomposition import PCA

        if dimensions is None:
            raise ValueError("pca requires dimensions")
        train = rows.split.to_numpy() == "train"
        transformer = PCA(n_components=dimensions, random_state=seed).fit(source[train])
        transformed = transformer.transform(source).astype(np.float32)
        details = {
            "explained_variance_ratio_sum": float(transformer.explained_variance_ratio_.sum())
        }
    elif method == "random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        if dimensions is None:
            raise ValueError("random_projection requires dimensions")
        transformer = GaussianRandomProjection(n_components=dimensions, random_state=seed)
        transformed = transformer.fit_transform(source).astype(np.float32)
        details = {"distribution": "gaussian", "label_fit": False}
    elif method == "random_relu_matched_l0":
        from sklearn.random_projection import SparseRandomProjection

        if dimensions is None or target_l0 is None or not 0 < target_l0 < dimensions:
            raise ValueError("random_relu_matched_l0 requires dimensions > target_l0 > 0")
        transformer = SparseRandomProjection(n_components=dimensions, random_state=seed)
        transformer.fit(source[:1])
        train_indices = np.flatnonzero(rows.split.to_numpy() == "train")
        calibration_indices = np.random.default_rng(seed).choice(
            train_indices, size=min(512, len(train_indices)), replace=False
        )
        calibration = transformer.transform(source[calibration_indices])
        threshold = float(np.quantile(calibration, 1 - target_l0 / dimensions))
        transformed = np.empty((len(source), dimensions), dtype=np.float32)
        for start in range(0, len(source), 256):
            stop = min(start + 256, len(source))
            projected = transformer.transform(source[start:stop])
            transformed[start:stop] = np.maximum(projected - threshold, 0).astype(np.float32)
        details = {
            "baseline": "untrained sparse-random encoder plus ReLU",
            "target_l0": target_l0,
            "threshold_fit_split": "train",
            "threshold_calibration_rows": len(calibration_indices),
            "threshold": threshold,
        }
    elif method == "dense_row_topk":
        if dimensions is None:
            raise ValueError("dense_row_topk requires dimensions")
        transformed = np.zeros(source.shape, dtype=np.float32)
        selected = np.argpartition(np.abs(source), -dimensions, axis=1)[:, -dimensions:]
        row_index = np.arange(len(source))[:, None]
        transformed[row_index, selected] = source[row_index, selected]
        details = {"nonzeros_per_row_max": dimensions}
    elif method == "l1_row_normalize":
        denominator = np.maximum(np.abs(source).sum(axis=1, dtype=np.float64), 1e-12)
        transformed = (source / denominator[:, None]).astype(np.float32)
        details = {"fit": False}
    elif method == "permuted_sae":
        # This is a selection-null baseline, not a randomly trained autoencoder:
        # it exactly preserves per-feature marginal sparsity within each split
        # while destroying prompt-feature associations.
        transformed = np.empty(source.shape, dtype=np.float32)
        rng = np.random.default_rng(seed)
        for split in ("discovery", "train", "test"):
            indices = np.flatnonzero(rows.split.to_numpy() == split)
            for column in range(source.shape[1]):
                transformed[indices, column] = source[rng.permutation(indices), column]
        details = {"null": "independent per-feature permutation within canonical split"}
    elif method == "prepend_features":
        if auxiliary_features_path is None:
            raise ValueError("prepend_features requires --auxiliary-features")
        auxiliary = np.load(auxiliary_features_path, mmap_mode="r")
        if auxiliary.ndim != 2 or len(auxiliary) != len(source):
            raise ValueError("auxiliary features must be a row-aligned matrix")
        transformed = np.concatenate((auxiliary, source), axis=1).astype(np.float32)
        details = {
            "operation": "auxiliary_then_source_column_concatenation",
            "auxiliary_dimensions": int(auxiliary.shape[1]),
        }
    else:
        raise ValueError(f"unknown representation transform {method!r}")

    np.save(output_dir / "features.npy", transformed, allow_pickle=False)
    rows.to_parquet(output_dir / "rows.parquet", index=False)
    summary = {
        "method": method,
        "source_shape": list(source.shape),
        "output_shape": list(transformed.shape),
        "dimensions": dimensions,
        "target_l0": target_l0,
        "seed": seed,
        **details,
    }
    atomic_json(output_dir / "summary.json", summary)
    completed = dict(manifest)
    completed["status"] = "materialized"
    completed["outputs"] = summary
    atomic_json(output_dir / "manifest.json", completed)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "pca",
            "random_projection",
            "random_relu_matched_l0",
            "dense_row_topk",
            "l1_row_normalize",
            "permuted_sae",
            "prepend_features",
        ),
        required=True,
    )
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--target-l0", type=float)
    parser.add_argument("--auxiliary-features", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(
        transform_features(
            features_path=args.features,
            rows_path=args.rows,
            output_dir=args.output,
            method=args.method,
            dimensions=args.dimensions,
            target_l0=args.target_l0,
            seed=args.seed,
            auxiliary_features_path=args.auxiliary_features,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
