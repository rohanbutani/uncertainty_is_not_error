"""Materialize aligned dense states and optional SAE codes for any registered run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ssep.analysis.paper.artifacts import atomic_json, create_experiment_manifest
from ssep.analysis.paper.data_contract import (
    load_run_frame,
    position_indices,
    validate_run_objects,
)
from ssep.analysis.paper.registry import load_registry


def _drop_prompt_exemplars(frame: pd.DataFrame, metadata: dict, *, dataset: str) -> pd.DataFrame:
    dataset_meta = next(item for item in metadata["datasets"] if item["name"] == dataset)
    excluded = set(dataset_meta.get("ptrue_fewshot_ids", []))
    result = frame.loc[~frame.prompt_id.isin(excluded)].reset_index(drop=True)
    if not len(result) or result.source_id.duplicated().any():
        raise ValueError("eligible paper rows must be nonempty and source_id-unique")
    return result


def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def materialize_run_site(
    *,
    config_path: Path,
    run_alias: str,
    layer: int,
    position: str,
    output_dir: Path,
    normalization: str,
    encode_sae: bool,
    sae_width: str,
    sae_l0: str,
    batch_size: int,
    device: str,
    activation_source: str = "auto",
) -> Path:
    registry = load_registry(config_path)
    spec = registry.run(run_alias)
    local_metadata = registry.metadata_path(run_alias)
    local_prompts = registry.tables_dir(run_alias) / "prompts.parquet"
    local_labels = registry.tables_dir(run_alias) / "labels.parquet"
    zarr_metadata = registry.acts_dir(run_alias) / "resid_post" / "zarr.json"
    local_tables = all(path.is_file() for path in (local_metadata, local_prompts, local_labels))
    local_activations = zarr_metadata.is_file()
    if activation_source not in {"auto", "local", "azure"}:
        raise ValueError("activation_source must be auto, local, or azure")
    resolved_activation_source = (
        "local" if activation_source == "auto" and local_activations else activation_source
    )
    if resolved_activation_source == "auto":
        resolved_activation_source = "azure"
    if resolved_activation_source == "local" and not local_activations:
        raise FileNotFoundError(f"local resid_post metadata is missing: {zarr_metadata}")

    azure = None
    remote_inputs: list[dict] = []
    if not local_tables or resolved_activation_source == "azure":
        from ssep.analysis.paper.azure import azure_store

        azure = azure_store()
    input_paths = (
        [local_metadata, local_prompts, local_labels] if local_tables else []
    ) + ([zarr_metadata] if resolved_activation_source == "local" else [])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = create_experiment_manifest(
        experiment_id=output_dir.name,
        study="shared",
        config_path=config_path,
        input_paths=input_paths,
        split_protocol={
            "run_alias": run_alias,
            "layer": layer,
            "resid_post_index": layer + 1,
            "position": position,
            "normalization": normalization,
            "exclude_ptrue_fewshot_ids": True,
            "sae_encoded": encode_sae,
            "table_source": "local" if local_tables else "azure",
            "activation_source": resolved_activation_source,
        },
    )
    if not local_tables:
        from ssep.analysis.paper.azure import load_remote_run_objects

        prompts, labels, metadata, table_provenance = load_remote_run_objects(
            registry, run_alias, store=azure
        )
        remote_inputs.extend(table_provenance)
        frame, metadata = validate_run_objects(registry, run_alias, prompts, labels, metadata)
    else:
        frame, metadata = load_run_frame(registry, run_alias)
    if resolved_activation_source == "azure":
        from ssep.analysis.paper.azure import remote_head, run_object_keys

        remote_inputs.append(
            remote_head(azure, run_object_keys(registry, run_alias)["resid_post_metadata"])
        )
    if remote_inputs:
        manifest["remote_inputs"] = remote_inputs
    atomic_json(output_dir / "manifest.pre_materialization.json", manifest)

    frame = _drop_prompt_exemplars(frame, metadata, dataset=spec.dataset)
    token_indices = position_indices(frame, position, prompt_regime=spec.prompt_regime)
    tensor_indices = frame.tensor_index.to_numpy(dtype=np.int64)
    import zarr

    from ssep.storage.dtypes import decode_bf16_np

    if resolved_activation_source == "local":
        array = zarr.open_array(
            store=str(registry.acts_dir(run_alias)), path="resid_post", mode="r"
        )
    else:
        from ssep.analysis.paper.azure import open_remote_array

        array = open_remote_array(registry, run_alias, "resid_post", store=azure)
    resid_index = layer + 1
    if resid_index < 0 or resid_index >= array.shape[1]:
        raise ValueError(f"layer {layer} maps outside resid_post axis of size {array.shape[1]}")
    if tensor_indices.min() < 0 or tensor_indices.max() >= array.shape[0]:
        raise ValueError("tensor_index maps outside resid_post prompt axis")
    if token_indices.max() >= array.shape[2]:
        raise ValueError("selected token position maps outside resid_post token axis")

    raw_path = output_dir / "dense_raw.npy"
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.float32, shape=(len(frame), array.shape[-1])
    )
    shard_size = int(array.chunks[0]) if array.chunks else 1024
    for start in range(0, array.shape[0], shard_size):
        mask = (tensor_indices >= start) & (tensor_indices < start + shard_size)
        if not mask.any():
            continue
        stop = min(start + shard_size, array.shape[0])
        block = array[start:stop, resid_index]
        raw[mask] = decode_bf16_np(block[tensor_indices[mask] - start, token_indices[mask]]).astype(
            np.float32
        )
    raw.flush()

    train = frame.split.to_numpy() == "train"
    if normalization == "none":
        model_input = raw
        rms_reference = None
        model_input_path = raw_path
    elif normalization == "rms":
        rms = np.empty(len(frame), dtype=np.float64)
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            values = np.asarray(raw[start:stop], dtype=np.float64)
            rms[start:stop] = np.sqrt(np.mean(values**2, axis=1))
        rms_reference = float(rms[train].mean())
        model_input_path = output_dir / "dense_rms.npy"
        model_input = np.lib.format.open_memmap(
            model_input_path, mode="w+", dtype=np.float32, shape=raw.shape
        )
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            scale = rms_reference / np.maximum(rms[start:stop], 1e-8)
            model_input[start:stop] = raw[start:stop] * scale[:, None]
        model_input.flush()
    else:
        raise ValueError("normalization must be none or rms")

    sae_summary = None
    if encode_sae:
        import torch

        from ssep.saes import SaeLensAdapter, validate_sae

        resolved_device = _device(device)
        suffix = "-canonical" if sae_l0 == "canonical" else ""
        release = f"gemma-scope-9b-pt-res{suffix}"
        sae_id = f"layer_{layer}/width_{sae_width}/{sae_l0}"
        sae = SaeLensAdapter(release, sae_id, device=resolved_device)
        if sae.d_in != model_input.shape[1]:
            raise ValueError(f"SAE input width {sae.d_in} != residual width {model_input.shape[1]}")
        codes = np.lib.format.open_memmap(
            output_dir / "sae_codes.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(frame), sae.d_sae),
        )
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            values = torch.from_numpy(np.ascontiguousarray(model_input[start:stop])).to(
                resolved_device
            )
            codes[start:stop] = sae.encode(values).float().cpu().numpy()
        codes.flush()
        validation_rows = np.flatnonzero(train)[: min(256, int(train.sum()))]
        validation = validate_sae(
            sae,
            torch.from_numpy(np.ascontiguousarray(model_input[validation_rows])).to(
                resolved_device
            ),
        )
        sae_summary = {**sae.metadata, **asdict(validation)}

    pd.DataFrame(
        {
            "prompt_id": frame.prompt_id,
            "source_id": frame.source_id,
            "split": frame.split,
            "tensor_index": tensor_indices,
            "token_index": token_indices,
        }
    ).to_parquet(output_dir / "rows.parquet", index=False)
    summary = {
        "run_alias": run_alias,
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "rows": len(frame),
        "split_counts": frame.split.value_counts().sort_index().to_dict(),
        "layer": layer,
        "resid_post_index": resid_index,
        "position": position,
        "normalization": normalization,
        "table_source": "local" if local_tables else "azure",
        "activation_source": resolved_activation_source,
        "rms_reference_train": rms_reference,
        "dense_raw_path": str(raw_path),
        "dense_model_input_path": str(model_input_path),
        "sae": sae_summary,
        "derived_channels": metadata.get("_paper_derived_channels", {}),
    }
    atomic_json(output_dir / "summary.json", summary)
    completed = dict(manifest)
    completed["status"] = "materialized"
    completed["outputs"] = summary
    atomic_json(output_dir / "manifest.json", completed)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--position",
        choices=("tbg", "answer_first", "content_slt", "stored_slt", "eoa", "generated_last"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization", choices=("none", "rms"), default="rms")
    parser.add_argument("--encode-sae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sae-width", default="16k")
    parser.add_argument("--sae-l0", default="canonical")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--activation-source",
        choices=("auto", "local", "azure"),
        default="auto",
        help="auto prefers a complete local resid_post array, otherwise streams Azure",
    )
    args = parser.parse_args(argv)
    print(
        materialize_run_site(
            config_path=Path(args.config),
            run_alias=args.run,
            layer=args.layer,
            position=args.position,
            output_dir=args.output,
            normalization=args.normalization,
            encode_sae=args.encode_sae,
            sae_width=args.sae_width,
            sae_l0=args.sae_l0,
            batch_size=args.batch_size,
            device=args.device,
            activation_source=args.activation_source,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
