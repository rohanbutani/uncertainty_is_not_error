"""Canonical prompt table and token-position contract across paper runs."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ssep.analysis.paper.registry import PaperRegistry
from ssep.analysis.paper.splits import validate_split_groups

REQUIRED_PROMPT_COLUMNS = {
    "prompt_id",
    "source_id",
    "split",
    "greedy_entropy_full",
    "greedy_sliced_token_count",
}
REQUIRED_LABEL_COLUMNS = {"prompt_id", "se_discrete", "f1_squad"}


def _unique_complete(frame: pd.DataFrame, column: str, table: str) -> None:
    if column not in frame:
        raise ValueError(f"{table} is missing {column}")
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"{table}.{column} must be complete and unique")


def _normalize_squad_answer(value: str) -> str:
    """The official SQuAD v1/v2 lowercase/punctuation/article normalization."""
    value = str(value).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _squad_f1(prediction: str, gold_answers: Any) -> float:
    prediction_tokens = _normalize_squad_answer(prediction).split()
    best = 0.0
    for gold in list(gold_answers):
        gold_tokens = _normalize_squad_answer(gold).split()
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if not prediction_tokens or not gold_tokens:
            score = float(prediction_tokens == gold_tokens)
        elif overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return 100.0 * best


def derive_string_correctness(prompts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Restore absent SQuAD correctness columns from banked text and gold answers.

    Older NLI-only label runs did not persist the cheap string channel. This uses
    the same official normalization and max-over-gold definition as the pinned
    ``evaluate`` SQuAD metric, while avoiding one metric invocation per answer.
    Existing stored columns always take precedence.
    """
    if "f1_squad" in labels and "sample_f1" in labels:
        return labels
    required = {"prompt_id", "gold_answers", "greedy_text_clean", "sample_texts_clean"}
    missing = required - set(prompts)
    if missing:
        return labels
    by_prompt = prompts.set_index("prompt_id", drop=False)
    result = labels.copy()
    prompt_rows = by_prompt.loc[result.prompt_id]
    if "f1_squad" not in result:
        result["f1_squad"] = [
            _squad_f1(prediction, gold)
            for prediction, gold in zip(
                prompt_rows.greedy_text_clean, prompt_rows.gold_answers, strict=True
            )
        ]
    if "sample_f1" not in result:
        result["sample_f1"] = [
            [_squad_f1(prediction, gold) for prediction in samples]
            for samples, gold in zip(
                prompt_rows.sample_texts_clean, prompt_rows.gold_answers, strict=True
            )
        ]
    return result


def load_run_frame(registry: PaperRegistry, alias: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, join, and validate one run without reordering prompt rows."""
    tables = registry.tables_dir(alias)
    metadata_path = registry.metadata_path(alias)
    prompts_path, labels_path = tables / "prompts.parquet", tables / "labels.parquet"
    missing = [
        str(path) for path in (prompts_path, labels_path, metadata_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"run {alias!r} is missing local artifacts: {missing}")
    prompts = pd.read_parquet(prompts_path)
    labels = pd.read_parquet(labels_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return validate_run_objects(registry, alias, prompts, labels, metadata)


def validate_run_objects(
    registry: PaperRegistry,
    alias: str,
    prompts: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join and validate run objects regardless of their storage backend."""
    _unique_complete(prompts, "prompt_id", "prompts")
    _unique_complete(labels, "prompt_id", "labels")
    stored_label_columns = set(labels)
    labels = derive_string_correctness(prompts, labels)
    missing_prompts = REQUIRED_PROMPT_COLUMNS - set(prompts)
    missing_labels = REQUIRED_LABEL_COLUMNS - set(labels)
    if missing_prompts or missing_labels:
        raise ValueError(
            f"run {alias!r} missing columns prompts={sorted(missing_prompts)}, "
            f"labels={sorted(missing_labels)}"
        )
    label_columns = [column for column in labels if column not in prompts or column == "prompt_id"]
    frame = prompts.merge(labels[label_columns], on="prompt_id", how="left", validate="one_to_one")
    if len(frame) != len(prompts) or frame.prompt_id.tolist() != prompts.prompt_id.tolist():
        raise AssertionError("prompt/label join changed row order")
    if frame[list(REQUIRED_LABEL_COLUMNS - {"prompt_id"})].isna().any().any():
        raise ValueError(f"run {alias!r} has incomplete primary labels")
    validate_split_groups(frame.split, frame.source_id)
    if metadata.get("run_id") != registry.run(alias).run_id:
        raise ValueError(f"run {alias!r} metadata run_id disagrees with registry")
    if "f1_squad" not in stored_label_columns and "f1_squad" in labels:
        metadata = dict(metadata)
        metadata["_paper_derived_channels"] = {
            "f1_squad": "official_squad_normalization_max_over_gold",
            "sample_f1": "official_squad_normalization_max_over_gold",
        }
    return frame, metadata


def content_slt_indices(frame: pd.DataFrame, *, allow_stored_fallback: bool = False) -> np.ndarray:
    """Last sliced answer token; reports must flag known trailing-newline runs.

    If a future table stores an explicit content-token count, it takes
    precedence.  The current long-arm tables need that column (or a tokenizer
    replay) to distinguish content SLT from the schema-faithful newline SLT.
    """
    if "greedy_content_token_count" in frame:
        column = "greedy_content_token_count"
    elif allow_stored_fallback:
        column = "greedy_sliced_token_count"
    else:
        raise ValueError(
            "content-SLT needs greedy_content_token_count; use stored_slt explicitly "
            "if the schema-faithful sliced endpoint is intended"
        )
    counts = frame[column].to_numpy(dtype=np.int64)
    if (counts < 1).any():
        raise ValueError("every analyzed answer must have at least one content token")
    answer_start = (
        frame.answer_start_token_rel.to_numpy(dtype=np.int64)
        if "answer_start_token_rel" in frame
        else np.zeros(len(frame), dtype=np.int64)
    )
    return answer_start + counts - 1


def position_indices(frame: pd.DataFrame, name: str, *, prompt_regime: str) -> np.ndarray:
    """Resolve a named paper position without silently conflating SLT variants."""
    answer_start = frame.answer_start_token_rel.to_numpy(dtype=np.int64)
    if name == "tbg":
        result = frame.tbg_index_rel.to_numpy(dtype=np.int64)
    elif name == "answer_first":
        result = answer_start
    elif name == "content_slt":
        result = content_slt_indices(
            frame,
            allow_stored_fallback=prompt_regime == "brief",
        )
    elif name == "stored_slt":
        result = answer_start + frame.greedy_sliced_token_count.to_numpy(dtype=np.int64) - 1
    elif name == "eoa":
        result = answer_start + frame.greedy_sliced_token_count.to_numpy(dtype=np.int64)
    elif name == "generated_last":
        result = answer_start + frame.greedy_token_count.to_numpy(dtype=np.int64) - 1
    else:
        raise ValueError(f"unknown position {name!r}")
    if (result < 0).any():
        raise ValueError(f"position {name!r} produced a negative token index")
    return result


def first_token_entropy(frame: pd.DataFrame) -> np.ndarray:
    values = np.asarray([float(np.asarray(row)[0]) for row in frame.greedy_entropy_full])
    if not np.isfinite(values).all() or (values < -1e-8).any():
        raise ValueError("first-token entropy must be finite and nonnegative")
    return np.maximum(values, 0.0)


def sample_correctness(
    frame: pd.DataFrame, *, fallback_f1_threshold: float = 50.0
) -> tuple[np.ndarray, str]:
    """Return (rows, samples) correctness and record the label channel used."""
    if "sample_judge_binary_labels" in frame and frame.sample_judge_binary_labels.notna().all():
        values = np.stack(frame.sample_judge_binary_labels.map(np.asarray))
        channel = "sample_judge_binary_labels"
    elif "sample_f1" in frame and frame.sample_f1.notna().all():
        values = np.stack(frame.sample_f1.map(np.asarray)) >= fallback_f1_threshold
        channel = f"sample_f1>={fallback_f1_threshold:g}"
    else:
        raise ValueError("neither complete sample judge labels nor sample_f1 is available")
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != len(frame):
        raise ValueError("sample correctness must have shape (prompts, samples)")
    if values.dtype != np.bool_:
        unique = set(np.unique(values).tolist())
        if not unique <= {0, 1}:
            raise ValueError("sample judge correctness must be binary")
        values = values.astype(bool)
    return values, channel
