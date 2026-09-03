"""Leakage-safe SAE feature selection and top-k benchmarking.

The supervised selectors operate on a discovery split only. The benchmark then
fits probes on a train sub-split, chooses ``(method, k)`` on an inner validation
split, refits on all train rows, and evaluates only that frozen winner on test.
This keeps method and k selection away from test labels.

Continuous targets such as ``se_lnrao`` can use correlation, mutual information,
or elastic-net screening directly. Binary screeners (Mann--Whitney, mean
difference, and L1 logistic) use only the bottom/top target tails when
``target_kind="continuous"``.

Contrastive DLA remains a separate, per-prompt correctness option because it
requires a discrete wrong/correct token pair and has no natural semantic-entropy
target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

SelectorMethod = Literal[
    "mann_whitney",
    "mean_difference",
    "l1_logistic",
    "spearman",
    "pearson",
    "mutual_information",
    "elastic_net",
]
TargetKind = Literal["binary", "continuous"]


@dataclass(frozen=True)
class FeatureSelectionResult:
    feature_ids: np.ndarray
    scores: np.ndarray
    method: str
    details: dict[str, object] | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    method: str
    k: int
    validation_metrics: dict[str, float]


@dataclass(frozen=True)
class BootstrapStability:
    iterations: int
    mean_jaccard: float
    feature_frequencies: np.ndarray


@dataclass(frozen=True)
class FeatureSelectionBenchmark:
    """Result of a validation-selected benchmark.

    Candidate rows intentionally contain validation metrics only. ``test_metrics``
    belongs only to the frozen winner, preventing accidental test-set selection.
    ``condition`` can record the layer/token/SAE-width configuration supplied by
    the caller.
    """

    candidates: tuple[CandidateEvaluation, ...]
    winner: CandidateEvaluation
    selection: FeatureSelectionResult
    test_metrics: dict[str, float]
    primary_metric: str
    condition: str | None
    stability: BootstrapStability | None
    final_probe: object


@dataclass(frozen=True)
class FeatureActivationSummary:
    feature_id: int
    selection_score: float
    active_fraction: float
    mean_activation: float
    top_example_indices: np.ndarray
    top_activations: np.ndarray


def _validate_matrix_target(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    require_binary: bool,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_values)
    y = np.asarray(targets)
    if z.ndim != 2:
        raise ValueError(f"activations must have shape (examples, features), got {z.shape}")
    if y.ndim != 1 or len(y) != len(z):
        raise ValueError(f"targets must have shape ({len(z)},), got {y.shape}")
    if len(z) < 2:
        raise ValueError("at least two examples are required")
    if not np.issubdtype(z.dtype, np.number) or not np.isfinite(z).all():
        raise ValueError("activations must contain only finite numeric values")
    if not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
        raise ValueError("targets must contain only finite numeric values")
    if require_binary and set(np.unique(y)) != {0, 1}:
        raise ValueError("targets must contain both binary classes 0 and 1")
    if not 1 <= k <= z.shape[1]:
        raise ValueError(f"k must be in [1, {z.shape[1]}], got {k}")
    return z, y.astype(np.int8, copy=False) if require_binary else y


def _validate_binary_inputs(
    z_values: np.ndarray, targets: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    return _validate_matrix_target(z_values, targets, k, require_binary=True)


def tail_binary_target(
    targets: np.ndarray, *, tail_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Return a mask and {0,1} labels for the bottom/top continuous-target tails."""
    y = np.asarray(targets)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
        raise ValueError("targets must be a finite numeric vector")
    if not 0 < tail_fraction <= 0.5:
        raise ValueError(f"tail_fraction must be in (0, 0.5], got {tail_fraction}")
    low, high = np.quantile(y, [tail_fraction, 1.0 - tail_fraction])
    if low >= high:
        raise ValueError("continuous target has no distinct bottom and top tails")
    mask = (y <= low) | (y >= high)
    labels = (y[mask] >= high).astype(np.int8)
    return mask, labels


def select_mean_difference(
    z_values: np.ndarray, targets: np.ndarray, k: int
) -> FeatureSelectionResult:
    """Rank SAE latents by absolute class-conditional mean difference."""
    z, y = _validate_binary_inputs(z_values, targets, k)
    signed = z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0)
    scores = np.abs(signed)
    feature_ids = np.argsort(-scores, kind="stable")[:k]
    return FeatureSelectionResult(
        feature_ids, scores, "mean_difference", {"signed_mean_difference": signed}
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(adjusted)
    q[order] = np.minimum(adjusted, 1.0)
    return q


def _cohens_d(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    positive, negative = z[y == 1], z[y == 0]
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("Mann--Whitney/effect size requires at least two rows per class")
    difference = positive.mean(axis=0) - negative.mean(axis=0)
    pooled_variance = (
        (len(positive) - 1) * positive.var(axis=0, ddof=1)
        + (len(negative) - 1) * negative.var(axis=0, ddof=1)
    ) / (len(positive) + len(negative) - 2)
    pooled_std = np.sqrt(np.maximum(pooled_variance, 0.0))
    return np.divide(
        difference,
        pooled_std,
        out=np.zeros_like(difference, dtype=np.float64),
        where=pooled_std > 0,
    )


def select_mann_whitney(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    fdr: float = 0.05,
) -> FeatureSelectionResult:
    """Mann--Whitney screening, FDR correction, then absolute Cohen's-d ranking.

    FDR-significant features rank before nonsignificant features; each group is
    ordered by absolute effect size. Raw p-values, BH q-values, signed Cohen's d,
    and the significance mask are retained in ``details``.
    """
    from scipy.stats import mannwhitneyu

    z, y = _validate_binary_inputs(z_values, targets, k)
    if not 0 < fdr <= 1:
        raise ValueError(f"fdr must be in (0, 1], got {fdr}")
    result = mannwhitneyu(z[y == 1], z[y == 0], axis=0, alternative="two-sided")
    p_values = np.nan_to_num(np.asarray(result.pvalue), nan=1.0)
    q_values = _benjamini_hochberg(p_values)
    effect_sizes = _cohens_d(z, y)
    scores = np.abs(effect_sizes)
    significant = q_values <= fdr
    feature_axis = np.arange(z.shape[1])
    ranking = np.lexsort((feature_axis, -scores, ~significant))
    return FeatureSelectionResult(
        ranking[:k],
        scores,
        "mann_whitney",
        {
            "p_values": p_values,
            "q_values": q_values,
            "cohens_d": effect_sizes,
            "significant": significant,
            "fdr": fdr,
        },
    )


def select_l1_logistic(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    c: float = 1.0,
    max_iter: int = 5_000,
    random_state: int = 0,
) -> FeatureSelectionResult:
    """Fit an L1 logistic model and rank latents by absolute coefficient."""
    from sklearn.linear_model import LogisticRegression

    z, y = _validate_binary_inputs(z_values, targets, k)
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=c,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(z, y)
    signed = model.coef_[0]
    scores = np.abs(signed)
    feature_ids = np.argsort(-scores, kind="stable")[:k]
    return FeatureSelectionResult(
        feature_ids, scores, "l1_logistic", {"signed_coefficients": signed, "c": c}
    )


def _pearson_scores(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_y = np.asarray(y, dtype=np.float64) - np.mean(y)
    y_norm = np.linalg.norm(centered_y)
    if y_norm == 0:
        raise ValueError("targets must not be constant")
    centered_z = z - z.mean(axis=0)
    denominators = np.linalg.norm(centered_z, axis=0) * y_norm
    return np.divide(
        centered_z.T @ centered_y,
        denominators,
        out=np.zeros(z.shape[1], dtype=np.float64),
        where=denominators > 0,
    )


def select_correlation(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    correlation: Literal["spearman", "pearson"] = "spearman",
    batch_features: int = 2_048,
) -> FeatureSelectionResult:
    """Rank by absolute feature/target correlation.

    Spearman ranks feature columns in batches to avoid an extra full-width rank
    matrix for large SAE dictionaries.
    """
    from scipy.stats import rankdata

    z, y = _validate_matrix_target(z_values, targets, k, require_binary=False)
    if correlation not in ("spearman", "pearson"):
        raise ValueError(f"unknown correlation: {correlation!r}")
    if batch_features < 1:
        raise ValueError("batch_features must be positive")
    if correlation == "pearson":
        signed = _pearson_scores(z, y)
    else:
        ranked_y = rankdata(y)
        signed = np.empty(z.shape[1], dtype=np.float64)
        for start in range(0, z.shape[1], batch_features):
            stop = min(start + batch_features, z.shape[1])
            signed[start:stop] = _pearson_scores(rankdata(z[:, start:stop], axis=0), ranked_y)
    scores = np.abs(signed)
    feature_ids = np.argsort(-scores, kind="stable")[:k]
    return FeatureSelectionResult(feature_ids, scores, correlation, {"signed_correlations": signed})


def select_elastic_net(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    alpha: float = 0.01,
    l1_ratio: float = 1.0,
    max_iter: int = 5_000,
    random_state: int = 0,
) -> FeatureSelectionResult:
    """Rank continuous-target features by absolute elastic-net coefficient.

    ``l1_ratio=1`` gives Lasso. Activations are intentionally left on their
    native SAE scale; choose regularization using validation data, never test.
    """
    from sklearn.linear_model import ElasticNet

    z, y = _validate_matrix_target(z_values, targets, k, require_binary=False)
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not 0 < l1_ratio <= 1:
        raise ValueError(f"l1_ratio must be in (0, 1], got {l1_ratio}")
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(z, y)
    signed = model.coef_
    scores = np.abs(signed)
    feature_ids = np.argsort(-scores, kind="stable")[:k]
    return FeatureSelectionResult(
        feature_ids,
        scores,
        "elastic_net",
        {"signed_coefficients": signed, "alpha": alpha, "l1_ratio": l1_ratio},
    )


def select_mutual_information(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    target_kind: TargetKind,
    random_state: int = 0,
) -> FeatureSelectionResult:
    """Rank features by estimated mutual information with the target.

    This nonparametric selector can detect nonlinear, non-monotonic dependence
    that Pearson/Spearman screening misses. The sklearn k-nearest-neighbor
    estimator is stochastic only when it must break ties in continuous values,
    so ``random_state`` makes rankings reproducible.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if target_kind not in ("binary", "continuous"):
        raise ValueError(f"unknown target_kind: {target_kind!r}")
    z, y = _validate_matrix_target(z_values, targets, k, require_binary=target_kind == "binary")
    estimator = mutual_info_classif if target_kind == "binary" else mutual_info_regression
    scores = np.asarray(
        estimator(z, y, discrete_features=False, random_state=random_state),
        dtype=np.float64,
    )
    feature_ids = np.argsort(-scores, kind="stable")[:k]
    return FeatureSelectionResult(
        feature_ids,
        scores,
        "mutual_information",
        {"estimator": "knn", "random_state": random_state},
    )


def select_contrastive_dla(
    feature_activations: np.ndarray,
    decoder_directions: np.ndarray,
    unembedding: np.ndarray,
    wrong_token_id: int,
    correct_token_id: int,
    k: int = 100,
) -> FeatureSelectionResult:
    """Rank one prompt's features by contrastive direct logit attribution.

    ``C_i = s_i * w_dec[i] @ (W_U[:, wrong] - W_U[:, correct])``. This is
    deliberately excluded from the semantic-entropy benchmark because it needs
    one discrete hallucinated/factual token pair per prompt.
    """
    activations = np.asarray(feature_activations)
    decoder = np.asarray(decoder_directions)
    w_u = np.asarray(unembedding)
    if activations.ndim != 1:
        raise ValueError(
            f"feature_activations must have shape (features,), got {activations.shape}"
        )
    if decoder.ndim != 2 or decoder.shape[0] != len(activations):
        raise ValueError(
            f"decoder_directions must have shape ({len(activations)}, d_model), got {decoder.shape}"
        )
    if w_u.ndim != 2 or w_u.shape[0] != decoder.shape[1]:
        raise ValueError(
            f"unembedding must have shape ({decoder.shape[1]}, vocabulary), got {w_u.shape}"
        )
    for name, value in (
        ("feature_activations", activations),
        ("decoder_directions", decoder),
        ("unembedding", w_u),
    ):
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite numeric values")
    vocabulary_size = w_u.shape[1]
    for name, token_id in (
        ("wrong_token_id", wrong_token_id),
        ("correct_token_id", correct_token_id),
    ):
        if not isinstance(token_id, int | np.integer) or not 0 <= token_id < vocabulary_size:
            raise ValueError(
                f"{name} must be an integer in [0, {vocabulary_size - 1}], got {token_id!r}"
            )
    if not 1 <= k <= len(activations):
        raise ValueError(f"k must be in [1, {len(activations)}], got {k}")
    direction = w_u[:, wrong_token_id] - w_u[:, correct_token_id]
    signed = activations * (decoder @ direction)
    scores = np.abs(signed)
    feature_ids = np.argsort(-scores, kind="stable")[:k]
    return FeatureSelectionResult(
        feature_ids, scores, "contrastive_dla", {"signed_attributions": signed}
    )


def select_sae_features(
    z_values: np.ndarray,
    targets: np.ndarray,
    k: int,
    *,
    method: SelectorMethod = "mean_difference",
    target_kind: TargetKind = "binary",
    tail_fraction: float = 0.25,
    mann_whitney_fdr: float = 0.05,
    l1_c: float = 1.0,
    elastic_alpha: float = 0.01,
    elastic_l1_ratio: float = 1.0,
    correlation_batch_features: int = 2_048,
    random_state: int = 0,
) -> FeatureSelectionResult:
    """Common entry point for supervised selectors.

    With a continuous target, binary methods screen only bottom/top tails. The
    returned score arrays always span the original feature dimension.
    """
    if target_kind not in ("binary", "continuous"):
        raise ValueError(f"unknown target_kind: {target_kind!r}")
    binary_methods = {"mann_whitney", "mean_difference", "l1_logistic"}
    z, y = np.asarray(z_values), np.asarray(targets)
    if method in binary_methods and target_kind == "continuous":
        mask, tail_targets = tail_binary_target(y, tail_fraction=tail_fraction)
        z, y = z[mask], tail_targets
    if method == "mann_whitney":
        return select_mann_whitney(z, y, k, fdr=mann_whitney_fdr)
    if method == "mean_difference":
        return select_mean_difference(z, y, k)
    if method == "l1_logistic":
        return select_l1_logistic(z, y, k, c=l1_c, random_state=random_state)
    if method in ("spearman", "pearson"):
        return select_correlation(
            z, y, k, correlation=method, batch_features=correlation_batch_features
        )
    if method == "mutual_information":
        return select_mutual_information(
            z, y, k, target_kind=target_kind, random_state=random_state
        )
    if method == "elastic_net":
        return select_elastic_net(
            z,
            y,
            k,
            alpha=elastic_alpha,
            l1_ratio=elastic_l1_ratio,
            random_state=random_state,
        )
    raise ValueError(f"unknown feature-selection method: {method!r}")


def _probe_metrics(
    target_kind: TargetKind, y: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    if target_kind == "binary":
        from sklearn.metrics import accuracy_score, roc_auc_score

        return {
            "roc_auc": float(roc_auc_score(y, predictions)),
            "accuracy": float(accuracy_score(y, predictions >= 0.5)),
        }
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    correlation = float(spearmanr(y, predictions).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "r2": float(r2_score(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "spearman": correlation,
    }


def _fit_probe(
    z: np.ndarray,
    y: np.ndarray,
    target_kind: TargetKind,
    *,
    random_state: int,
    ridge_alpha: float,
) -> object:
    if target_kind == "binary":
        from sklearn.linear_model import LogisticRegression

        probe = LogisticRegression(max_iter=5_000, random_state=random_state)
    else:
        from sklearn.linear_model import Ridge

        probe = Ridge(alpha=ridge_alpha)
    probe.fit(z, y)
    return probe


def _probe_predictions(probe: object, z: np.ndarray, target_kind: TargetKind) -> np.ndarray:
    if target_kind == "binary":
        return probe.predict_proba(z)[:, 1]  # type: ignore[attr-defined, no-any-return]
    return probe.predict(z)  # type: ignore[attr-defined, no-any-return]


def _bootstrap_stability(
    z: np.ndarray,
    y: np.ndarray,
    *,
    original: FeatureSelectionResult,
    k: int,
    method: SelectorMethod,
    target_kind: TargetKind,
    iterations: int,
    random_state: int,
    selector_kwargs: dict[str, object],
) -> BootstrapStability | None:
    if iterations <= 0:
        return None
    rng = np.random.default_rng(random_state)
    original_set = set(original.feature_ids.tolist())
    frequencies = np.zeros(z.shape[1], dtype=np.float64)
    jaccards = []
    class_indices = [np.flatnonzero(y == value) for value in (0, 1)]
    for iteration in range(iterations):
        if target_kind == "binary":
            sampled = np.concatenate(
                [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
            )
            rng.shuffle(sampled)
        else:
            sampled = rng.choice(len(z), len(z), replace=True)
        result = select_sae_features(
            z[sampled],
            y[sampled],
            k,
            method=method,
            target_kind=target_kind,
            random_state=random_state + iteration + 1,
            **selector_kwargs,
        )
        selected = set(result.feature_ids.tolist())
        frequencies[result.feature_ids] += 1
        jaccards.append(len(original_set & selected) / len(original_set | selected))
    frequencies /= iterations
    return BootstrapStability(iterations, float(np.mean(jaccards)), frequencies)


def benchmark_sae_feature_selectors(
    z_discovery: np.ndarray,
    y_discovery: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    target_kind: TargetKind,
    methods: Sequence[SelectorMethod] | None = None,
    k_values: Sequence[int] = (32, 64, 128, 256, 512, 1_024),
    validation_fraction: float = 0.2,
    random_state: int = 0,
    bootstrap_iterations: int = 20,
    condition: str | None = None,
    ridge_alpha: float = 1.0,
    tail_fraction: float = 0.25,
    mann_whitney_fdr: float = 0.05,
    l1_c: float = 1.0,
    elastic_alpha: float = 0.01,
    elastic_l1_ratio: float = 1.0,
    correlation_batch_features: int = 2_048,
) -> FeatureSelectionBenchmark:
    """Select a feature method and k without consulting test performance.

    Feature rankings are fit once on ``discovery``. Candidate probes use an
    inner train/validation split. Only the validation winner is refit on all
    train rows and evaluated on test. Run SE and correctness as separate calls;
    record layer/token/width in ``condition`` and compare conditions by their
    validation metric before inspecting their test results.
    """
    from sklearn.model_selection import train_test_split

    if target_kind not in ("binary", "continuous"):
        raise ValueError(f"unknown target_kind: {target_kind!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    arrays = [np.asarray(v) for v in (z_discovery, z_train, z_test)]
    targets = [np.asarray(v) for v in (y_discovery, y_train, y_test)]
    feature_counts = {z.shape[1] for z in arrays if z.ndim == 2}
    if len(feature_counts) != 1 or any(z.ndim != 2 for z in arrays):
        raise ValueError("all activation splits must be 2D with the same feature dimension")
    n_features = feature_counts.pop()
    for z, y in zip(arrays, targets, strict=True):
        _validate_matrix_target(z, y, min(1, n_features), require_binary=target_kind == "binary")
    ks = tuple(sorted(set(int(k) for k in k_values)))
    if not ks or ks[0] < 1 or ks[-1] > n_features:
        raise ValueError(f"k_values must be within [1, {n_features}], got {ks}")
    if methods is None:
        methods = (
            ("mann_whitney", "mean_difference", "l1_logistic", "mutual_information")
            if target_kind == "binary"
            else (
                "mann_whitney",
                "mean_difference",
                "spearman",
                "pearson",
                "mutual_information",
                "elastic_net",
            )
        )
    if not methods:
        raise ValueError("at least one selector method is required")

    selector_kwargs: dict[str, object] = {
        "tail_fraction": tail_fraction,
        "mann_whitney_fdr": mann_whitney_fdr,
        "l1_c": l1_c,
        "elastic_alpha": elastic_alpha,
        "elastic_l1_ratio": elastic_l1_ratio,
        "correlation_batch_features": correlation_batch_features,
    }
    rankings = {
        method: select_sae_features(
            arrays[0],
            targets[0],
            ks[-1],
            method=method,
            target_kind=target_kind,
            random_state=random_state,
            **selector_kwargs,
        )
        for method in methods
    }
    indices = np.arange(len(arrays[1]))
    stratify = targets[1] if target_kind == "binary" else None
    fit_idx, validation_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=stratify,
    )
    candidates = []
    primary_metric = "roc_auc" if target_kind == "binary" else "r2"
    for method in methods:
        ranking = rankings[method]
        for k in ks:
            feature_ids = ranking.feature_ids[:k]
            probe = _fit_probe(
                arrays[1][fit_idx][:, feature_ids],
                targets[1][fit_idx],
                target_kind,
                random_state=random_state,
                ridge_alpha=ridge_alpha,
            )
            predictions = _probe_predictions(
                probe, arrays[1][validation_idx][:, feature_ids], target_kind
            )
            metrics = _probe_metrics(target_kind, targets[1][validation_idx], predictions)
            candidates.append(CandidateEvaluation(method, k, metrics))
    winner = max(
        candidates,
        key=lambda candidate: (candidate.validation_metrics[primary_metric], -candidate.k),
    )
    ranking = rankings[winner.method]
    selected = FeatureSelectionResult(
        ranking.feature_ids[: winner.k], ranking.scores, ranking.method, ranking.details
    )
    final_probe = _fit_probe(
        arrays[1][:, selected.feature_ids],
        targets[1],
        target_kind,
        random_state=random_state,
        ridge_alpha=ridge_alpha,
    )
    test_predictions = _probe_predictions(
        final_probe, arrays[2][:, selected.feature_ids], target_kind
    )
    test_metrics = _probe_metrics(target_kind, targets[2], test_predictions)
    stability = _bootstrap_stability(
        arrays[0],
        targets[0],
        original=selected,
        k=winner.k,
        method=winner.method,  # type: ignore[arg-type]
        target_kind=target_kind,
        iterations=bootstrap_iterations,
        random_state=random_state,
        selector_kwargs=selector_kwargs,
    )
    return FeatureSelectionBenchmark(
        tuple(candidates),
        winner,
        selected,
        test_metrics,
        primary_metric,
        condition,
        stability,
        final_probe,
    )


def summarize_selected_features(
    activations: np.ndarray,
    selection: FeatureSelectionResult,
    *,
    top_examples: int = 10,
) -> tuple[FeatureActivationSummary, ...]:
    """Return top-activating row indices for qualitative context inspection.

    Map ``top_example_indices`` back to prompt text outside this module. These
    summaries support interpretation; they do not establish causality, which
    still requires ablation or steering.
    """
    z = np.asarray(activations)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("activations must be a finite (examples, features) matrix")
    if not 1 <= top_examples <= len(z):
        raise ValueError(f"top_examples must be in [1, {len(z)}]")
    if np.any(selection.feature_ids < 0) or np.any(selection.feature_ids >= z.shape[1]):
        raise ValueError("selection contains feature ids outside the activation matrix")
    summaries = []
    for feature_id in selection.feature_ids:
        values = z[:, feature_id]
        top = np.argsort(-values, kind="stable")[:top_examples]
        summaries.append(
            FeatureActivationSummary(
                int(feature_id),
                float(selection.scores[feature_id]),
                float(np.mean(values != 0)),
                float(np.mean(values)),
                top,
                values[top],
            )
        )
    return tuple(summaries)
