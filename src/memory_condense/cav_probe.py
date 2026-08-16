"""Measure whether Qwen prefix residuals support stable live-memory concepts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from memory_condense.qwen_prefix import Qwen3PrefixEncoder


@dataclass(frozen=True, slots=True)
class ConceptDataset:
    name: str
    description: str
    train_positive: tuple[str, ...]
    train_negative: tuple[str, ...]
    test_positive: tuple[str, ...]
    test_negative: tuple[str, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return (
            self.train_positive
            + self.train_negative
            + self.test_positive
            + self.test_negative
        )


@dataclass(frozen=True, slots=True)
class FittedCAV:
    vector: Any
    threshold: float


def load_concept_datasets(path: str | Path) -> list[ConceptDataset]:
    """Load explicit train/test concept examples from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets: list[ConceptDataset] = []
    names: set[str] = set()
    for raw in payload.get("concepts", []):
        name = str(raw["name"])
        if name in names:
            raise ValueError(f"duplicate concept name: {name}")
        names.add(name)
        train = raw["train"]
        test = raw["test"]
        dataset = ConceptDataset(
            name=name,
            description=str(raw.get("description", "")),
            train_positive=tuple(train["positive"]),
            train_negative=tuple(train["negative"]),
            test_positive=tuple(test["positive"]),
            test_negative=tuple(test["negative"]),
        )
        for split_name, examples in (
            ("train positive", dataset.train_positive),
            ("train negative", dataset.train_negative),
            ("test positive", dataset.test_positive),
            ("test negative", dataset.test_negative),
        ):
            if len(examples) < 2:
                raise ValueError(
                    f"{dataset.name} needs at least two {split_name} examples"
                )
        datasets.append(dataset)
    if not datasets:
        raise ValueError("dataset contains no concepts")
    return datasets


def fit_mean_difference_cav(positive: Any, negative: Any) -> FittedCAV:
    """Fit an oriented concept direction and midpoint decision threshold."""
    _validate_matrix_pair(positive, negative)
    vector = positive.mean(dim=0) - negative.mean(dim=0)
    norm = vector.float().norm()
    if not math.isfinite(float(norm)) or float(norm) == 0.0:
        raise ValueError("positive and negative concept centroids are identical")
    vector = vector.float() / norm
    positive_mean = float((positive.float() @ vector).mean())
    negative_mean = float((negative.float() @ vector).mean())
    return FittedCAV(
        vector=vector,
        threshold=(positive_mean + negative_mean) / 2.0,
    )


def evaluate_cav(cav: FittedCAV, positive: Any, negative: Any) -> dict[str, float]:
    """Evaluate a CAV on explicitly held-out positive and negative examples."""
    _validate_matrix_pair(positive, negative)
    positive_scores = positive.float() @ cav.vector
    negative_scores = negative.float() @ cav.vector
    positive_accuracy = float((positive_scores >= cav.threshold).float().mean())
    negative_accuracy = float((negative_scores < cav.threshold).float().mean())
    effect = _cohens_d(positive_scores, negative_scores)
    return {
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "positive_score_mean": float(positive_scores.mean()),
        "negative_score_mean": float(negative_scores.mean()),
        "score_margin": float(positive_scores.mean() - negative_scores.mean()),
        "cohens_d": effect,
    }


def bootstrap_stability(
    positive: Any,
    negative: Any,
    *,
    repeats: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Measure cosine stability of resampled concept directions."""
    if repeats < 2:
        raise ValueError("bootstrap repeats must be at least two")
    base = fit_mean_difference_cav(positive, negative).vector
    import torch

    rng = torch.Generator(device="cpu").manual_seed(seed)
    cosines: list[float] = []
    positive_cpu = positive.float().cpu()
    negative_cpu = negative.float().cpu()
    for _ in range(repeats):
        positive_indices = torch.randint(
            len(positive_cpu), (len(positive_cpu),), generator=rng
        )
        negative_indices = torch.randint(
            len(negative_cpu), (len(negative_cpu),), generator=rng
        )
        sampled = fit_mean_difference_cav(
            positive_cpu[positive_indices], negative_cpu[negative_indices]
        ).vector
        cosines.append(float(sampled @ base.cpu()))
    values = torch.tensor(cosines)
    return {
        "mean_cosine": float(values.mean()),
        "minimum_cosine": float(values.min()),
        "std_cosine": float(values.std(unbiased=False)),
    }


def random_label_control(
    train_positive: Any,
    train_negative: Any,
    test_positive: Any,
    test_negative: Any,
    *,
    repeats: int = 32,
    seed: int = 0,
) -> dict[str, float]:
    """Fit permuted training labels and score against true held-out labels."""
    import torch

    if repeats < 1:
        raise ValueError("control repeats must be positive")
    combined = torch.cat([train_positive.float(), train_negative.float()], dim=0).cpu()
    positive_count = len(train_positive)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    accuracies: list[float] = []
    for _ in range(repeats):
        permutation = torch.randperm(len(combined), generator=rng)
        shuffled_positive = combined[permutation[:positive_count]]
        shuffled_negative = combined[permutation[positive_count:]]
        try:
            cav = fit_mean_difference_cav(shuffled_positive, shuffled_negative)
        except ValueError:
            # A perfectly symmetric permutation has no direction and therefore
            # no predictive information.  Record chance instead of making the
            # control depend on an arbitrary fallback vector.
            accuracies.append(0.5)
            continue
        metrics = evaluate_cav(cav, test_positive.cpu(), test_negative.cpu())
        accuracies.append(metrics["balanced_accuracy"])
    values = torch.tensor(accuracies)
    return {
        "mean_balanced_accuracy": float(values.mean()),
        "maximum_balanced_accuracy": float(values.max()),
        "std_balanced_accuracy": float(values.std(unbiased=False)),
    }


def _validate_matrix_pair(positive: Any, negative: Any) -> None:
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("concept activations must be rank-2 matrices")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative activations need equal dimensions")
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("both concept classes must contain examples")


def _cohens_d(positive_scores: Any, negative_scores: Any) -> float:
    positive_variance = positive_scores.float().var(unbiased=True)
    negative_variance = negative_scores.float().var(unbiased=True)
    pooled = ((positive_variance + negative_variance) / 2.0).sqrt()
    if float(pooled) == 0.0:
        return math.inf
    return float(
        (positive_scores.float().mean() - negative_scores.float().mean()) / pooled
    )


def _take(vectors: Any, text_to_index: dict[str, int], texts: Sequence[str]) -> Any:
    import torch

    indices = torch.tensor([text_to_index[text] for text in texts], dtype=torch.long)
    return vectors[indices]


def _parse_layers(raw: str, available: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(available))
    layers = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not layers:
        raise ValueError("no layers selected")
    invalid = [layer for layer in layers if not 0 <= layer < available]
    if invalid:
        raise ValueError(f"layers outside prefix [0, {available}): {invalid}")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--prefix-layers", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=64)
    parser.add_argument("--control-repeats", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    datasets = load_concept_datasets(args.dataset)
    layers = _parse_layers(args.layers, args.prefix_layers)

    unique_texts = list(
        dict.fromkeys(text for dataset in datasets for text in dataset.texts)
    )
    text_to_index = {text: index for index, text in enumerate(unique_texts)}

    encoder = Qwen3PrefixEncoder(
        args.model_dir,
        layers=args.prefix_layers,
        device=args.device,
        dtype=args.dtype,
    )
    activations = encoder.encode_layers(
        unique_texts,
        layers=layers,
        batch_size=args.batch_size,
    )

    vector_tensors: dict[str, Any] = {}
    concept_reports: list[dict[str, Any]] = []
    passing_by_concept: list[set[int]] = []
    for dataset in datasets:
        layer_reports: list[dict[str, Any]] = []
        passing_layers: set[int] = set()
        for layer in layers:
            vectors = activations[layer]
            train_positive = _take(vectors, text_to_index, dataset.train_positive)
            train_negative = _take(vectors, text_to_index, dataset.train_negative)
            test_positive = _take(vectors, text_to_index, dataset.test_positive)
            test_negative = _take(vectors, text_to_index, dataset.test_negative)

            cav = fit_mean_difference_cav(train_positive, train_negative)
            train_metrics = evaluate_cav(cav, train_positive, train_negative)
            test_metrics = evaluate_cav(cav, test_positive, test_negative)
            stability = bootstrap_stability(
                train_positive,
                train_negative,
                repeats=args.bootstrap_repeats,
                seed=args.seed + layer,
            )
            control = random_label_control(
                train_positive,
                train_negative,
                test_positive,
                test_negative,
                repeats=args.control_repeats,
                seed=args.seed + 1000 + layer,
            )
            passed = (
                test_metrics["balanced_accuracy"] >= 0.75
                and stability["mean_cosine"] >= 0.50
                and control["mean_balanced_accuracy"] <= 0.65
            )
            if passed:
                passing_layers.add(layer)
            vector_tensors[f"{dataset.name}.layer_{layer}"] = cav.vector.contiguous()
            layer_reports.append(
                {
                    "layer": layer,
                    "threshold": cav.threshold,
                    "train": train_metrics,
                    "test": test_metrics,
                    "bootstrap": stability,
                    "random_label_control": control,
                    "gate_passed": passed,
                }
            )

        best = max(
            layer_reports,
            key=lambda report: (
                report["test"]["balanced_accuracy"],
                report["bootstrap"]["mean_cosine"],
            ),
        )
        passing_by_concept.append(passing_layers)
        concept_reports.append(
            {
                "name": dataset.name,
                "description": dataset.description,
                "counts": {
                    "train_positive": len(dataset.train_positive),
                    "train_negative": len(dataset.train_negative),
                    "test_positive": len(dataset.test_positive),
                    "test_negative": len(dataset.test_negative),
                },
                "best_layer": best["layer"],
                "passing_layers": sorted(passing_layers),
                "layers": layer_reports,
            }
        )

    common_passing = set(layers)
    for passing in passing_by_concept:
        common_passing &= passing

    report = {
        "status": "measured",
        "model": "Qwen/Qwen3-8B",
        "prefix_layers": args.prefix_layers,
        "layers_tested": list(layers),
        "runtime_dtype": args.dtype,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "unique_examples": len(unique_texts),
        "seed": args.seed,
        "gate": {
            "heldout_balanced_accuracy_min": 0.75,
            "bootstrap_mean_cosine_min": 0.50,
            "random_label_mean_accuracy_max": 0.65,
            "common_passing_layers": sorted(common_passing),
            "passed": bool(common_passing),
        },
        "concepts": concept_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    from safetensors.torch import save_file

    vector_path = args.output.with_suffix(".safetensors")
    save_file(
        vector_tensors,
        vector_path,
        metadata={
            "model": "Qwen/Qwen3-8B",
            "dataset_sha256": report["dataset_sha256"],
            "method": "mean_difference",
        },
    )
    print(json.dumps(report["gate"], indent=2))
    for concept in concept_reports:
        best_layer = concept["best_layer"]
        best = next(item for item in concept["layers"] if item["layer"] == best_layer)
        print(
            f"{concept['name']}: best layer {best_layer}, "
            f"heldout={best['test']['balanced_accuracy']:.3f}, "
            f"stability={best['bootstrap']['mean_cosine']:.3f}, "
            f"control={best['random_label_control']['mean_balanced_accuracy']:.3f}"
        )
    print(f"report: {args.output}")
    print(f"vectors: {vector_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
