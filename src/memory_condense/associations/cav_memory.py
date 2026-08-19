"""Compact concept-vector indexing and artifact loading."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from memory_condense.associations.head_memory_models import CAVNeighbor


class CAVLinkIndex:
    """Compact Concept↔Episode index; stores no teacher token activations."""

    def __init__(self, concept_names: Sequence[str]) -> None:
        names = tuple(concept_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("concept_names must be non-empty and unique")
        self.concept_names = names
        self._signatures: dict[str, tuple[float, ...]] = {}
        self._active_to_episodes: dict[int, set[str]] = {
            index: set() for index in range(len(names))
        }
        self._coactivations: Counter[tuple[int, int]] = Counter()

    def add(self, episode_id: str, signature: Sequence[float]) -> None:
        if episode_id in self._signatures:
            raise ValueError(f"duplicate episode id: {episode_id}")
        values = tuple(float(value) for value in signature)
        if len(values) != len(self.concept_names):
            raise ValueError("signature width does not match concept_names")
        self._signatures[episode_id] = values
        active = [index for index, value in enumerate(values) if value > 0.0]
        for index in active:
            self._active_to_episodes[index].add(episode_id)
        for left_position, left in enumerate(active):
            for right in active[left_position + 1 :]:
                self._coactivations[(left, right)] += 1

    def remove(self, episode_ids: Sequence[str]) -> int:
        removed = 0
        for episode_id in set(episode_ids):
            signature = self._signatures.pop(episode_id, None)
            if signature is None:
                continue
            removed += 1
            active = [
                index for index, value in enumerate(signature) if value > 0.0
            ]
            for index in active:
                self._active_to_episodes[index].discard(episode_id)
            for left_position, left in enumerate(active):
                for right in active[left_position + 1 :]:
                    pair = (left, right)
                    self._coactivations[pair] -= 1
                    if self._coactivations[pair] <= 0:
                        del self._coactivations[pair]
        return removed

    def neighbors(
        self,
        seed_episode_ids: Sequence[str],
        *,
        top_k: int,
        exclude: Sequence[str] = (),
    ) -> tuple[CAVNeighbor, ...]:
        """Find episodes sharing active CAVs with any seed episode."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        excluded = set(exclude) | set(seed_episode_ids)
        candidates: set[str] = set()
        seeds = [
            self._signatures[episode_id]
            for episode_id in dict.fromkeys(seed_episode_ids)
            if episode_id in self._signatures
        ]
        for signature in seeds:
            for index, value in enumerate(signature):
                if value > 0.0:
                    candidates.update(self._active_to_episodes[index])
        candidates.difference_update(excluded)

        ranked: list[CAVNeighbor] = []
        for episode_id in candidates:
            candidate = self._signatures[episode_id]
            best_score = -math.inf
            best_shared: tuple[str, ...] = ()
            for seed in seeds:
                shared_indices = [
                    index
                    for index, (seed_value, candidate_value) in enumerate(
                        zip(seed, candidate, strict=True)
                    )
                    if seed_value > 0.0 and candidate_value > 0.0
                ]
                if not shared_indices:
                    continue
                seed_positive = [max(0.0, value) for value in seed]
                candidate_positive = [max(0.0, value) for value in candidate]
                numerator = sum(
                    left * right
                    for left, right in zip(
                        seed_positive, candidate_positive, strict=True
                    )
                )
                denominator = math.sqrt(sum(value * value for value in seed_positive))
                denominator *= math.sqrt(
                    sum(value * value for value in candidate_positive)
                )
                cosine = numerator / max(denominator, 1e-12)
                union_count = sum(
                    left > 0.0 or right > 0.0
                    for left, right in zip(seed, candidate, strict=True)
                )
                score = cosine + 0.1 * len(shared_indices) / max(union_count, 1)
                if score > best_score:
                    best_score = score
                    best_shared = tuple(
                        self.concept_names[index] for index in shared_indices
                    )
            if best_shared:
                ranked.append(CAVNeighbor(episode_id, best_score, best_shared))
        ranked.sort(key=lambda hit: (hit.score, hit.episode_id), reverse=True)
        return tuple(ranked[:top_k])

    def concept_neighbors(
        self, concept_name: str, *, top_k: int = 5
    ) -> tuple[tuple[str, int], ...]:
        """Return CAVs most often coactivated with a named CAV."""
        if concept_name not in self.concept_names:
            raise KeyError(concept_name)
        concept = self.concept_names.index(concept_name)
        scores: list[tuple[str, int]] = []
        for (left, right), count in self._coactivations.items():
            if left == concept:
                scores.append((self.concept_names[right], count))
            elif right == concept:
                scores.append((self.concept_names[left], count))
        scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return tuple(scores[:top_k])

    @property
    def episode_count(self) -> int:
        return len(self._signatures)

    @property
    def signature_bytes(self) -> int:
        """Float32 payload size, excluding Python/DB row overhead."""
        return len(self._signatures) * len(self.concept_names) * 4


@dataclass(frozen=True, slots=True)
class CAVBank:
    names: tuple[str, ...]
    vectors: Any
    thresholds: Any
    layer: int

    def signature(self, residual: Any) -> Any:
        vector = residual.float()
        if vector.ndim == 3:
            vector = vector.mean(dim=1)
        if vector.ndim == 2:
            vector = vector.mean(dim=0)
        return vector @ self.vectors.T - self.thresholds

    @classmethod
    def load(
        cls,
        report_path: str | Path,
        vectors_path: str | Path,
        *,
        layer: int,
        concepts: Sequence[str] | None = None,
        device: Any = "cpu",
    ) -> "CAVBank":
        import torch
        from safetensors import safe_open

        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        available = {concept["name"]: concept for concept in report["concepts"]}
        names = tuple(available) if concepts is None else tuple(concepts)
        unknown = [name for name in names if name not in available]
        if unknown:
            raise KeyError(f"unknown CAV concepts: {unknown}")
        vectors: list[Any] = []
        thresholds: list[float] = []
        with safe_open(vectors_path, framework="pt", device="cpu") as artifact:
            for name in names:
                key = f"{name}.layer_{layer}"
                if key not in artifact.keys():
                    raise KeyError(f"missing CAV vector: {key}")
                vectors.append(artifact.get_tensor(key).float())
                layer_report = next(
                    item
                    for item in available[name]["layers"]
                    if item["layer"] == layer
                )
                thresholds.append(float(layer_report["threshold"]))
        return cls(
            names=names,
            vectors=torch.stack(vectors).to(device),
            thresholds=torch.tensor(thresholds, dtype=torch.float32, device=device),
            layer=layer,
        )
