"""Compact concept-vector artifact loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def cav_layer_key(name: str, layer: int) -> str:
    """Name one concept's per-layer vector inside a CAV safetensors artifact."""
    return f"{name}.layer_{layer}"


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

    def signatures(self, pooled: Any) -> Any:
        """Project a batch of already-pooled residuals onto every concept.

        Rows stay on the CPU: the batched path never materializes an attention
        map, so the bank is moved to the pooled vectors rather than the reverse.
        """
        vectors = self.vectors.float().cpu()
        thresholds = self.thresholds.float().cpu()
        return pooled.float() @ vectors.T - thresholds

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
                key = cav_layer_key(name, layer)
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
