"""Sparse in-memory head association graph and bounded graph walk."""

from __future__ import annotations

import math
from typing import Any, Sequence

from memory_condense.associations.head_memory_models import HeadAssociationEdge


class HeadAssociationGraph:
    """Sparse episode graph compiled from QK maps in shared write contexts."""

    def __init__(self) -> None:
        self._adjacency: dict[str, dict[str, HeadAssociationEdge]] = {}
        self.selected_heads: tuple[int, ...] = ()
        self.selected_temporal_forward: bool | None = None

    def _score(self, weights: Any) -> float:
        import torch

        if self.selected_heads:
            return float(weights[list(self.selected_heads)].mean())
        strongest = torch.topk(weights, k=min(4, len(weights))).values
        return float(strongest.mean())

    def add(
        self,
        source_id: str,
        destination_id: str,
        head_weights: Any,
        *,
        reverse: bool = True,
        ov_transport: float = 0.0,
    ) -> None:
        if source_id == destination_id:
            return
        weights = head_weights.detach().float().cpu()
        self._merge(
            source_id,
            destination_id,
            weights,
            temporal_forward=False,
            ov_transport=ov_transport,
        )
        if reverse:
            self._merge(
                destination_id,
                source_id,
                weights,
                temporal_forward=True,
                ov_transport=ov_transport,
            )

    def _merge(
        self,
        source_id: str,
        destination_id: str,
        weights: Any,
        *,
        temporal_forward: bool,
        ov_transport: float,
    ) -> None:
        edges = self._adjacency.setdefault(source_id, {})
        current = edges.get(destination_id)
        if current is None:
            edges[destination_id] = HeadAssociationEdge(
                source_id=source_id,
                destination_id=destination_id,
                head_weights=weights,
                score=self._score(weights),
                ov_transport=max(0.0, float(ov_transport)),
                temporal_forward=temporal_forward,
            )
            return
        count = current.evidence_count
        current.head_weights = (current.head_weights * count + weights) / (count + 1)
        current.score = self._score(current.head_weights)
        current.ov_transport = (
            current.ov_transport * count + max(0.0, float(ov_transport))
        ) / (count + 1)
        current.evidence_count += 1
        if current.temporal_forward != temporal_forward:
            current.temporal_forward = None

    def calibrate_heads(
        self,
        associations: Sequence[tuple[str, str]],
        *,
        keep: int = 4,
    ) -> dict[str, Any]:
        """Select heads whose edge weights recover known related episodes."""
        import torch

        if not associations:
            self.selected_heads = ()
            self.selected_temporal_forward = None
            return {"selected_heads": [], "head_mrr": []}
        sample_edge = next(
            (
                edge
                for edges in self._adjacency.values()
                for edge in edges.values()
            ),
            None,
        )
        if sample_edge is None:
            self.selected_heads = ()
            self.selected_temporal_forward = None
            return {"selected_heads": [], "head_mrr": []}
        head_count = len(sample_edge.head_weights)
        reciprocal_ranks = torch.zeros(head_count, dtype=torch.float32)
        observations = torch.zeros(head_count, dtype=torch.float32)
        direction_votes: list[bool] = []
        for source_id, expected_id in associations:
            edges = list(self._adjacency.get(source_id, {}).values())
            if not edges:
                continue
            expected_edge = next(
                (edge for edge in edges if edge.destination_id == expected_id),
                None,
            )
            if expected_edge is not None and expected_edge.temporal_forward is not None:
                direction_votes.append(expected_edge.temporal_forward)
            for head in range(head_count):
                ranked = sorted(
                    edges,
                    key=lambda edge: float(edge.head_weights[head]),
                    reverse=True,
                )
                observations[head] += 1
                for rank, edge in enumerate(ranked, start=1):
                    if edge.destination_id == expected_id:
                        reciprocal_ranks[head] += 1.0 / rank
                        break
        head_mrr = reciprocal_ranks / observations.clamp_min(1)
        keep = max(1, min(keep, head_count))
        selected = torch.topk(head_mrr, k=keep).indices.tolist()
        self.selected_heads = tuple(int(head) for head in selected)
        if direction_votes:
            forward_votes = sum(direction_votes)
            self.selected_temporal_forward = forward_votes * 2 >= len(direction_votes)
        else:
            self.selected_temporal_forward = None
        for edges in self._adjacency.values():
            for edge in edges.values():
                edge.score = self._score(edge.head_weights)
        return {
            "selected_heads": list(self.selected_heads),
            "head_mrr": [float(value) for value in head_mrr.tolist()],
            "selected_head_mrr": [float(head_mrr[head]) for head in self.selected_heads],
            "selected_temporal_direction": (
                None
                if self.selected_temporal_forward is None
                else "forward"
                if self.selected_temporal_forward
                else "backward"
            ),
        }

    def neighbors(self, episode_id: str) -> tuple[HeadAssociationEdge, ...]:
        edges = list(self._adjacency.get(episode_id, {}).values())
        if self.selected_temporal_forward is not None:
            edges = [
                edge
                for edge in edges
                if edge.temporal_forward == self.selected_temporal_forward
            ]
        return tuple(sorted(edges, key=lambda edge: edge.score, reverse=True))

    def edges(self) -> tuple[HeadAssociationEdge, ...]:
        """Export compact directed edges for an external persistence backend."""
        return tuple(
            edge
            for source_id in sorted(self._adjacency)
            for edge in self.neighbors(source_id)
        )

    def remove_episode_ids(self, episode_ids: Sequence[str]) -> int:
        """Remove pruned episodes as both graph sources and destinations."""
        removed_ids = set(episode_ids)
        if not removed_ids:
            return 0
        removed_edges = 0
        for source_id in list(self._adjacency):
            if source_id in removed_ids:
                removed_edges += len(self._adjacency.pop(source_id))
                continue
            edges = self._adjacency[source_id]
            for destination_id in list(edges):
                if destination_id in removed_ids:
                    del edges[destination_id]
                    removed_edges += 1
            if not edges:
                del self._adjacency[source_id]
        return removed_edges

    def prune_neighbors(self, max_neighbors: int) -> int:
        """Bound persistent graph degree using QK score plus transported value."""
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        removed = 0
        for source_id in list(self._adjacency):
            edges = self._adjacency[source_id]
            ranked = sorted(
                edges.values(),
                key=lambda edge: (edge.score + math.log1p(edge.ov_transport)),
                reverse=True,
            )
            keep = {edge.destination_id for edge in ranked[:max_neighbors]}
            for destination_id in list(edges):
                if destination_id not in keep:
                    del edges[destination_id]
                    removed += 1
            if not edges:
                del self._adjacency[source_id]
        return removed

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adjacency.values())


def _rank_association_walk(
    graph: HeadAssociationGraph,
    seeds: Sequence[tuple[str, float]],
    *,
    top_k: int,
    hops: int,
) -> tuple[list[tuple[str, float]], tuple[tuple[str, ...], ...]]:
    """Fuse semantic seeds and graph evidence without cycle score inflation."""
    selected = dict(seeds)
    calibrated_direction = graph.selected_temporal_forward is not None
    frontier = list(selected)
    hop_ids: list[tuple[str, ...]] = [tuple(frontier)]
    for depth in range(hops):
        candidates: list[tuple[float, str]] = []
        for parent_rank, parent_id in enumerate(frontier):
            parent_score = selected[parent_id]
            for edge in graph.neighbors(parent_id):
                score = parent_score + edge.score / (depth + 1) - 0.01 * parent_rank
                if edge.destination_id in selected:
                    if calibrated_direction:
                        # An edge in the calibrated direction corroborates an
                        # existing semantic seed. Max avoids cycle inflation.
                        selected[edge.destination_id] = max(
                            selected[edge.destination_id], score
                        )
                    continue
                candidates.append((score, edge.destination_id))
        candidates.sort(reverse=True)
        frontier = []
        for score, episode_id in candidates:
            if episode_id in selected:
                continue
            selected[episode_id] = score
            frontier.append(episode_id)
            if len(selected) >= top_k:
                break
        hop_ids.append(tuple(frontier))
        if len(selected) >= top_k or not frontier:
            break
    if calibrated_direction:
        ranked = sorted(
            selected.items(), key=lambda pair: (pair[1], pair[0]), reverse=True
        )[:top_k]
    else:
        # Without calibration, graph-score scale and temporal orientation are
        # unknown. Preserve the stronger semantic ordering and only fill slots.
        ranked = list(selected.items())[:top_k]
    return ranked, tuple(hop_ids)
