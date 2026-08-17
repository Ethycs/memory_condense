"""Experimental live per-head memory backed by the Qwen3 prefix encoder."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from memory_condense.decay import decay_factor
from memory_condense.qwen_prefix import Qwen3PrefixEncoder


@dataclass(slots=True)
class HeadMemoryItem:
    episode_id: str
    text: str
    keys: Any
    values: Any
    cav_signature: Any | None
    residual: Any | None = None
    association_residual: Any | None = None
    importance: float = 0.0
    created_turn: int = 0
    last_access_turn: int = 0
    access_count: int = 0
    qk_attention_mass: float = 0.0
    ov_transport: float = 0.0
    last_head_turn: int = 0
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HeadAddress:
    indices: Any
    aggregate_scores: Any
    head_scores: Any
    # Full external-attention probabilities: [query_heads, query_slots, memory_slots].
    head_weights: Any
    # OV input before W_O: [query_slots, query_heads, head_dim].
    mixed_values: Any
    concept_scores: Any | None
    slot_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LiveMemoryHit:
    episode_id: str
    text: str
    score: float
    access_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LiveMemoryResult:
    hits: tuple[LiveMemoryHit, ...]
    hop_episode_ids: tuple[tuple[str, ...], ...]
    query_cav_signature: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AssociativeMemoryCandidate:
    """A retrieval candidate tagged with the route that produced it."""

    episode_id: str
    text: str
    score: float = 0.0
    route: str = "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssociativeComposition:
    """Bounded result of recycling redundant direct-retrieval slots."""

    candidates: tuple[AssociativeMemoryCandidate, ...]
    duplicates_removed: int
    qk_added: int
    residual_added: int
    anchors_displaced: int


@dataclass(frozen=True, slots=True)
class MemoryLinkHit:
    """Compact evidence retained after a transient head-linking pass."""

    episode_id: str
    qk_score: float
    ov_transport: float
    head_weights: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryLinkResult:
    hits: tuple[MemoryLinkHit, ...]
    source_cav_signature: tuple[float, ...]
    workspace_candidates: int
    workspace_tokens: int
    passes: int = 1
    total_candidate_inspections: int = 0


@dataclass(frozen=True, slots=True)
class NestedMemoryInspection:
    """Finalists from recursive fresh workspaces; no token state crosses hops."""

    hits: tuple[MemoryLinkHit, ...]
    passes: int
    max_workspace_candidates: int
    max_workspace_tokens: int
    total_candidate_inspections: int


@dataclass(frozen=True, slots=True)
class CAVNeighbor:
    episode_id: str
    score: float
    shared_concepts: tuple[str, ...]


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


class QwenMemoryLinker:
    """Use real QK/OV circuits transiently, retaining only compact links.

    Candidate text and token activations exist only for one bounded forward
    pass. The persistent memory is an external episode store plus the sparse
    association graph; no per-token K/V cache grows with corpus size.
    """

    def __init__(
        self,
        encoder: Qwen3PrefixEncoder,
        *,
        layer: int = 1,
        cav_bank: CAVBank | None = None,
        max_candidates: int = 8,
        max_workspace_tokens: int = 1024,
        max_neighbors_per_episode: int = 16,
        head_vote_k: int = 4,
    ) -> None:
        if not 0 <= layer < encoder.layers:
            raise IndexError(f"layer must be in [0, {encoder.layers})")
        if cav_bank is not None and not 0 <= cav_bank.layer < encoder.layers:
            raise IndexError("CAV layer is outside the loaded Qwen prefix")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if max_workspace_tokens < 1:
            raise ValueError("max_workspace_tokens must be positive")
        if max_neighbors_per_episode < 1:
            raise ValueError("max_neighbors_per_episode must be positive")
        if head_vote_k < 1:
            raise ValueError("head_vote_k must be positive")
        self.encoder = encoder
        self.layer = int(layer)
        self.cav_bank = cav_bank
        self.max_candidates = int(max_candidates)
        self.max_workspace_tokens = int(max_workspace_tokens)
        self.max_neighbors_per_episode = int(max_neighbors_per_episode)
        self.head_vote_k = int(head_vote_k)

    def signature(self, source_text: str) -> tuple[float, ...]:
        """Compile one memory to compact CAV coordinates without retaining K/V."""
        if self.cav_bank is None:
            return ()
        tokenized = self.encoder.tokenizer(source_text, return_tensors="pt")
        token_count = int(tokenized["input_ids"].shape[1])
        if token_count > self.max_workspace_tokens:
            raise MemoryError(
                f"signature workspace needs {token_count} tokens, above the "
                f"hard cap of {self.max_workspace_tokens}; chunk the memory first"
            )
        capture = self.encoder.capture(source_text, layer=self.cav_bank.layer)
        values = self.cav_bank.signature(capture.residual)
        return tuple(float(value) for value in values.cpu().tolist())

    def link(
        self,
        source_text: str,
        candidates: Sequence[AssociativeMemoryCandidate],
        *,
        top_k: int | None = None,
    ) -> MemoryLinkResult:
        """Score a bounded candidate workspace and immediately shed activations."""
        torch = self.encoder._torch
        bounded: list[AssociativeMemoryCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.episode_id in seen:
                continue
            seen.add(candidate.episode_id)
            bounded.append(candidate)
            if len(bounded) >= self.max_candidates:
                break
        if not bounded:
            raise ValueError("at least one unique link candidate is required")

        parts: list[str] = []
        character_spans: list[tuple[int, int]] = []
        cursor = 0
        for position, candidate in enumerate(bounded):
            part = f"[Memory {position}] {candidate.text}\n"
            parts.append(part)
            character_spans.append((cursor, cursor + len(part)))
            cursor += len(part)
        query_part = f"[New memory] {source_text}\n"
        parts.append(query_part)
        query_character_span = (cursor, cursor + len(query_part))
        joint_text = "".join(parts)

        tokenized = self.encoder.tokenizer(
            joint_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        workspace_tokens = int(tokenized["input_ids"].shape[1])
        if workspace_tokens > self.max_workspace_tokens:
            raise MemoryError(
                f"link workspace needs {workspace_tokens} tokens, above the "
                f"hard cap of {self.max_workspace_tokens}; shrink or chunk candidates"
            )
        offsets = tokenized["offset_mapping"][0].tolist()

        def token_span(character_span: tuple[int, int]) -> list[int]:
            start, stop = character_span
            return [
                token_index
                for token_index, (token_start, token_stop) in enumerate(offsets)
                if token_stop > start and token_start < stop
            ]

        candidate_token_spans = [token_span(span) for span in character_spans]
        source_tokens = token_span(query_character_span)
        if not source_tokens or any(not span for span in candidate_token_spans):
            raise ValueError("tokenizer produced an empty memory span")

        capture_layers = [self.layer]
        if self.cav_bank is not None and self.cav_bank.layer not in capture_layers:
            capture_layers.append(self.cav_bank.layer)
        captures = self.encoder.capture_layers(joint_text, layers=capture_layers)
        capture = captures[self.layer]
        attention = capture.attention[0].float()
        decoder_attention = self.encoder.model.layers[self.layer].self_attn
        groups_per_key = (
            self.encoder.config.num_attention_heads
            // self.encoder.config.num_key_value_heads
        )
        key_groups = (
            torch.arange(
                self.encoder.config.num_attention_heads,
                device=self.encoder.device,
            )
            // groups_per_key
        )
        expanded_values = capture.values[0][key_groups]
        hits: list[MemoryLinkHit] = []
        with torch.inference_mode():
            for candidate, memory_tokens in zip(
                bounded, candidate_token_spans, strict=True
            ):
                block = attention[:, source_tokens, :][:, :, memory_tokens]
                head_weights = block.sum(dim=-1).mean(dim=-1)
                strongest = torch.topk(
                    head_weights,
                    k=min(self.head_vote_k, len(head_weights)),
                ).values
                candidate_values = expanded_values[:, memory_tokens, :]
                moved_values = torch.einsum(
                    "hqs,hsd->qhd", block.to(candidate_values.dtype), candidate_values
                )
                update = decoder_attention.o_proj(
                    moved_values.reshape(1, moved_values.shape[0], -1)
                )
                hits.append(
                    MemoryLinkHit(
                        episode_id=candidate.episode_id,
                        qk_score=float(strongest.mean()),
                        ov_transport=float(update.float().square().mean().sqrt()),
                        head_weights=tuple(
                            float(value) for value in head_weights.cpu().tolist()
                        ),
                        metadata=dict(candidate.metadata),
                    )
                )
        hits.sort(
            key=lambda hit: (hit.qk_score, hit.ov_transport, hit.episode_id),
            reverse=True,
        )
        count = len(hits) if top_k is None else max(1, min(int(top_k), len(hits)))
        signature: tuple[float, ...] = ()
        if self.cav_bank is not None:
            cav_capture = captures[self.cav_bank.layer]
            values = self.cav_bank.signature(
                cav_capture.residual[:, source_tokens, :]
            )
            signature = tuple(float(value) for value in values.cpu().tolist())
        return MemoryLinkResult(
            hits=tuple(hits[:count]),
            source_cav_signature=signature,
            workspace_candidates=len(bounded),
            workspace_tokens=workspace_tokens,
            passes=1,
            total_candidate_inspections=len(bounded),
        )

    def link_into_graph(
        self,
        graph: HeadAssociationGraph,
        source_id: str,
        source_text: str,
        candidates: Sequence[AssociativeMemoryCandidate],
        *,
        top_k: int | None = None,
    ) -> MemoryLinkResult:
        """Compile a transient pass into a bounded persistent association graph."""
        result = self.link(source_text, candidates, top_k=top_k)
        torch = self.encoder._torch
        for hit in result.hits:
            graph.add(
                source_id,
                hit.episode_id,
                torch.tensor(hit.head_weights),
                ov_transport=hit.ov_transport,
            )
        graph.prune_neighbors(self.max_neighbors_per_episode)
        return result

    def inspect_nested(
        self,
        source_text: str,
        candidate_groups: Sequence[Sequence[AssociativeMemoryCandidate]],
        *,
        beam_per_group: int = 2,
        top_k: int = 4,
        score_mode: Literal["qk", "qk_ov"] = "qk",
    ) -> NestedMemoryInspection:
        """Recursively inspect candidate groups without carrying transformer state.

        Each pass re-encodes only its small group. Only candidate IDs, text
        pointers, and scalar scores cross into the next tournament layer.
        """
        if beam_per_group < 1:
            raise ValueError("beam_per_group must be positive")
        if beam_per_group >= self.max_candidates:
            raise ValueError("beam_per_group must be smaller than max_candidates")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if score_mode not in {"qk", "qk_ov"}:
            raise ValueError("score_mode must be 'qk' or 'qk_ov'")
        groups = [list(group) for group in candidate_groups if group]
        if not groups:
            raise ValueError("at least one non-empty candidate group is required")
        if any(len(group) > self.max_candidates for group in groups):
            raise ValueError("a candidate group exceeds max_candidates")

        passes = 0
        max_candidates = 0
        max_tokens = 0
        total_inspections = 0

        def hit_score(hit: MemoryLinkHit) -> float:
            qk = max(0.0, float(hit.qk_score))
            if score_mode == "qk":
                return qk
            return qk + math.log1p(max(0.0, float(hit.ov_transport)))

        def ranked_hits(result: MemoryLinkResult) -> list[MemoryLinkHit]:
            return sorted(
                result.hits,
                key=lambda hit: (hit_score(hit), hit.episode_id),
                reverse=True,
            )

        def inspect_groups(
            current_groups: Sequence[Sequence[AssociativeMemoryCandidate]],
        ) -> list[AssociativeMemoryCandidate]:
            nonlocal passes, max_candidates, max_tokens, total_inspections
            finalists: list[AssociativeMemoryCandidate] = []
            seen_finalists: set[str] = set()
            for group in current_groups:
                result = self.link(
                    source_text,
                    group,
                    top_k=None,
                )
                passes += 1
                max_candidates = max(max_candidates, result.workspace_candidates)
                max_tokens = max(max_tokens, result.workspace_tokens)
                total_inspections += result.workspace_candidates
                by_id = {candidate.episode_id: candidate for candidate in group}
                for hit in ranked_hits(result)[:beam_per_group]:
                    if hit.episode_id in seen_finalists:
                        continue
                    seen_finalists.add(hit.episode_id)
                    original = by_id[hit.episode_id]
                    finalists.append(
                        AssociativeMemoryCandidate(
                            episode_id=original.episode_id,
                            text=original.text,
                            score=hit_score(hit),
                            route=original.route,
                            metadata=dict(original.metadata),
                        )
                    )
            return finalists

        finalists = inspect_groups(groups)
        while len(finalists) > self.max_candidates:
            next_groups = [
                finalists[start : start + self.max_candidates]
                for start in range(0, len(finalists), self.max_candidates)
            ]
            finalists = inspect_groups(next_groups)
        while True:
            try:
                final = self.link(
                    source_text,
                    finalists,
                    top_k=None,
                )
                break
            except MemoryError:
                if len(finalists) == 1:
                    raise
                reduced: list[AssociativeMemoryCandidate] = []
                for start in range(0, len(finalists), 2):
                    group = finalists[start : start + 2]
                    result = self.link(source_text, group, top_k=None)
                    passes += 1
                    max_candidates = max(
                        max_candidates, result.workspace_candidates
                    )
                    max_tokens = max(max_tokens, result.workspace_tokens)
                    total_inspections += result.workspace_candidates
                    winner = ranked_hits(result)[0]
                    original = next(
                        candidate
                        for candidate in group
                        if candidate.episode_id == winner.episode_id
                    )
                    reduced.append(
                        AssociativeMemoryCandidate(
                            episode_id=original.episode_id,
                            text=original.text,
                            score=hit_score(winner),
                            route=original.route,
                            metadata=dict(original.metadata),
                        )
                    )
                finalists = reduced
        passes += 1
        max_candidates = max(max_candidates, final.workspace_candidates)
        max_tokens = max(max_tokens, final.workspace_tokens)
        total_inspections += final.workspace_candidates
        return NestedMemoryInspection(
            hits=tuple(ranked_hits(final)[:top_k]),
            passes=passes,
            max_workspace_candidates=max_candidates,
            max_workspace_tokens=max_tokens,
            total_candidate_inspections=total_inspections,
        )


def compose_associative_candidates(
    anchors: Sequence[AssociativeMemoryCandidate],
    *,
    qk_neighbors: Sequence[AssociativeMemoryCandidate] = (),
    residual_candidates: Sequence[AssociativeMemoryCandidate] = (),
    top_k: int = 10,
    qk_reserve: int = 1,
    association_slots: int = 0,
    protected_anchor_ids: Sequence[str] = (),
) -> AssociativeComposition:
    """Keep every unique direct anchor and recycle only redundant slots.

    QK neighbors get first use of the recycled capacity, followed by residual
    candidates. Neither association route can displace unique direct evidence.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if qk_reserve < 0:
        raise ValueError("qk_reserve must be non-negative")
    if association_slots < 0:
        raise ValueError("association_slots must be non-negative")
    protected = set(protected_anchor_ids)

    unique_anchors: list[AssociativeMemoryCandidate] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    duplicates_removed = 0

    def content_key(candidate: AssociativeMemoryCandidate) -> str:
        return re.sub(r"\s+", " ", candidate.text).strip().casefold()

    for candidate in anchors[:top_k]:
        content = content_key(candidate)
        if candidate.episode_id in seen_ids or content in seen_content:
            duplicates_removed += 1
            continue
        seen_ids.add(candidate.episode_id)
        seen_content.add(content)
        unique_anchors.append(candidate)

    # Only a contiguous unprotected suffix may be displaced. If the weakest
    # anchor carries protected direct evidence, moving the reservation upward
    # would perversely replace an even stronger anchor instead.
    reserved = 0
    for candidate in reversed(unique_anchors):
        if reserved >= association_slots or candidate.episode_id in protected:
            break
        reserved += 1
    selected = list(unique_anchors[: len(unique_anchors) - reserved])
    held_anchors = unique_anchors[len(unique_anchors) - reserved :]
    seen_ids = {candidate.episode_id for candidate in selected}
    seen_content = {content_key(candidate) for candidate in selected}

    def add(candidate: AssociativeMemoryCandidate) -> bool:
        content = content_key(candidate)
        if candidate.episode_id in seen_ids or content in seen_content:
            return False
        seen_ids.add(candidate.episode_id)
        seen_content.add(content)
        selected.append(candidate)
        return True

    # Association routes may consume only slots freed by duplicate direct
    # results unless the caller explicitly reserves a fixed number of slots.
    capacity = min(duplicates_removed + reserved, top_k - len(selected))
    qk_added = 0
    for candidate in qk_neighbors:
        if qk_added >= min(qk_reserve, capacity):
            break
        if add(candidate):
            qk_added += 1

    residual_added = 0
    residual_capacity = capacity - qk_added
    for candidate in residual_candidates:
        if residual_added >= residual_capacity:
            break
        if add(candidate):
            residual_added += 1

    # If association routes cannot use every reserved slot, restore the held
    # direct anchors before returning a short result.
    for candidate in held_anchors:
        if len(selected) >= top_k:
            break
        add(candidate)

    selected_ids = {candidate.episode_id for candidate in selected}
    anchors_displaced = sum(
        candidate.episode_id not in selected_ids for candidate in unique_anchors
    )

    return AssociativeComposition(
        candidates=tuple(selected[:top_k]),
        duplicates_removed=duplicates_removed,
        qk_added=qk_added,
        residual_added=residual_added,
        anchors_displaced=anchors_displaced,
    )


@dataclass(slots=True)
class HeadAssociationEdge:
    source_id: str
    destination_id: str
    head_weights: Any
    score: float
    ov_transport: float = 0.0
    evidence_count: int = 1
    temporal_forward: bool | None = None


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


class HeadKVStore:
    """Append-immediate external K/V memory with GQA-aware addressing."""

    def __init__(
        self,
        *,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
        device: Any,
        head_vote_k: int = 4,
    ) -> None:
        if query_heads % key_value_heads:
            raise ValueError("query_heads must be divisible by key_value_heads")
        if head_vote_k < 1:
            raise ValueError("head_vote_k must be positive")
        self.query_heads = query_heads
        self.key_value_heads = key_value_heads
        self.head_dim = head_dim
        self.device = device
        self.head_vote_k = min(head_vote_k, query_heads)
        self.items: list[HeadMemoryItem] = []
        self._episode_ids: set[str] = set()
        self._episode_to_index: dict[str, int] = {}
        self.turn = 0

    def write(self, item: HeadMemoryItem) -> None:
        if item.episode_id in self._episode_ids:
            raise ValueError(f"duplicate episode id: {item.episode_id}")
        if item.keys.ndim == 2:
            item.keys = item.keys.unsqueeze(1)
        if item.values.ndim == 2:
            item.values = item.values.unsqueeze(1)
        expected_edges = (self.key_value_heads, self.head_dim)
        valid_keys = (
            item.keys.ndim == 3
            and (item.keys.shape[0], item.keys.shape[2]) == expected_edges
        )
        valid_values = (
            item.values.ndim == 3
            and (item.values.shape[0], item.values.shape[2]) == expected_edges
            and item.values.shape[1] == item.keys.shape[1]
        )
        if not valid_keys or not valid_values:
            raise ValueError(
                "keys and values must have shape "
                f"[kv_heads, slots, head_dim] with edges {expected_edges}; got "
                f"{tuple(item.keys.shape)} and {tuple(item.values.shape)}"
            )
        if item.cav_signature is not None and item.cav_signature.ndim != 1:
            raise ValueError("CAV signatures must be rank-1")
        self.turn += 1
        item.created_turn = self.turn
        item.last_access_turn = self.turn
        item.last_head_turn = self.turn
        item.keys = item.keys.detach().to(self.device)
        item.values = item.values.detach().to(self.device)
        if item.cav_signature is not None:
            item.cav_signature = item.cav_signature.detach().float().to(self.device)
        if item.residual is not None:
            if item.residual.ndim != 1:
                raise ValueError("pooled residuals must be rank-1")
            item.residual = item.residual.detach().float().to(self.device)
        if item.association_residual is not None:
            if item.association_residual.ndim != 1:
                raise ValueError("association residuals must be rank-1")
            item.association_residual = (
                item.association_residual.detach().float().to(self.device)
            )
        self.items.append(item)
        self._episode_ids.add(item.episode_id)
        self._episode_to_index[item.episode_id] = len(self.items) - 1

    def indices_for_episode_ids(self, episode_ids: Sequence[str]) -> list[int]:
        """Resolve existing episode IDs to de-duplicated item indices."""
        return list(
            dict.fromkeys(
                self._episode_to_index[episode_id]
                for episode_id in episode_ids
                if episode_id in self._episode_to_index
            )
        )

    def address(
        self,
        queries: Any,
        *,
        top_k: int,
        scaling: float,
        cav_signature: Any | None = None,
        cav_weight: float = 0.0,
        cav_mode: str = "similarity",
        candidate_indices: Sequence[int] | None = None,
    ) -> HeadAddress:
        import torch

        if not self.items:
            raise LookupError("cannot address an empty head memory")
        if queries.ndim == 2:
            queries = queries.unsqueeze(1)
        if (
            queries.ndim != 3
            or queries.shape[0] != self.query_heads
            or queries.shape[2] != self.head_dim
        ):
            raise ValueError(
                "queries must have shape [query_heads, query_slots, head_dim]"
            )
        if candidate_indices is None:
            item_indices = list(range(len(self.items)))
        else:
            item_indices = list(dict.fromkeys(int(index) for index in candidate_indices))
            if not item_indices:
                raise ValueError("candidate_indices cannot be empty")
            if min(item_indices) < 0 or max(item_indices) >= len(self.items):
                raise IndexError("candidate index is outside the memory store")
        candidate_items = [self.items[index] for index in item_indices]
        top_k = max(1, min(int(top_k), len(candidate_items)))
        groups_per_key = self.query_heads // self.key_value_heads
        key_groups = torch.arange(self.query_heads, device=self.device) // groups_per_key
        slot_ranges: list[tuple[int, int]] = [(0, 0)] * len(self.items)
        slot_cursor = 0
        for item_index, item in zip(item_indices, candidate_items, strict=True):
            slot_stop = slot_cursor + item.keys.shape[1]
            slot_ranges[item_index] = (slot_cursor, slot_stop)
            slot_cursor = slot_stop
        keys = torch.cat([item.keys for item in candidate_items], dim=1)
        values = torch.cat([item.values for item in candidate_items], dim=1)
        slot_owners = torch.cat(
            [
                torch.full(
                    (item.keys.shape[1],),
                    item_index,
                    dtype=torch.long,
                    device=self.device,
                )
                for item_index, item in zip(item_indices, candidate_items, strict=True)
            ]
        )
        expanded_keys = keys[key_groups]
        expanded_values = values[key_groups]
        logits = torch.einsum("hqd,hsd->hqs", queries, expanded_keys) * scaling

        concept_scores = None
        if cav_signature is not None and cav_weight != 0.0:
            if any(item.cav_signature is None for item in candidate_items):
                raise ValueError("all memory items need CAV signatures when gating is enabled")
            signatures = torch.stack(
                [item.cav_signature for item in candidate_items], dim=0
            )
            scale = signatures.std(dim=0, unbiased=False).clamp_min(0.25)
            if cav_mode == "similarity":
                query_signature = cav_signature.float().to(self.device)
                if signatures.shape[1:] != query_signature.shape:
                    raise ValueError("query and item CAV signatures have different shapes")
                concept_scores = -(
                    ((signatures - query_signature) / scale).square().mean(dim=1)
                ).sqrt()
            elif cav_mode == "positive":
                # A CAV threshold is a type decision, not a reason to let an
                # extreme margin dominate semantic relevance.
                concept_scores = (signatures > 0).float().mean(dim=1)
            else:
                raise ValueError("cav_mode must be 'similarity' or 'positive'")
            local_slot_owners = torch.cat(
                [
                    torch.full(
                        (item.keys.shape[1],),
                        local_index,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for local_index, item in enumerate(candidate_items)
                ]
            )
            logits = logits + cav_weight * concept_scores[local_slot_owners].view(
                1, 1, -1
            )

        # This is the external-memory equivalent of the real attention circuit:
        # QK supplies a distribution over every stored token slot, then V and O
        # determine what residual information is transported back to the query.
        attention_weights = logits.float().softmax(dim=-1).to(expanded_values.dtype)
        mixed_values = torch.einsum(
            "hqs,hsd->qhd", attention_weights, expanded_values
        )
        episode_mass = torch.zeros(
            self.query_heads,
            queries.shape[1],
            len(self.items),
            dtype=attention_weights.dtype,
            device=self.device,
        )
        episode_mass.scatter_add_(
            2,
            slot_owners.view(1, 1, -1).expand(
                self.query_heads, queries.shape[1], -1
            ),
            attention_weights,
        )
        head_scores = episode_mass.mean(dim=1)
        strongest = torch.topk(head_scores, k=self.head_vote_k, dim=0).values
        aggregate = strongest.mean(dim=0)
        candidate_tensor = torch.tensor(
            item_indices, dtype=torch.long, device=self.device
        )
        selected_scores, selected_local_indices = torch.topk(
            aggregate[candidate_tensor], k=top_k
        )
        selected_indices = candidate_tensor[selected_local_indices]
        selected_head_scores = head_scores[:, selected_indices]
        return HeadAddress(
            indices=selected_indices,
            aggregate_scores=selected_scores,
            head_scores=selected_head_scores,
            head_weights=attention_weights,
            mixed_values=mixed_values,
            concept_scores=(
                None
                if concept_scores is None
                else concept_scores[selected_local_indices]
            ),
            slot_ranges=tuple(slot_ranges),
        )

    def residual_scores(
        self,
        query_residual: Any,
        *,
        cav_weight: float = 0.0,
        cav_mode: str = "positive",
        query_cav_signature: Any | None = None,
        association_layer: bool = False,
    ) -> Any:
        """Score the semantic entry point before traversing head associations."""
        import torch
        import torch.nn.functional as functional

        if not self.items:
            raise LookupError("cannot address an empty head memory")
        residuals_by_item = [
            item.association_residual if association_layer else item.residual
            for item in self.items
        ]
        if any(residual is None for residual in residuals_by_item):
            kind = "association-layer" if association_layer else "entry-layer"
            raise ValueError(f"all memory items need {kind} pooled residuals")
        residuals = torch.stack(residuals_by_item, dim=0)
        query = query_residual.float().to(self.device)
        scores = functional.cosine_similarity(residuals, query.unsqueeze(0), dim=1)
        if cav_weight:
            if any(item.cav_signature is None for item in self.items):
                raise ValueError("all memory items need CAV signatures when gating is enabled")
            signatures = torch.stack(
                [item.cav_signature for item in self.items], dim=0
            )
            scale = signatures.std(dim=0, unbiased=False).clamp_min(0.25)
            if cav_mode == "positive":
                concept_scores = (signatures > 0).float().mean(dim=1)
            elif cav_mode == "similarity":
                if query_cav_signature is None:
                    raise ValueError("similarity gating requires a query CAV signature")
                concept_scores = -(
                    (
                        (signatures - query_cav_signature.float().to(self.device))
                        / scale
                    )
                    .square()
                    .mean(dim=1)
                ).sqrt()
            else:
                raise ValueError("cav_mode must be 'similarity' or 'positive'")
            scores = scores + cav_weight * concept_scores
        return scores

    def touch(
        self,
        indices: Any,
        *,
        attention_mass: Any | None = None,
        ov_transport: Any | None = None,
        usage_half_life: float = 100.0,
    ) -> None:
        self.turn += 1
        index_values = indices.tolist()
        attention_values = (
            None if attention_mass is None else attention_mass.float().tolist()
        )
        transport_values = (
            None if ov_transport is None else ov_transport.float().tolist()
        )
        if attention_values is not None and len(attention_values) != len(index_values):
            raise ValueError("attention_mass must align with indices")
        if transport_values is not None and len(transport_values) != len(index_values):
            raise ValueError("ov_transport must align with indices")
        for position, index in enumerate(index_values):
            item = self.items[int(index)]
            item.access_count += 1
            item.last_access_turn = self.turn
            if attention_values is None and transport_values is None:
                continue
            decay = decay_factor(
                item.last_head_turn,
                self.turn,
                half_life_turns=usage_half_life,
            )
            item.qk_attention_mass *= decay
            item.ov_transport *= decay
            if attention_values is not None:
                item.qk_attention_mass += max(0.0, float(attention_values[position]))
            if transport_values is not None:
                item.ov_transport += max(0.0, float(transport_values[position]))
            item.last_head_turn = self.turn

    def prune(self, max_items: int, *, age_half_life: float = 100.0) -> list[str]:
        """Remove the lowest-utility unpinned items from this in-memory store."""
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        removed: list[str] = []
        while len(self.items) > max_items:
            candidates = [
                (self._utility(item, age_half_life), index)
                for index, item in enumerate(self.items)
                if not item.pinned
            ]
            if not candidates:
                break
            _, index = min(candidates)
            item = self.items.pop(index)
            self._episode_ids.remove(item.episode_id)
            removed.append(item.episode_id)
        if removed:
            self._episode_to_index = {
                item.episode_id: index for index, item in enumerate(self.items)
            }
        return removed

    def _utility(self, item: HeadMemoryItem, age_half_life: float) -> float:
        recency = decay_factor(
            item.last_access_turn,
            self.turn,
            half_life_turns=age_half_life,
        )
        head_decay = decay_factor(
            item.last_head_turn,
            self.turn,
            half_life_turns=age_half_life,
        )
        live_head_utility = head_decay * (
            math.log1p(item.qk_attention_mass) + math.log1p(item.ov_transport)
        )
        return item.importance + live_head_utility + recency


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


class QwenLiveHeadMemory:
    """Bounded K/V workspace for experiments, not a corpus-scale memory store."""

    def __init__(
        self,
        encoder: Qwen3PrefixEncoder,
        *,
        layer: int,
        association_layer: int | None = None,
        cav_bank: CAVBank | None = None,
        recursion_gate: float = 0.25,
        association_candidates: int = 4,
        max_items: int = 64,
    ) -> None:
        if not 0 <= layer < encoder.layers:
            raise IndexError(f"layer must be in [0, {encoder.layers})")
        if association_layer is None:
            association_layer = layer
        if not 0 <= association_layer < encoder.layers:
            raise IndexError(f"association_layer must be in [0, {encoder.layers})")
        if cav_bank is not None and cav_bank.layer != layer:
            raise ValueError("CAV bank and head memory must use the same layer")
        self.encoder = encoder
        self.layer = layer
        self.association_layer = association_layer
        self.cav_bank = cav_bank
        self.recursion_gate = float(recursion_gate)
        self.association_candidates = max(0, int(association_candidates))
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.max_items = int(max_items)
        self.decoder = encoder.model.layers[layer]
        attention = self.decoder.self_attn
        self.store = HeadKVStore(
            query_heads=encoder.config.num_attention_heads,
            key_value_heads=encoder.config.num_key_value_heads,
            head_dim=attention.head_dim,
            device=encoder.device,
        )
        self.graph = HeadAssociationGraph()

    def write(
        self,
        episode_id: str,
        text: str,
        *,
        importance: float = 0.0,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> HeadMemoryItem:
        if len(self.store.items) >= self.max_items:
            raise MemoryError(
                "the bounded live-head workspace is full; persist links/CAVs and "
                "discard K/V instead of using it as a corpus store"
            )
        torch = self.encoder._torch
        attention = self.decoder.self_attn
        capture_layers = tuple(dict.fromkeys((self.layer, self.association_layer)))
        captures = self.encoder.capture_layers(text, layers=capture_layers)
        capture = captures[self.layer]
        association_capture = captures[self.association_layer]
        with torch.inference_mode():
            state = capture.attention_input
            head_shape = (*state.shape[:-1], -1, attention.head_dim)
            keys = attention.k_norm(
                attention.k_proj(state).view(head_shape)
            ).transpose(1, 2)[0]
            values = attention.v_proj(state).view(head_shape).transpose(1, 2)[0]
            signature = (
                None
                if self.cav_bank is None
                else self.cav_bank.signature(capture.residual)
            )
        item = HeadMemoryItem(
            episode_id=episode_id,
            text=text,
            keys=keys,
            values=values,
            cav_signature=signature,
            residual=capture.residual.mean(dim=1)[0],
            association_residual=association_capture.residual.mean(dim=1)[0],
            importance=importance,
            pinned=pinned,
            metadata={} if metadata is None else dict(metadata),
        )
        self.store.write(item)
        self._associate_new_item(item)
        return item

    def _associate_new_item(self, item: HeadMemoryItem) -> None:
        """Write QK edges while the new episode and candidates share context."""
        torch = self.encoder._torch
        if self.association_candidates == 0 or len(self.store.items) < 2:
            return
        with torch.inference_mode():
            import torch.nn.functional as functional

            if any(
                stored.association_residual is None for stored in self.store.items
            ):
                raise ValueError("all items need association-layer residuals")
            association_residuals = torch.stack(
                [stored.association_residual for stored in self.store.items], dim=0
            )
            scores = functional.cosine_similarity(
                association_residuals,
                item.association_residual.unsqueeze(0),
                dim=1,
            )
            new_index = len(self.store.items) - 1
            scores[new_index] = -torch.inf
            count = min(self.association_candidates, len(self.store.items) - 1)
            candidate_indices = torch.topk(scores, k=count).indices.tolist()

        candidates = [self.store.items[int(index)] for index in candidate_indices]
        parts: list[str] = []
        character_spans: list[tuple[int, int]] = []
        cursor = 0
        for position, candidate in enumerate([*candidates, item]):
            part = f"[Memory {position}] {candidate.text}\n"
            parts.append(part)
            character_spans.append((cursor, cursor + len(part)))
            cursor += len(part)
        joint_text = "".join(parts)
        tokenized = self.encoder.tokenizer(
            joint_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized["offset_mapping"][0].tolist()
        token_spans: list[list[int]] = []
        for start, stop in character_spans:
            token_spans.append(
                [
                    token_index
                    for token_index, (token_start, token_stop) in enumerate(offsets)
                    if token_stop > start and token_start < stop
                ]
            )

        capture = self.encoder.capture(joint_text, layer=self.association_layer)
        attention = capture.attention[0].float()
        destination_tokens = token_spans[-1]
        for candidate, source_tokens in zip(candidates, token_spans[:-1], strict=True):
            if not destination_tokens or not source_tokens:
                continue
            block = attention[:, destination_tokens, :][:, :, source_tokens]
            # Total source-episode attention mass, averaged over destination
            # tokens, is retained per head.  This is a routing observation,
            # not a causal claim.
            head_weights = block.sum(dim=-1).mean(dim=-1)
            self.graph.add(item.episode_id, candidate.episode_id, head_weights)

    def _selected_ov_transport(self, address: HeadAddress) -> Any:
        """Measure each selected episode's actual contribution through W_O."""
        torch = self.encoder._torch
        attention = self.decoder.self_attn
        groups_per_key = self.store.query_heads // self.store.key_value_heads
        key_groups = (
            torch.arange(self.store.query_heads, device=self.store.device)
            // groups_per_key
        )
        transports: list[Any] = []
        with torch.inference_mode():
            for item_index in address.indices.tolist():
                item = self.store.items[int(item_index)]
                start, stop = address.slot_ranges[int(item_index)]
                item_weights = address.head_weights[:, :, start:stop]
                expanded_values = item.values[key_groups]
                item_values = torch.einsum(
                    "hqs,hsd->qhd", item_weights, expanded_values
                )
                item_update = attention.o_proj(
                    item_values.reshape(1, item_values.shape[0], -1)
                )
                # RMS keeps this comparable across query lengths and model widths.
                transports.append(item_update.float().square().mean().sqrt())
        return torch.stack(transports)

    def _record_candidate_head_use(
        self,
        query: str,
        candidate_indices: Sequence[int],
        *,
        cav_weight: float,
    ) -> None:
        """Activate the entry-layer heads over returned evidence for lifecycle use."""
        if not candidate_indices:
            return
        torch = self.encoder._torch
        capture = self.encoder.capture(query, layer=self.layer)
        query_signature = (
            None
            if self.cav_bank is None
            else self.cav_bank.signature(capture.residual)
        )
        attention = self.decoder.self_attn
        with torch.inference_mode():
            normalized = capture.attention_input
            queries = attention.q_norm(
                attention.q_proj(normalized).view(
                    *normalized.shape[:-1], -1, attention.head_dim
                )
            ).transpose(1, 2)[0]
            addressed = self.store.address(
                queries,
                top_k=len(candidate_indices),
                scaling=float(attention.scaling),
                cav_signature=query_signature,
                cav_weight=cav_weight,
                cav_mode="positive",
                candidate_indices=candidate_indices,
            )
            transports = self._selected_ov_transport(addressed)
        self.store.touch(
            addressed.indices,
            attention_mass=addressed.aggregate_scores,
            ov_transport=transports,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        hops: int = 1,
        cav_weight: float = 0.0,
        cav_mode: str = "similarity",
    ) -> LiveMemoryResult:
        torch = self.encoder._torch
        if hops < 1:
            raise ValueError("hops must be positive")
        capture = self.encoder.capture(query, layer=self.layer)
        query_signature = (
            None
            if self.cav_bank is None
            else self.cav_bank.signature(capture.residual)
        )
        state = capture.layer_input
        initial_attention_input = capture.attention_input
        attention = self.decoder.self_attn
        addresses: list[HeadAddress] = []
        with torch.inference_mode():
            for hop in range(hops):
                normalized = (
                    initial_attention_input
                    if hop == 0
                    else self.decoder.input_layernorm(state)
                )
                queries = attention.q_norm(
                    attention.q_proj(normalized).view(
                        *normalized.shape[:-1], -1, attention.head_dim
                    )
                ).transpose(1, 2)[0]
                addressed = self.store.address(
                    queries,
                    top_k=top_k,
                    scaling=float(attention.scaling),
                    cav_signature=query_signature,
                    cav_weight=cav_weight,
                    cav_mode=cav_mode,
                )
                addresses.append(addressed)
                mixed = addressed.mixed_values.reshape(
                    1, addressed.mixed_values.shape[0], -1
                )
                update = attention.o_proj(mixed)
                state = state + self.recursion_gate * update

        final = addresses[-1]
        self.store.touch(
            final.indices,
            attention_mass=final.aggregate_scores,
            ov_transport=self._selected_ov_transport(final),
        )
        hits = tuple(
            LiveMemoryHit(
                episode_id=self.store.items[int(index)].episode_id,
                text=self.store.items[int(index)].text,
                score=float(score),
                access_count=self.store.items[int(index)].access_count,
                metadata=dict(self.store.items[int(index)].metadata),
            )
            for index, score in zip(
                final.indices.tolist(), final.aggregate_scores.tolist(), strict=True
            )
        )
        hop_ids = tuple(
            tuple(self.store.items[int(index)].episode_id for index in hop.indices.tolist())
            for hop in addresses
        )
        return LiveMemoryResult(
            hits=hits,
            hop_episode_ids=hop_ids,
            query_cav_signature=(
                ()
                if query_signature is None
                else tuple(float(value) for value in query_signature.tolist())
            ),
        )

    def retrieve_candidates(
        self,
        query: str,
        episode_ids: Sequence[str],
        *,
        top_k: int = 4,
        cav_weight: float = 0.0,
        cav_mode: str = "similarity",
    ) -> LiveMemoryResult:
        """Run live QK/OV addressing only over a supplied candidate subgraph.

        Stored K/V tensors are reused, so only the query is encoded. IDs absent
        from the current store are ignored, which keeps calls safe after pruning
        as long as at least one candidate remains.
        """
        torch = self.encoder._torch
        candidate_indices = self.store.indices_for_episode_ids(episode_ids)
        if not candidate_indices:
            raise LookupError("none of the candidate episode IDs are in memory")
        capture = self.encoder.capture(query, layer=self.layer)
        query_signature = (
            None
            if self.cav_bank is None
            else self.cav_bank.signature(capture.residual)
        )
        attention = self.decoder.self_attn
        with torch.inference_mode():
            normalized = capture.attention_input
            queries = attention.q_norm(
                attention.q_proj(normalized).view(
                    *normalized.shape[:-1], -1, attention.head_dim
                )
            ).transpose(1, 2)[0]
            addressed = self.store.address(
                queries,
                top_k=top_k,
                scaling=float(attention.scaling),
                cav_signature=query_signature,
                cav_weight=cav_weight,
                cav_mode=cav_mode,
                candidate_indices=candidate_indices,
            )
            transports = self._selected_ov_transport(addressed)
        self.store.touch(
            addressed.indices,
            attention_mass=addressed.aggregate_scores,
            ov_transport=transports,
        )
        hits = tuple(
            LiveMemoryHit(
                episode_id=self.store.items[int(index)].episode_id,
                text=self.store.items[int(index)].text,
                score=float(score),
                access_count=self.store.items[int(index)].access_count,
                metadata=dict(self.store.items[int(index)].metadata),
            )
            for index, score in zip(
                addressed.indices.tolist(),
                addressed.aggregate_scores.tolist(),
                strict=True,
            )
        )
        ids = tuple(hit.episode_id for hit in hits)
        return LiveMemoryResult(
            hits=hits,
            hop_episode_ids=(ids,),
            query_cav_signature=(
                ()
                if query_signature is None
                else tuple(float(value) for value in query_signature.tolist())
            ),
        )

    def retrieve_residual(
        self,
        query: str,
        *,
        top_k: int = 4,
        cav_weight: float = 0.0,
        cav_mode: str = "positive",
        record_access: bool = True,
        association_layer: bool = False,
    ) -> LiveMemoryResult:
        """Retrieve semantic entry episodes before recursive graph expansion."""
        if association_layer and cav_weight:
            raise ValueError(
                "association-layer residual retrieval cannot use an entry-layer CAV"
            )
        torch = self.encoder._torch
        capture_layer = self.association_layer if association_layer else self.layer
        capture = self.encoder.capture(query, layer=capture_layer)
        query_residual = capture.residual.mean(dim=1)[0]
        query_signature = (
            None
            if self.cav_bank is None or association_layer
            else self.cav_bank.signature(capture.residual)
        )
        with torch.inference_mode():
            scores = self.store.residual_scores(
                query_residual,
                cav_weight=cav_weight,
                cav_mode=cav_mode,
                query_cav_signature=query_signature,
                association_layer=association_layer,
            )
            count = max(1, min(top_k, len(self.store.items)))
            selected_scores, selected_indices = torch.topk(scores, k=count)
        if record_access:
            self.store.touch(selected_indices)
        hits = tuple(
            LiveMemoryHit(
                episode_id=self.store.items[int(index)].episode_id,
                text=self.store.items[int(index)].text,
                score=float(score),
                access_count=self.store.items[int(index)].access_count,
                metadata=dict(self.store.items[int(index)].metadata),
            )
            for index, score in zip(
                selected_indices.tolist(), selected_scores.tolist(), strict=True
            )
        )
        ids = tuple(hit.episode_id for hit in hits)
        return LiveMemoryResult(
            hits=hits,
            hop_episode_ids=(ids,),
            query_cav_signature=(
                ()
                if query_signature is None
                else tuple(float(value) for value in query_signature.tolist())
            ),
        )

    def retrieve_associative(
        self,
        query: str,
        *,
        top_k: int = 4,
        seed_k: int = 2,
        hops: int = 1,
        cav_weight: float = 0.10,
    ) -> LiveMemoryResult:
        """Seed by CAV/residual similarity, then walk stored shared-context QK edges."""
        if hops < 0:
            raise ValueError("hops must be non-negative")
        seeds = self.retrieve_residual(
            query,
            top_k=min(seed_k, top_k),
            cav_weight=cav_weight,
            cav_mode="positive",
            record_access=False,
        )
        ranked, hop_ids = _rank_association_walk(
            self.graph,
            [(hit.episode_id, hit.score) for hit in seeds.hits],
            top_k=top_k,
            hops=hops,
        )

        by_id = {item.episode_id: item for item in self.store.items}
        selected_ids = {episode_id for episode_id, _ in ranked}
        selected_indices = [
            index
            for index, stored_item in enumerate(self.store.items)
            if stored_item.episode_id in selected_ids
        ]
        self._record_candidate_head_use(
            query,
            selected_indices,
            cav_weight=cav_weight,
        )
        hits = tuple(
            LiveMemoryHit(
                episode_id=episode_id,
                text=by_id[episode_id].text,
                score=score,
                access_count=by_id[episode_id].access_count,
                metadata=dict(by_id[episode_id].metadata),
            )
            for episode_id, score in ranked
        )
        return LiveMemoryResult(
            hits=hits,
            hop_episode_ids=hop_ids,
            query_cav_signature=seeds.query_cav_signature,
        )

    def prune(self, max_items: int, *, age_half_life: float = 100.0) -> list[str]:
        """Prune low-utility K/V entries and their association-graph edges."""
        removed = self.store.prune(max_items, age_half_life=age_half_life)
        self.graph.remove_episode_ids(removed)
        return removed


def run_smoke_benchmark(
    memory: QwenLiveHeadMemory,
    dataset_path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    benchmark_cav_weight = float(payload.get("cav_weight", 0.10))
    for episode in payload["memories"]:
        memory.write(
            episode["id"],
            episode["text"],
            importance=float(episode.get("importance", 0.0)),
            metadata={"source": str(dataset_path)},
        )

    associations = [tuple(pair) for pair in payload.get("associations", [])]
    diagnostics = memory.graph.calibrate_heads(associations, keep=4)
    diagnostics["directed_edge_count"] = memory.graph.edge_count
    diagnostics["entry_layer"] = memory.layer
    diagnostics["association_layer"] = memory.association_layer

    arms = {
        "residual": {"mode": "residual", "cav_weight": 0.0},
        "cav_residual": {
            "mode": "residual",
            "cav_weight": benchmark_cav_weight,
            "cav_mode": "positive",
        },
        "associative_cav_residual_qk": {
            "mode": "associative",
            "cav_weight": benchmark_cav_weight,
            "seed_k": 2,
            "hops": 1,
        },
        "direct_qk": {"hops": 1, "cav_weight": 0.0},
        "cav_qk": {"hops": 1, "cav_weight": 0.25, "cav_mode": "positive"},
        "recursive_cav_qk_ov": {
            "hops": 2,
            "cav_weight": 0.25,
            "cav_mode": "positive",
        },
    }
    results: dict[str, Any] = {}
    for arm, options in arms.items():
        rows: list[dict[str, Any]] = []
        recall_at_1 = 0
        recall_at_3 = 0
        for query in payload["queries"]:
            arm_options = dict(options)
            mode = arm_options.pop("mode", "head")
            if mode == "residual":
                result = memory.retrieve_residual(
                    query["text"], top_k=3, **arm_options
                )
            elif mode == "associative":
                result = memory.retrieve_associative(
                    query["text"], top_k=3, **arm_options
                )
            else:
                result = memory.retrieve(query["text"], top_k=3, **arm_options)
            ids = [hit.episode_id for hit in result.hits]
            recall_at_1 += int(query["answer_id"] in ids[:1])
            recall_at_3 += int(query["answer_id"] in ids[:3])
            rows.append(
                {
                    "query": query["text"],
                    "answer_id": query["answer_id"],
                    "retrieved": ids,
                    "hops": result.hop_episode_ids,
                    "cav_signature": result.query_cav_signature,
                }
            )
        count = len(payload["queries"])
        results[arm] = {
            "recall_at_1": recall_at_1 / count,
            "recall_at_3": recall_at_3 / count,
            "rows": rows,
        }
    results["_diagnostics"] = diagnostics
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cav-report", type=Path, required=True)
    parser.add_argument("--cav-vectors", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--association-layer", type=int)
    parser.add_argument("--concept", action="append", default=["binding_constraint"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = Qwen3PrefixEncoder(
        args.model_dir,
        layers=7,
        device="cuda",
        dtype="bfloat16",
    )
    bank = CAVBank.load(
        args.cav_report,
        args.cav_vectors,
        layer=args.layer,
        concepts=args.concept,
        device=encoder.device,
    )
    memory = QwenLiveHeadMemory(
        encoder,
        layer=args.layer,
        association_layer=args.association_layer,
        cav_bank=bank,
    )
    results = run_smoke_benchmark(memory, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for arm, result in results.items():
        if arm.startswith("_"):
            print(f"{arm}: {json.dumps(result)}")
            continue
        print(
            f"{arm}: recall@1={result['recall_at_1']:.3f}, "
            f"recall@3={result['recall_at_3']:.3f}"
        )
    print(f"result: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
