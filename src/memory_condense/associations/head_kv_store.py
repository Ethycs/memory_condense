"""Bounded live per-head K/V workspace."""

from __future__ import annotations

import math
from typing import Any, Sequence

from memory_condense.associations.head_memory_models import HeadAddress, HeadMemoryItem
from memory_condense.domain.decay import decay_factor


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
        return HeadAddress(
            indices=selected_indices,
            aggregate_scores=selected_scores,
            head_weights=attention_weights,
            mixed_values=mixed_values,
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
