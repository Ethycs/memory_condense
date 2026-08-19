"""Experimental Qwen-backed live head-memory workflow."""

from __future__ import annotations

from typing import Any, Sequence

from memory_condense.associations.cav_memory import CAVBank
from memory_condense.associations.head_association_graph import (
    HeadAssociationGraph,
    _rank_association_walk,
)
from memory_condense.associations.head_kv_store import HeadKVStore
from memory_condense.associations.head_memory_models import (
    HeadAddress,
    HeadMemoryItem,
    LiveMemoryHit,
    LiveMemoryResult,
)
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder


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
