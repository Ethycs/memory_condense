"""Bounded Qwen QK/OV inspection compiled into compact links."""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

from memory_condense.associations.cav_memory import CAVBank
from memory_condense.associations.head_association_graph import HeadAssociationGraph
from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
    MemoryLinkHit,
    MemoryLinkResult,
    NestedMemoryInspection,
)
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder


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

    def signatures(
        self,
        source_texts: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> tuple[tuple[float, ...], ...]:
        """Batch-compile memories to CAV coordinates and shed residuals.

        ``encode_layers`` returns only mean-pooled CPU vectors, so this path
        never constructs an attention map or retains per-token state.  The
        returned width is fixed by the CAV bank, independent of corpus size.
        """

        if self.cav_bank is None or not source_texts:
            return ()
        pooled = self.encoder.encode_layers(
            source_texts,
            layers=(self.cav_bank.layer,),
            batch_size=batch_size,
        )[self.cav_bank.layer]
        vectors = self.cav_bank.vectors.float().cpu()
        thresholds = self.cav_bank.thresholds.float().cpu()
        values = pooled.float() @ vectors.T - thresholds
        return tuple(
            tuple(float(value) for value in row.tolist())
            for row in values
        )

    def link(
        self,
        source_text: str,
        candidates: Sequence[AssociativeMemoryCandidate],
        *,
        top_k: int | None = None,
        include_transport_signature: bool = False,
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
        cav_capture = (
            captures[self.cav_bank.layer]
            if self.cav_bank is not None
            else None
        )
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
                transport_signature = None
                if include_transport_signature:
                    pooled_update = update.float().mean(dim=1)[0]
                    norm = pooled_update.square().sum().sqrt().clamp_min(1e-12)
                    transport_signature = (
                        (pooled_update / norm).to(dtype=torch.float16).cpu()
                    )
                metadata = dict(candidate.metadata)
                if self.cav_bank is not None and cav_capture is not None:
                    candidate_signature = self.cav_bank.signature(
                        cav_capture.residual[:, memory_tokens, :]
                    )
                    signature_values = tuple(
                        float(value)
                        for value in candidate_signature.cpu().tolist()
                    )
                    metadata.update(
                        {
                            "cav_signature": signature_values,
                            "active_cav_concepts": tuple(
                                name
                                for name, value in zip(
                                    self.cav_bank.names,
                                    signature_values,
                                    strict=True,
                                )
                                if value > 0.0
                            ),
                        }
                    )
                hits.append(
                    MemoryLinkHit(
                        episode_id=candidate.episode_id,
                        qk_score=float(strongest.mean()),
                        ov_transport=float(update.float().square().mean().sqrt()),
                        head_weights=tuple(
                            float(value) for value in head_weights.cpu().tolist()
                        ),
                        metadata=metadata,
                        transport_signature=transport_signature,
                    )
                )
        hits.sort(
            key=lambda hit: (hit.qk_score, hit.ov_transport, hit.episode_id),
            reverse=True,
        )
        count = len(hits) if top_k is None else max(1, min(int(top_k), len(hits)))
        signature: tuple[float, ...] = ()
        if self.cav_bank is not None and cav_capture is not None:
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

    def inspect_coverage(
        self,
        source_text: str,
        candidates: Sequence[AssociativeMemoryCandidate],
    ) -> MemoryLinkResult:
        """Batch independent query/candidate rows through the six-layer slice.

        Every row is ``[Memory] candidate [Question] query [Readout]``. Rows do
        not attend one another, so the resulting candidate contributions are
        invariant to frontier order. Layers below ``self.layer`` contextualize
        its readout; only the final prefix layer's QK block and OV direction are
        materialized. The reduced CPU vectors are transient and never written
        to the graph or memory store.
        """

        if self.cav_bank is not None:
            raise ValueError(
                "prefix coverage CAV projection is a separate ablation; "
                "construct this linker without a CAV bank"
            )
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
            raise ValueError("at least one unique coverage candidate is required")

        texts: list[str] = []
        memory_character_spans: list[tuple[int, int]] = []
        readout_character_spans: list[tuple[int, int]] = []
        for candidate in bounded:
            memory_prefix = "[Memory]\n"
            memory_body = f"{candidate.text}\n"
            memory_part = memory_prefix + memory_body
            question_part = f"[Question] {source_text}\n"
            readout_part = "[Readout]\n"
            texts.append(memory_part + question_part + readout_part)
            # The constant section marker is not candidate evidence. Excluding
            # it prevents the same label vector from dominating every OV
            # signature when candidate excerpts are short.
            memory_character_spans.append(
                (len(memory_prefix), len(memory_prefix) + len(memory_body))
            )
            readout_start = len(memory_part) + len(question_part)
            readout_character_spans.append(
                (readout_start, readout_start + len(readout_part))
            )

        original_padding_side = self.encoder.tokenizer.padding_side
        self.encoder.tokenizer.padding_side = "right"
        captured: dict[str, Any] = {}

        def save_attention_input(
            _module: Any,
            _args: Any,
            kwargs: dict[str, Any],
        ) -> None:
            captured["hidden"] = kwargs["hidden_states"]

        attention_module = self.encoder.model.layers[self.layer].self_attn
        handle = attention_module.register_forward_pre_hook(
            save_attention_input,
            with_kwargs=True,
        )
        try:
            tokenized = self.encoder.tokenizer(
                texts,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = tokenized.pop("offset_mapping")
            attention_mask = tokenized.get("attention_mask")
            row_tokens = [
                int(value) for value in attention_mask.sum(dim=1).tolist()
            ]
            kept_rows = 0
            sequence_length = 0
            for token_count in row_tokens:
                next_length = max(sequence_length, token_count)
                next_workspace = (kept_rows + 1) * next_length
                if next_workspace > self.max_workspace_tokens:
                    break
                kept_rows += 1
                sequence_length = next_length
            if kept_rows == 0:
                raise MemoryError(
                    f"coverage row needs {row_tokens[0]} tokens, above the "
                    f"hard cap of {self.max_workspace_tokens}; shrink the candidates"
                )
            # This cap measures the padded token positions concurrently active
            # in the prefix, not merely the longest row. Uninspected rows remain
            # recall-safe candidates in the caller.
            bounded = bounded[:kept_rows]
            offsets = offsets[:kept_rows, :sequence_length]
            tokenized = {
                key: value[:kept_rows, :sequence_length]
                for key, value in tokenized.items()
            }
            attention_mask = tokenized["attention_mask"]
            workspace_tokens = kept_rows * sequence_length
            model_inputs = {
                key: value.to(self.encoder.device)
                for key, value in tokenized.items()
            }
            with torch.inference_mode():
                self.encoder.model(**model_inputs, use_cache=False)
                hidden = captured["hidden"]
                batch, sequence_length, _width = hidden.shape
                input_shape = hidden.shape[:-1]
                head_shape = (*input_shape, -1, attention_module.head_dim)
                queries = attention_module.q_norm(
                    attention_module.q_proj(hidden).view(head_shape)
                ).transpose(1, 2)
                keys = attention_module.k_norm(
                    attention_module.k_proj(hidden).view(head_shape)
                ).transpose(1, 2)
                values = attention_module.v_proj(hidden).view(head_shape).transpose(1, 2)
                position_ids = torch.arange(
                    sequence_length,
                    device=self.encoder.device,
                ).unsqueeze(0).expand(batch, -1)
                cos, sin = self.encoder.model.rotary_emb(hidden, position_ids)
                queries, keys = self.encoder._apply_rotary_pos_emb(
                    queries,
                    keys,
                    cos,
                    sin,
                )
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
                expanded_keys = keys[:, key_groups]
                expanded_values = values[:, key_groups]

                hits: list[MemoryLinkHit] = []
                key_positions = torch.arange(
                    sequence_length,
                    device=self.encoder.device,
                )
                for row, candidate in enumerate(bounded):
                    row_offsets = offsets[row].tolist()

                    def token_span(character_span: tuple[int, int]) -> list[int]:
                        start, stop = character_span
                        return [
                            token_index
                            for token_index, (token_start, token_stop) in enumerate(
                                row_offsets
                            )
                            if token_stop > start and token_start < stop
                        ]

                    memory_tokens = token_span(memory_character_spans[row])
                    readout_tokens = token_span(readout_character_spans[row])
                    if not memory_tokens or not readout_tokens:
                        raise ValueError("tokenizer produced an empty coverage span")
                    query_states = queries[row, :, readout_tokens, :]
                    full_logits = torch.einsum(
                        "hqd,hkd->hqk",
                        query_states,
                        expanded_keys[row],
                    ) * float(attention_module.scaling)
                    readout_positions = torch.tensor(
                        readout_tokens,
                        device=self.encoder.device,
                    )
                    allowed = key_positions.unsqueeze(0) <= readout_positions.unsqueeze(1)
                    allowed &= model_inputs["attention_mask"][row].bool().unsqueeze(0)
                    masked_logits = full_logits.masked_fill(
                        ~allowed.unsqueeze(0),
                        torch.finfo(full_logits.dtype).min,
                    )
                    full_attention = masked_logits.float().softmax(dim=-1).to(
                        values.dtype
                    )
                    block = full_attention[:, :, memory_tokens]
                    head_weights = block.sum(dim=-1).mean(dim=-1)
                    memory_logits = full_logits[:, :, memory_tokens].float()
                    length_normalized_logits = (
                        torch.logsumexp(memory_logits, dim=-1)
                        - math.log(len(memory_tokens))
                    ).mean(dim=-1)
                    strongest = torch.topk(
                        length_normalized_logits,
                        k=min(self.head_vote_k, len(length_normalized_logits)),
                    ).values
                    candidate_values = expanded_values[row, :, memory_tokens, :]
                    moved_values = torch.einsum(
                        "hqs,hsd->qhd",
                        block.to(candidate_values.dtype),
                        candidate_values,
                    )
                    update = attention_module.o_proj(
                        moved_values.reshape(1, moved_values.shape[0], -1)
                    )
                    pooled_update = update.float().mean(dim=1)[0]
                    norm = pooled_update.square().sum().sqrt().clamp_min(1e-12)
                    hits.append(
                        MemoryLinkHit(
                            episode_id=candidate.episode_id,
                            qk_score=float(strongest.mean()),
                            ov_transport=float(
                                update.float().square().mean().sqrt()
                            ),
                            head_weights=tuple(
                                float(value)
                                for value in head_weights.cpu().tolist()
                            ),
                            metadata=dict(candidate.metadata),
                            transport_signature=(
                                (pooled_update / norm)
                                .to(dtype=torch.float16)
                                .cpu()
                            ),
                        )
                    )
        finally:
            handle.remove()
            self.encoder.tokenizer.padding_side = original_padding_side

        hits.sort(
            key=lambda hit: (hit.qk_score, hit.ov_transport, hit.episode_id),
            reverse=True,
        )
        return MemoryLinkResult(
            hits=tuple(hits),
            source_cav_signature=(),
            workspace_candidates=len(bounded),
            workspace_tokens=workspace_tokens,
            passes=1,
            total_candidate_inspections=len(bounded),
        )

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
            score = qk + math.log1p(max(0.0, float(hit.ov_transport)))
            signature = hit.metadata.get("cav_signature", ())
            if self.cav_bank is not None and signature:
                positive_margin = sum(
                    max(0.0, float(value)) for value in signature
                ) / len(signature)
                score += math.log1p(positive_margin)
            return score

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
                            metadata={**original.metadata, **hit.metadata},
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
