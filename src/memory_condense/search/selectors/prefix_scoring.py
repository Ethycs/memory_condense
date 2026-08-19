"""Prefix frontier inspection, scalar scoring, and event assignment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from memory_condense.domain._tokenizer import truncate_to_tokens
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.selectors.evidence_features import (
    _canonical_answer_object_key,
    _normalized_scalars,
    _normalized_transport,
    _optional_probability,
    _source_id,
    resolve_surface_value_evidence,
)
from memory_condense.search.selectors.prefix_models import (
    _PrefixAssignment,
    _PrefixEventCluster,
    _PreparedCoverage,
    _ScoredCoverage,
)

def score_prefix_coverage(
    self: Any,
    prepared: _PreparedCoverage,
) -> _ScoredCoverage | list[RetrievalResult]:
    """Inspect the frontier and assign transient event hypotheses."""

    started = prepared.started
    query = prepared.query
    program = prepared.program
    unique = prepared.unique
    timestamps = prepared.timestamps
    semantic_scores = prepared.semantic_scores
    active_partition_total = prepared.active_partition_total
    active_partition_inspected = prepared.active_partition_inspected
    normalized_scan_fields = prepared.normalized_scan_fields
    performance_event_keys_by_id = prepared.performance_event_keys_by_id
    effective_answerability = prepared.effective_answerability
    effective_membership = prepared.effective_membership
    surface_value_evidence = resolve_surface_value_evidence()

    attempted_candidates = 0
    inspected_candidates = 0
    frontier_batches = 0
    max_workspace_tokens = 0
    try:
        from memory_condense.associations.head_memory_models import (
            AssociativeMemoryCandidate,
        )

        linker_limit = int(getattr(self.linker, "max_candidates", 0))
        batch_limit = min(self.candidate_pool, linker_limit)
        if batch_limit < 1:
            raise ValueError("prefix linker max_candidates must be positive")
        hits: dict[str, Any] = {}
        query_text = truncate_to_tokens(query, self.query_tokens)
        cursor = 0
        while cursor < len(unique):
            batch = unique[cursor : cursor + batch_limit]
            by_id = {result.chunk.chunk_id: result for result in batch}
            inspectable = [
                AssociativeMemoryCandidate(
                    episode_id=result.chunk.chunk_id,
                    text=truncate_to_tokens(
                        (
                            f"[Source time: "
                            f"{timestamps[_source_id(result)]}]\n"
                            f"{result.chunk.text}"
                            if _source_id(result) in timestamps
                            else result.chunk.text
                        ),
                        self.candidate_tokens,
                    ),
                    score=float(result.score),
                    route=result.route or "coverage_frontier",
                    metadata={"source_id": _source_id(result)},
                )
                for result in batch
            ]
            linked = self.linker.inspect_coverage(query_text, inspectable)
            consumed = int(
                getattr(linked, "workspace_candidates", len(linked.hits))
            )
            if consumed < 1 or consumed > len(batch):
                raise ValueError(
                    "prefix linker reported an invalid workspace candidate count"
                )
            accepted_ids = {
                result.chunk.chunk_id for result in batch[:consumed]
            }
            for hit in linked.hits:
                if hit.episode_id in by_id and hit.episode_id in accepted_ids:
                    hits[hit.episode_id] = hit
            cursor += consumed
            attempted_candidates = cursor
            inspected_candidates += consumed
            frontier_batches += 1
            max_workspace_tokens = max(
                max_workspace_tokens,
                int(linked.workspace_tokens),
            )
    except Exception as exc:
        if self.strict:
            raise
        return self._fail_open(
            unique,
            program,
            started=started,
            reason=f"{type(exc).__name__}: {exc}",
            attempted=attempted_candidates,
            inspected=inspected_candidates,
            batches=frontier_batches,
            workspace_tokens=max_workspace_tokens,
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=normalized_scan_fields,
        )

    scored = [
        result for result in unique if result.chunk.chunk_id in hits
    ]
    qk_scores = _normalized_scalars(
        [float(hits[result.chunk.chunk_id].qk_score) for result in scored]
    )
    ov_scores = _normalized_scalars(
        [
            math.log1p(
                max(0.0, float(hits[result.chunk.chunk_id].ov_transport))
            )
            for result in scored
        ]
    )
    semantic_raw: list[float] = []
    semantic_kind_by_id: dict[str, str] = {}
    for result in scored:
        chunk_id = result.chunk.chunk_id
        supplied = (semantic_scores or {}).get(chunk_id)
        if supplied is not None and math.isfinite(float(supplied)):
            semantic_raw.append(float(supplied))
            semantic_kind_by_id[chunk_id] = "ms_marco_logit"
        else:
            semantic_raw.append(float(result.score))
            semantic_kind_by_id[chunk_id] = "retrieval_score"
    scalar_scores = _normalized_scalars(semantic_raw)
    surface_value_scores = [
        surface_value_evidence(
            result.chunk.text,
            timestamps.get(_source_id(result)),
        )
        for result in scored
    ]
    answerability_by_id = {
        result.chunk.chunk_id: _optional_probability(
            (effective_answerability or {}).get(result.chunk.chunk_id),
            "explicit_probability",
            "answerability",
            "answerability_probability",
            "probability",
            "score",
        )
        for result in scored
    }
    membership_by_id = {
        result.chunk.chunk_id: _optional_probability(
            (effective_membership or {}).get(result.chunk.chunk_id),
            "membership_probability",
            "member_probability",
            "answerability",
            "answerability_probability",
            "probability",
            "score",
        )
        for result in scored
    }
    value_scores = [
        (
            0.70 * answerability + 0.30 * surface
            if answerability is not None
            else surface
        )
        for result, surface in zip(
            scored,
            surface_value_scores,
            strict=True,
        )
        for answerability in [answerability_by_id[result.chunk.chunk_id]]
    ]
    score_by_id = {
        result.chunk.chunk_id: 0.80 * (
            0.55 * membership + 0.45 * prefix_member
            if membership is not None
            else prefix_member
        )
        + 0.20 * value
        for result, qk, ov, scalar, value in zip(
            scored,
            qk_scores,
            ov_scores,
            scalar_scores,
            value_scores,
            strict=True,
        )
        for prefix_member in [0.25 * scalar + 0.40 * qk + 0.35 * ov]
        for membership in [membership_by_id[result.chunk.chunk_id]]
    }
    value_by_id = {
        result.chunk.chunk_id: value
        for result, value in zip(scored, value_scores, strict=True)
    }
    semantic_raw_by_id = {
        result.chunk.chunk_id: value
        for result, value in zip(scored, semantic_raw, strict=True)
    }
    canonical_answer_object_keys_by_id = {
        result.chunk.chunk_id: _canonical_answer_object_key(
            query,
            result.chunk.text,
        )
        for result in scored
    }
    answer_object_keys_by_id = dict(canonical_answer_object_keys_by_id)
    # Performance identities are transient partition labels, not semantic
    # text.  Apply the key to every direct row, not only the representative:
    # exact recaps then contract even when HSC/baseline routing reintroduces
    # one outside the typed scan frontier.
    for chunk_id, event_key in performance_event_keys_by_id.items():
        if chunk_id in answer_object_keys_by_id:
            answer_object_keys_by_id[chunk_id] = (
                f"performance-event:{event_key}"
            )

    clusters: list[_PrefixEventCluster] = []
    uncertain: list[tuple[int, RetrievalResult]] = []
    posterior_uncertain_rows: list[_PrefixAssignment] = []
    null_rows: list[_PrefixAssignment] = []
    existing_count = 0
    new_count = 0
    expected_width: int | None = None
    for index, result in enumerate(unique):
        hit = hits.get(result.chunk.chunk_id)
        vector = _normalized_transport(
            getattr(hit, "transport_signature", None)
        )
        if vector is None:
            uncertain.append((index, result))
            continue
        if expected_width is None:
            expected_width = int(vector.size)
        elif vector.size != expected_width:
            # A malformed backend row must not turn recall-safe selection
            # into a shape error during pairwise comparison.
            uncertain.append((index, result))
            continue
        source_id = _source_id(result)
        timestamp = timestamps.get(source_id)
        answer_object_key = answer_object_keys_by_id.get(
            result.chunk.chunk_id
        )
        temporal_in_scope = self._timestamp_in_scope(program, timestamp)
        quality = score_by_id.get(result.chunk.chunk_id, 0.0)
        (
            p_existing,
            p_new,
            p_null,
            existing_energy,
            new_energy,
            null_energy,
            best_index,
            best_similarity,
            best_threshold,
        ) = self._cluster_posterior(
            quality=quality,
            vector=vector,
            source_id=source_id,
            timestamp=timestamp,
            program=program,
            clusters=clusters,
            temporal_in_scope=temporal_in_scope,
            answer_object_key=answer_object_key,
        )
        performance_identity = performance_event_keys_by_id.get(
            result.chunk.chunk_id
        )
        if performance_identity is not None:
            # A non-empty typed key is a deterministic equality relation:
            # equal keys merge despite vector variance; conflicting keys
            # remain separate.  A keyless direct row takes the ordinary
            # uncertain/null path and therefore stays fail-open.
            exact_key = f"performance-event:{performance_identity}"
            exact_clusters = [
                cluster_index
                for cluster_index, cluster in enumerate(clusters)
                if cluster.answer_object_keys == {exact_key}
            ]
            best_index = exact_clusters[0] if len(exact_clusters) == 1 else None
        aggregate = [p_existing, p_new, p_null]
        entropy = -sum(
            value * math.log(max(value, 1e-12))
            for value in aggregate
            if value > 0.0
        ) / math.log(3.0)
        surprisal = -math.log(max(1e-12, 1.0 - p_new))
        if performance_identity is not None:
            hypothesis = "existing" if best_index is not None else "new"
        elif entropy >= self.uncertainty_entropy:
            hypothesis = "uncertain"
        elif p_null >= self.null_threshold:
            hypothesis = "null"
        elif best_index is not None:
            hypothesis = "existing"
        else:
            hypothesis = "new"
        assignment = _PrefixAssignment(
            index=index,
            result=result,
            quality=quality,
            value_evidence=value_by_id.get(result.chunk.chunk_id, 0.0),
            membership_score=membership_by_id.get(result.chunk.chunk_id),
            vector=vector,
            p_existing=p_existing,
            p_new=p_new,
            p_null=p_null,
            existing_energy=existing_energy,
            new_energy=new_energy,
            null_energy=null_energy,
            temporal_in_scope=temporal_in_scope,
            entropy=entropy,
            semantic_surprisal=surprisal,
            hypothesis=hypothesis,
            existing_cluster=best_index,
            merge_similarity=best_similarity,
            merge_threshold=best_threshold,
        )
        if hypothesis == "uncertain":
            # Entropy is a control decision, not merely a counter. Do not
            # let an unresolved row create, merge, or reserve an event;
            # retain it in stable fail-open order immediately after the
            # credible coverage representatives.
            posterior_uncertain_rows.append(assignment)
            continue
        if hypothesis == "null":
            null_rows.append(assignment)
            continue
        if best_index is None:
            clusters.append(
                _PrefixEventCluster(
                    prototype=vector,
                    vectors=[vector],
                    members=[assignment],
                    source_ids={source_id},
                    timestamps={timestamp} if timestamp else set(),
                    answer_object_keys=(
                        {answer_object_key} if answer_object_key else set()
                    ),
                )
            )
            new_count += 1
            continue
        cluster = clusters[best_index]
        cluster.vectors.append(vector)
        cluster.members.append(assignment)
        cluster.source_ids.add(source_id)
        if timestamp:
            cluster.timestamps.add(timestamp)
        if answer_object_key:
            cluster.answer_object_keys.add(answer_object_key)
        prototype = cluster.prototype + vector
        cluster.prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
        existing_count += 1

    return _ScoredCoverage(
        prepared=prepared,
        attempted_candidates=attempted_candidates,
        inspected_candidates=inspected_candidates,
        frontier_batches=frontier_batches,
        max_workspace_tokens=max_workspace_tokens,
        hits=hits,
        semantic_kind_by_id=semantic_kind_by_id,
        answerability_by_id=answerability_by_id,
        membership_by_id=membership_by_id,
        score_by_id=score_by_id,
        value_by_id=value_by_id,
        semantic_raw_by_id=semantic_raw_by_id,
        canonical_answer_object_keys_by_id=canonical_answer_object_keys_by_id,
        answer_object_keys_by_id=answer_object_keys_by_id,
        clusters=clusters,
        uncertain=uncertain,
        posterior_uncertain_rows=posterior_uncertain_rows,
        null_rows=null_rows,
        existing_count=existing_count,
        new_count=new_count,
    )
