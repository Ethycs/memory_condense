"""Focused tests for explicit, bounded conversation-envelope hydration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.conversation_envelope_retrieval import (
    CONVERSATION_ENVELOPE_MEMBER_ROUTE,
    CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT,
    ConversationEnvelopeRetrievalExpansion,
    hydrate_conversation_envelope_plan,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult
from memory_condense.persistence.conversation_envelope_expansion import (
    CONVERSATION_ENVELOPE_EXPANSION_FORMAT,
    ConversationEnvelopeExpansionPlan,
)


class _Embedder:
    dim = 8

    def embed_query(self, _query: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vector = self.embed_query("").tolist()
        return [chunk.model_copy(update={"embedding": vector}) for chunk in chunks]


def _condenser(path) -> MemoryCondenser:
    return MemoryCondenser(
        data_dir=path,
        embedder=_Embedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
    )


def _raw(condenser: MemoryCondenser, chunk_id: str, *, score: float = 0.9):
    result = condenser.retriever.hydrate_chunk(
        chunk_id,
        score=score,
        route="hot_raw",
    )
    assert result is not None
    return result


def test_assistant_hit_hydrates_user_lead_first_and_seals_exact_rows(tmp_path) -> None:
    with _condenser(tmp_path / "lead") as condenser:
        (_user, user_chunks), (_assistant, assistant_chunks) = condenser.ingest_many(
            [
                ("user", "private lead marker", "session-a", None, "u1"),
                (
                    "assistant",
                    "private answer marker",
                    "session-a",
                    None,
                    "a1",
                ),
            ]
        )
        anchor = _raw(condenser, assistant_chunks[0].chunk_id)

        expanded = condenser.expand_conversation_envelopes([anchor])

        assert [row.chunk.chunk_id for row in expanded.results] == [
            user_chunks[0].chunk_id,
            assistant_chunks[0].chunk_id,
        ]
        assert expanded.results[0].route == CONVERSATION_ENVELOPE_MEMBER_ROUTE
        assert expanded.results[0].anchor_chunk_id == anchor.chunk.chunk_id
        assert expanded.results[1] is anchor
        assert expanded.original_chunk_ids == (anchor.chunk.chunk_id,)
        assert expanded.companion_chunk_ids == (user_chunks[0].chunk_id,)
        assert expanded.plan is not None
        assert expanded.plan.format == CONVERSATION_ENVELOPE_EXPANSION_FORMAT
        assert expanded.format == CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT
        assert "private lead marker" not in repr(expanded.plan)
        assert "private answer marker" not in repr(expanded.plan)

        replay = ConversationEnvelopeRetrievalExpansion(
            format=expanded.format,
            results=expanded.results,
            original_chunk_ids=expanded.original_chunk_ids,
            original_chunk_token_counts=expanded.original_chunk_token_counts,
            companion_chunk_ids=expanded.companion_chunk_ids,
            companion_token_count=expanded.companion_token_count,
            max_companion_chunks=expanded.max_companion_chunks,
            max_companion_tokens=expanded.max_companion_tokens,
            duplicate_input_chunk_ids=expanded.duplicate_input_chunk_ids,
            unhydrated_chunk_ids=expanded.unhydrated_chunk_ids,
            hydration_diagnostics=expanded.hydration_diagnostics,
            result_rows_sha256=expanded.result_rows_sha256,
            plan=expanded.plan,
            planner_failure_kind=expanded.planner_failure_kind,
            receipt_sha256=expanded.receipt_sha256,
        )
        assert replay == expanded


def test_two_raw_hits_keep_objects_but_user_lead_wins_group_order(tmp_path) -> None:
    with _condenser(tmp_path / "two-raw") as condenser:
        (_user, user_chunks), (_assistant, assistant_chunks) = condenser.ingest_many(
            [
                ("user", "ranked second", "session-b", None, "u2"),
                ("assistant", "ranked first", "session-b", None, "a2"),
            ]
        )
        assistant_hit = _raw(condenser, assistant_chunks[0].chunk_id, score=0.95)
        user_hit = _raw(condenser, user_chunks[0].chunk_id, score=0.8)

        expanded = condenser.expand_conversation_envelopes(
            [assistant_hit, user_hit, assistant_hit]
        )

        assert expanded.original_chunk_ids == (
            assistant_hit.chunk.chunk_id,
            user_hit.chunk.chunk_id,
        )
        assert expanded.duplicate_input_chunk_ids == (assistant_hit.chunk.chunk_id,)
        assert expanded.results == (user_hit, assistant_hit)
        assert expanded.results[0] is user_hit
        assert expanded.results[1] is assistant_hit
        assert expanded.companion_chunk_ids == ()
        assert expanded.plan is not None and len(expanded.plan.groups) == 1
        assert expanded.plan.groups[0].selected_turn_ids == ("u2", "a2")

        packed = condenser.build_context(
            "render the exchange",
            recent_turns=0,
            k_memories=0,
            k_expansions=0,
            use_consolidation=False,
            reheat_memories=False,
            expansion_results=expanded,
        )
        assert packed.expansion_chunk_ids == [
            user_hit.chunk.chunk_id,
            assistant_hit.chunk.chunk_id,
        ]
        rendered = "\n".join(packed.expansions)
        assert rendered.index("ranked second") < rendered.index("ranked first")


def test_truncation_is_nearest_selected_but_source_ordered_and_diagnostic(
    tmp_path,
) -> None:
    with _condenser(tmp_path / "nearest") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "lead", "session-c", None, "u3"),
                ("assistant", "older member", "session-c", None, "a3-old"),
                ("system", "near member", "session-c", None, "s3-near"),
                ("assistant", "anchor", "session-c", None, "a3-anchor"),
            ]
        )
        anchor = _raw(condenser, rows[-1][1][0].chunk_id)

        expanded = condenser.expand_conversation_envelopes(
            [anchor],
            max_turns_per_envelope=3,
        )

        assert expanded.plan is not None
        group = expanded.plan.groups[0]
        assert group.selected_turn_ids == ("u3", "s3-near", "a3-anchor")
        assert group.omitted_turn_count == 1
        assert group.omitted_chunk_count == 1
        assert "turn_bound" in group.truncation_reasons
        assert [row.chunk.turn_id for row in expanded.results] == [
            "u3",
            "s3-near",
            "a3-anchor",
        ]


def test_mandatory_bounds_fail_open_without_partial_group(tmp_path) -> None:
    with _condenser(tmp_path / "bounds") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "bounded lead", "session-d", None, "u4"),
                ("assistant", "bounded answer", "session-d", None, "a4"),
            ]
        )
        anchor = _raw(condenser, rows[-1][1][0].chunk_id)

        expanded = condenser.expand_conversation_envelopes(
            [anchor],
            max_companion_chunks=0,
        )

        assert expanded.results == (anchor,)
        assert expanded.results[0] is anchor
        assert expanded.companion_chunk_ids == ()
        assert expanded.plan is not None and expanded.plan.groups == ()
        assert [row.reason for row in expanded.plan.diagnostics] == [
            "mandatory_companion_chunk_or_token_bound"
        ]

        additive = condenser.expand_conversation_envelopes(
            [anchor],
            max_companion_chunks=1,
        )
        assert additive.results[1] is anchor
        assert len(additive.companion_chunk_ids) == 1
        assert additive.original_chunk_ids == (anchor.chunk.chunk_id,)


def test_hydration_failure_is_atomic_and_preserves_raw_baseline(
    tmp_path, monkeypatch
) -> None:
    with _condenser(tmp_path / "hydrate-failure") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "lead unavailable later", "session-e", None, "u5"),
                ("assistant", "raw anchor survives", "session-e", None, "a5"),
            ]
        )
        anchor = _raw(condenser, rows[-1][1][0].chunk_id)
        lead_chunk_id = rows[0][1][0].chunk_id
        original_hydrate = condenser.retriever.hydrate_chunk

        def fail_lead(chunk_id, **kwargs):
            if chunk_id == lead_chunk_id:
                return None
            return original_hydrate(chunk_id, **kwargs)

        monkeypatch.setattr(condenser.retriever, "hydrate_chunk", fail_lead)

        expanded = condenser.expand_conversation_envelopes([anchor])

        assert expanded.results == (anchor,)
        assert expanded.companion_chunk_ids == ()
        assert expanded.unhydrated_chunk_ids == (lead_chunk_id,)


def test_no_anchor_and_planner_failure_both_fail_open(tmp_path, monkeypatch) -> None:
    with _condenser(tmp_path / "fallback") as condenser:
        (_turn, chunks) = condenser.ingest(
            "assistant",
            "machine prelude",
            source_id="session-f",
            turn_id="prelude",
        )
        raw = _raw(condenser, chunks[0].chunk_id)

        no_anchor = condenser.expand_conversation_envelopes([raw])
        assert no_anchor.results == (raw,)
        assert no_anchor.plan is not None and no_anchor.plan.groups == ()
        assert no_anchor.plan.diagnostics[0].reason == "no_anchor:no_prior_user"

        def fail_plan(*_args, **_kwargs):
            raise RuntimeError("synthetic planner outage")

        monkeypatch.setattr(
            condenser._conversation_envelopes,
            "plan_retrieval_expansion",
            fail_plan,
        )
        failed = condenser.expand_conversation_envelopes([raw])
        assert failed.results == (raw,)
        assert failed.results[0] is raw
        assert failed.plan is None
        assert failed.planner_failure_kind == "builtins.RuntimeError"


def test_companions_do_not_recurse_or_train_as_independent_hits(tmp_path) -> None:
    with _condenser(tmp_path / "nonrecursive") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "lead", "session-g", None, "u7"),
                ("assistant", "answer", "session-g", None, "a7"),
            ]
        )
        anchor = _raw(condenser, rows[-1][1][0].chunk_id)
        expanded = condenser.expand_conversation_envelopes([anchor])
        companion = expanded.results[0]
        assert companion.route == CONVERSATION_ENVELOPE_MEMBER_ROUTE

        recursive = condenser.expand_conversation_envelopes([companion])
        assert recursive.results == (companion,)
        assert recursive.results[0] is companion
        assert recursive.plan is not None and recursive.plan.groups == ()

        packed = condenser.build_context(
            "pack explicitly",
            recent_turns=0,
            k_memories=0,
            k_expansions=0,
            use_consolidation=False,
            reheat_memories=False,
            expansion_results=expanded.results,
        )
        assert companion.chunk.chunk_id in packed.expansion_chunk_ids
        assert companion.chunk.chunk_id not in packed.direct_expansion_chunk_ids
        assert anchor.chunk.chunk_id in packed.direct_expansion_chunk_ids


def test_format_versions_are_receipt_bound(tmp_path) -> None:
    with _condenser(tmp_path / "format") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "format lead", "session-h", None, "u8"),
                ("assistant", "format answer", "session-h", None, "a8"),
            ]
        )
        expanded = condenser.expand_conversation_envelopes(
            [_raw(condenser, rows[-1][1][0].chunk_id)]
        )
        assert expanded.plan is not None

        plan_values = {
            name: getattr(expanded.plan, name)
            for name in expanded.plan.__dataclass_fields__
        }
        plan_values["format"] = "future-plan-format"
        try:
            ConversationEnvelopeExpansionPlan(**plan_values)
        except ValueError as error:
            assert "format" in str(error)
        else:
            raise AssertionError("an unknown plan format must be rejected")

        expansion_values = {
            name: getattr(expanded, name)
            for name in expanded.__dataclass_fields__
        }
        expansion_values["format"] = "future-retrieval-format"
        try:
            ConversationEnvelopeRetrievalExpansion(**expansion_values)
        except ValueError as error:
            assert "format" in str(error)
        else:
            raise AssertionError("an unknown retrieval format must be rejected")

        foreign_plan = replace(
            expanded.plan,
            original_chunk_token_counts=(),
            receipt_sha256="",
        )
        try:
            replace(expanded, plan=foreign_plan, receipt_sha256="")
        except ValueError as error:
            assert "original footprint" in str(error)
        else:
            raise AssertionError("a receipt cannot pair unrelated plan and results")

        different_bounds = replace(
            expanded.plan,
            max_companion_chunks=expanded.max_companion_chunks + 1,
            receipt_sha256="",
        )
        try:
            replace(expanded, plan=different_bounds, receipt_sha256="")
        except ValueError as error:
            assert "companion bounds" in str(error)
        else:
            raise AssertionError("a receipt cannot pair different plan/result bounds")

        try:
            replace(expanded, companion_token_count=True, receipt_sha256="")
        except ValueError as error:
            assert "companion token count" in str(error)
        else:
            raise AssertionError("a boolean is not a valid token footprint")


def test_multiple_out_of_order_anchors_emit_one_chronological_group(tmp_path) -> None:
    with _condenser(tmp_path / "multi-anchor") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "lead", "session-i", None, "u9"),
                ("assistant", "intervening", "session-i", None, "a9-middle"),
                ("system", "earlier anchor", "session-i", None, "s9-anchor"),
                ("assistant", "later anchor", "session-i", None, "a9-anchor"),
            ]
        )
        later = _raw(condenser, rows[3][1][0].chunk_id, score=0.95)
        earlier = _raw(condenser, rows[2][1][0].chunk_id, score=0.8)

        expanded = condenser.expand_conversation_envelopes(
            [later, earlier],
            max_turns_per_envelope=4,
        )

        assert [result.chunk.turn_id for result in expanded.results] == [
            "u9",
            "a9-middle",
            "s9-anchor",
            "a9-anchor",
        ]
        assert expanded.results[2] is earlier
        assert expanded.results[3] is later
        assert expanded.plan is not None and len(expanded.plan.groups) == 1
        group = expanded.plan.groups[0]
        assert group.selected_turn_ids == (
            "u9",
            "a9-middle",
            "s9-anchor",
            "a9-anchor",
        )
        assert group.selected_turn_ordinals == tuple(
            sorted(group.selected_turn_ordinals)
        )
        assert tuple(dict.fromkeys(group.ordered_chunk_turn_ids)) == (
            "u9",
            "a9-middle",
            "s9-anchor",
            "a9-anchor",
        )


def test_hydrated_turn_source_mismatch_rejects_group_atomically(
    tmp_path, monkeypatch
) -> None:
    with _condenser(tmp_path / "source-mismatch") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "lead", "session-j", None, "u10"),
                ("assistant", "anchor", "session-j", None, "a10"),
            ]
        )
        anchor = _raw(condenser, rows[1][1][0].chunk_id)
        lead_chunk_id = rows[0][1][0].chunk_id
        original_hydrate = condenser.retriever.hydrate_chunk

        def foreign_source(chunk_id, **kwargs):
            hydrated = original_hydrate(chunk_id, **kwargs)
            assert hydrated is not None and hydrated.turn is not None
            if chunk_id == lead_chunk_id:
                return hydrated.model_copy(
                    update={
                        "turn": hydrated.turn.model_copy(
                            update={"source_id": "foreign-session"}
                        )
                    }
                )
            return hydrated

        monkeypatch.setattr(condenser.retriever, "hydrate_chunk", foreign_source)

        expanded = condenser.expand_conversation_envelopes([anchor])

        assert expanded.results == (anchor,)
        assert expanded.unhydrated_chunk_ids == (lead_chunk_id,)
        assert [row.reason for row in expanded.hydration_diagnostics] == [
            "evidence_coordinate_mismatch"
        ]


def test_stale_plan_without_original_anchor_is_rejected_before_hydration(
    tmp_path,
) -> None:
    with _condenser(tmp_path / "stale-plan") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "target lead", "session-k1", None, "u11-target"),
                (
                    "assistant",
                    "target anchor",
                    "session-k1",
                    None,
                    "a11-target",
                ),
                ("user", "unrelated raw", "session-k2", None, "u11-other"),
            ]
        )
        target = _raw(condenser, rows[1][1][0].chunk_id)
        unrelated = _raw(condenser, rows[2][1][0].chunk_id)
        valid = condenser.expand_conversation_envelopes([target])
        assert valid.plan is not None
        stale = replace(
            valid.plan,
            original_chunk_token_counts=(
                (unrelated.chunk.chunk_id, unrelated.chunk.token_count),
            ),
            receipt_sha256="",
        )

        def hydration_must_not_run(*_args, **_kwargs):
            raise AssertionError("stale group attempted hydration")

        rejected = hydrate_conversation_envelope_plan(
            [unrelated],
            plan=stale,
            hydrate_chunk=hydration_must_not_run,
            live_chunk_ids=condenser._conversation_envelopes.live_chunk_ids,
            max_companion_chunks=stale.max_companion_chunks,
            max_companion_tokens=stale.max_companion_tokens,
        )

        assert rejected.results == (unrelated,)
        assert rejected.results[0] is unrelated
        assert rejected.companion_chunk_ids == ()
        assert [row.reason for row in rejected.hydration_diagnostics] == [
            "absent_original_anchor"
        ]


def test_companion_budget_is_additive_to_all_unrelated_raw_rows(tmp_path) -> None:
    with _condenser(tmp_path / "additive") as condenser:
        rows = condenser.ingest_many(
            [
                ("user", "target lead", "session-l1", None, "u12-target"),
                (
                    "assistant",
                    "target anchor",
                    "session-l1",
                    None,
                    "a12-target",
                ),
                ("user", "other raw", "session-l2", None, "u12-other"),
            ]
        )
        anchor = _raw(condenser, rows[1][1][0].chunk_id)
        unrelated = _raw(condenser, rows[2][1][0].chunk_id)

        expanded = condenser.expand_conversation_envelopes(
            [anchor, unrelated],
            max_companion_chunks=1,
        )

        assert expanded.original_chunk_ids == (
            anchor.chunk.chunk_id,
            unrelated.chunk.chunk_id,
        )
        assert expanded.original_chunk_token_counts == (
            anchor.chunk.token_count,
            unrelated.chunk.token_count,
        )
        assert len(expanded.companion_chunk_ids) == 1
        assert expanded.max_companion_chunks == 1
        assert expanded.results[1] is anchor
        assert expanded.results[-1] is unrelated
        assert len(expanded.results) == len(expanded.original_chunk_ids) + 1


def test_pending_envelope_assignment_keeps_exact_raw_result(tmp_path) -> None:
    with _condenser(tmp_path / "pending") as condenser:
        turn, chunks = condenser.capture(
            "user",
            "not indexed or assigned yet",
            source_id="session-m",
            turn_id="u13-pending",
        )
        raw = RetrievalResult(
            chunk=chunks[0],
            turn=turn,
            score=0.7,
            route="hot_raw",
        )

        expanded = condenser.expand_conversation_envelopes([raw])

        assert expanded.results == (raw,)
        assert expanded.results[0] is raw
        assert expanded.plan is not None and expanded.plan.groups == ()
        assert [row.reason for row in expanded.plan.diagnostics] == [
            "missing_or_pending_assignment"
        ]
