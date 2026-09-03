from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.persistence.db import Database
from tests import test_confirmation_query_expansion_adapter as query_fixture
from tests.test_matched_eval_closure_live import _parent_plane
from tests.test_matched_eval_query_operator_refinement_live import _direct_plane
from tools import confirmation_query_expansion_adapter as confirmation_query
from tools import confirmation_source_streams as subject
from tools._routed_repair_routing import route_question
from tools.matched_eval import live
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    MAP_PARSE_FORMAT,
    VerifiedEvidenceMapPlane,
    VerifiedEvidenceMapRow,
    _ANSWER_KIND,
    build_evidence_map_plan,
    parse_evidence_map,
)
from tools.matched_eval.query_fact_adapter import build_query_fact_population
from tools.matched_eval.query_payload_live import build_query_payload_answer_plan
from tools.confirmation_query_payload_parent import VerifiedQueryExpansionArtifacts


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _create_complete_store(
    root: Path,
    index: int,
) -> tuple[Path, str, str, str, str]:
    """A real store containing both global and protected-source membership."""

    store = root / "combined-store"
    store.mkdir(parents=True)
    database_path = store / "memory.db"
    database = Database(database_path)
    global_source = f"partition-{index}::global-history"
    global_chunk = f"global-chunk-{index}"
    rows = [
        (
            f"global-turn-{index}",
            global_source,
            global_chunk,
            f"Global evidence for namespace {index}.",
        )
    ] + [
        (
            f"memory-turn-{semantic}",
            f"session-{semantic}",
            f"memory-chunk-{semantic}",
            f"Memory {semantic} is value {semantic}.",
        )
        for semantic in range(12)
    ]
    try:
        for ordinal, (turn_id, source_id, chunk_id, text) in enumerate(rows):
            database.execute(
                "INSERT INTO turns(turn_id, role, text, source_id, created_at, ordinal) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, "user", text, source_id, "2026-01-01T00:00:00Z", ordinal),
            )
            database.execute(
                "INSERT INTO chunks(chunk_id, turn_id, text, start_char, end_char, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, turn_id, text, 0, len(text), count_tokens(text)),
            )
        database.commit()
    finally:
        database.close()
    (store / "hnsw_index.bin").write_bytes(
        f"source-stream-index-{index}".encode("ascii")
    )
    return (
        store,
        file_sha256(database_path),
        file_sha256(store / "hnsw_index.bin"),
        global_source,
        global_chunk,
    )


def _query_plane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 3):
    monkeypatch.setattr(query_fixture, "_create_store", _create_complete_store)
    receipt_type = query_fixture.CombinedCumulativeStoreReceipt

    def complete_receipt(**values):
        values["turn_count"] = 13
        values["chunk_count"] = 13
        return receipt_type(**values)

    monkeypatch.setattr(
        query_fixture,
        "CombinedCumulativeStoreReceipt",
        complete_receipt,
    )
    semantics = tuple((index * 3 + 1) % 11 for index in range(count))
    namespace_sizes = (2, count - 2) if count > 2 else (count,)
    fixture, context, _checkpoints, coordinates = query_fixture._context(
        tmp_path / "fixture",
        semantics=semantics,
        namespace_sizes=namespace_sizes,
        prefix="source-streams",
    )
    output = fixture.root / "query-expansion"
    preflight = confirmation_query.preflight_confirmation_query_expansion(
        context,
        output_root=output,
    )
    release, _created = (
        confirmation_query.approve_confirmation_query_expansion_provider_release(
            context,
            output_root=output,
            expected_query_preflight_sha256=preflight.sha256,
            approve_provider_release=True,
            authorized_provider_calls=count,
        )
    )
    provider = confirmation_query.run_confirmation_query_expansion_provider(
        context,
        output_root=output,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=count,
        client=query_fixture._QueryClient(),
    )
    assert provider.physical_provider_calls == count
    retrievers = query_fixture._retrievers(context, coordinates)
    materialized = confirmation_query.materialize_confirmation_query_expansion(
        context,
        output_root=output,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        retrievers_by_namespace=retrievers,
    )
    replayed = confirmation_query.replay_confirmation_query_expansion(
        context,
        output_root=output,
        expected_query_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        retrievers_by_namespace=retrievers,
        expected_run_sha256=materialized.run_artifact.sha256,
        expected_runtime_ledger_sha256=materialized.runtime_ledger_artifact.sha256,
    )
    artifacts = VerifiedQueryExpansionArtifacts(
        preflight,
        materialized.run_artifact,
        read_sealed_json(output / confirmation_query.query_expansion.RUN_REPLAY_NAME),
        materialized.runtime_ledger_artifact,
        read_sealed_json(
            output / confirmation_query.query_expansion.RUNTIME_LEDGER_REPLAY_NAME
        ),
    )
    assert replayed.run_artifact.sha256 == artifacts.run.sha256
    return context, artifacts


def _map_plane(map_plan) -> VerifiedEvidenceMapPlane:
    rows = []
    for planned in map_plan.rows:
        packet = planned.direct_plan_row.adapter.source.packet
        if planned.submitted:
            alias = planned.aliases[-1]
            evidence = planned.retained_query_delta[-1]
            answer_kind = _ANSWER_KIND[planned.route.style]
            completion = json.dumps(
                {
                    "items": [
                        {
                            "alias": alias.alias,
                            "candidate": planned.direct_answer_row.prediction,
                            "citation": evidence.text,
                            "kind": answer_kind,
                        }
                    ]
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            parsed = parse_evidence_map(
                completion,
                answer_kind=answer_kind,
                evidence_text_by_alias={alias.alias: evidence.text},
            )
            status = "validated_items" if parsed.accepted_items else "no_valid_items"
            call_key = _digest(f"map-call:{planned.ordinal}")
            request = _digest(f"map-request:{planned.ordinal}")
            response = _digest(f"map-response:{planned.ordinal}")
        else:
            parsed = type("Parsed", (), {})()
            parsed.accepted_items = ()
            parsed.rejected_items = ()
            parsed.parse_receipt_sha256 = identity_sha256(
                {
                    "accepted_item_sha256s": [],
                    "format": MAP_PARSE_FORMAT,
                    "rejected_item_sha256s": [],
                }
            )
            answer_kind = None
            status = "not_submitted_state_chain"
            call_key = request = response = None
        rows.append(
            VerifiedEvidenceMapRow(
                ordinal=planned.ordinal,
                question_id=packet.question_id,
                question_sha256=packet.question_sha256,
                dated_question_sha256=packet.dated_question_sha256,
                route_id=planned.route.style.value,
                answer_kind=answer_kind,
                accepted_items=parsed.accepted_items,
                rejected_items=parsed.rejected_items,
                map_status=status,
                map_parse_receipt_sha256=parsed.parse_receipt_sha256,
                map_plan_row_receipt_sha256=planned.receipt_sha256,
                direct_parent_prediction_sha256=(
                    planned.direct_answer_row.prediction_sha256
                ),
                source_row_sha256=_digest(f"map-source:{planned.ordinal}"),
                runtime_row_id=_digest(f"map-runtime:{planned.ordinal}"),
                call_key_sha256=call_key,
                request_journal_sha256=request,
                response_journal_sha256=response,
            )
        )
    runtime = live._freeze_json({"source_stream_map_fixture": True})
    runtime_sha = hashlib.sha256(
        canonical_json_bytes(live._thaw_json(runtime))
    ).hexdigest()
    run_sha = _digest("source-stream-map-run")
    return VerifiedEvidenceMapPlane(
        run_sha256=run_sha,
        replay_sha256=run_sha,
        runtime_ledger_sha256=runtime_sha,
        runtime_ledger=runtime,
        parent_answer_run_sha256=map_plan.direct_plane.run_sha256,
        adapter_population_id=map_plan.direct_plan.adapter_population.population_id,
        retrieval_sha256=(
            map_plan.direct_plan.adapter_population.source_population.retrieval_sha256
        ),
        snapshot_id=map_plan.snapshot.snapshot_id,
        rows=tuple(rows),
        parent_plane=map_plan.direct_plane,
    )


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 3):
    context, artifacts = _query_plane(tmp_path, monkeypatch, count=count)
    source = context.population.source_population
    facts = build_query_fact_population(
        source,
        query_preflight=artifacts.preflight,
        query_run=artifacts.run,
        expected_retrieval_sha256=source.retrieval_sha256,
        expected_source_population_id=source.population_id,
        expected_query_preflight_sha256=artifacts.preflight.sha256,
        expected_query_run_sha256=artifacts.run.sha256,
        expected_query_population_id=context.population.population_id,
        expected_query_prompt_population_sha256=(
            context.population.prompt_population.prompt_population_sha256
        ),
    )
    direct_plan = build_query_payload_answer_plan(facts, _parent_plane(source))
    direct_plane = _direct_plane(direct_plan)
    direct_plane = replace(
        direct_plane,
        runtime_ledger_sha256=hashlib.sha256(
            canonical_json_bytes(live._thaw_json(direct_plane.runtime_ledger))
        ).hexdigest(),
    )
    map_plan = build_evidence_map_plan(direct_plan, direct_plane)
    return context, artifacts, map_plan, _map_plane(map_plan)


def _materialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 3):
    context, artifacts, map_plan, map_plane = _inputs(
        tmp_path, monkeypatch, count=count
    )
    output = tmp_path / "source-streams"
    result = subject.materialize_confirmation_source_streams(
        context,
        artifacts,
        map_plan,
        map_plane,
        output_root=output,
    )
    return context, artifacts, map_plan, map_plane, output, result


def test_arbitrary_n_real_sqlite_materialize_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, artifacts, map_plan, map_plane, output, result = _materialized(
        tmp_path, monkeypatch, count=3
    )

    assert context.question_count == 3
    assert result.physical_provider_calls == 0
    assert result.guided.physical_provider_calls == 0
    assert result.repack.physical_provider_calls == 0
    assert len(result.verified_base_rows) == 3
    assert len(result.base_population.questions) == 3
    assert result.query_map_adapter.obligation_compilation_mode == (
        subject.CONSOLIDATED_OBLIGATION_MODE
    )
    assert result.query_map_adapter.state_chain_profile == (
        subject.STATE_CHAIN_DIRECT_AUTHORITY_PROFILE
    )
    assert result.base_population.questions[0].plan.policy.policy_id.endswith(
        "d1-p0-g1-v1"
    )
    assert result.repack_population.direct_stream_profile.endswith("v2")
    # Exact S0 coordinates participate in selection and are removed only after.
    guided_rows = result.guided.run_artifact.payload["questions"]
    assert any(
        set(row["dedup_excluded_candidate_ids"])
        <= set(row["selected_before_dedup_candidate_ids"])
        and row["dedup_excluded_candidate_ids"]
        for row in guided_rows
    )

    replay = subject.replay_confirmation_source_streams(
        context,
        artifacts,
        map_plan,
        map_plane,
        output_root=output,
        expected_plane_sha256=result.plane_artifact.sha256,
    )
    assert replay.plane_artifact.sha256 == result.plane_artifact.sha256
    assert replay.base_population.receipt_sha256 == result.base_population.receipt_sha256
    assert replay.repack_population.receipt_sha256 == (
        result.repack_population.receipt_sha256
    )
    assert read_sealed_json(output / subject.PLANE_REPLAY_NAME).sha256 == (
        result.plane_artifact.sha256
    )


def test_question_only_route_eligibility_is_exact_union() -> None:
    direct = subject.question_only_partition_eligible(
        route_question(
            "[Question asked at 2026-02-01]\nWhat color was the notebook?"
        )
    )
    temporal = subject.question_only_partition_eligible(
        route_question(
            "[Question asked at 2026-02-01]\nWhat is my current notebook color?"
        )
    )
    complete = subject.question_only_partition_eligible(
        route_question(
            "[Question asked at 2026-02-01]\nWhat are all notebook colors?"
        )
    )
    assert (direct, temporal, complete) == (False, True, True)


def test_store_tamper_fails_before_source_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, artifacts, map_plan, map_plane, output, result = _materialized(
        tmp_path, monkeypatch, count=2
    )
    first = context.namespace_snapshots[0].store_dir / "hnsw_index.bin"
    first.write_bytes(b"tampered-after-freeze")
    with pytest.raises(
        confirmation_query.ConfirmationQueryExpansionError,
        match="index changed",
    ):
        subject.replay_confirmation_source_streams(
            context,
            artifacts,
            map_plan,
            map_plane,
            output_root=output,
            expected_plane_sha256=result.plane_artifact.sha256,
        )


def test_resealed_component_tamper_is_rejected_by_plane_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, artifacts, map_plan, map_plane, output, result = _materialized(
        tmp_path, monkeypatch, count=2
    )
    path = output / subject.ELIGIBILITY_NAME
    changed = copy.deepcopy(read_sealed_json(path).payload)
    changed["selection_policy"]["focus"] = "resealed-but-changed"
    raw = canonical_json_bytes(changed)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    with pytest.raises(subject.ConfirmationSourceStreamsError, match="eligibility artifact"):
        subject.replay_confirmation_source_streams(
            context,
            artifacts,
            map_plan,
            map_plane,
            output_root=output,
            expected_plane_sha256=result.plane_artifact.sha256,
        )


def test_source_stage_has_no_provider_or_validation_population_surface() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8").casefold()
    assert "expected_question_count" not in source
    assert "validation_question" not in source
    assert "authorized_provider_calls" not in source
    assert "enable_provider" not in source
    assert "import openai" not in source
    assert "import litellm" not in source
