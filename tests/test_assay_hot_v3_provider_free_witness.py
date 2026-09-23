from __future__ import annotations

import copy
import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import assay_hot_v3_provider_free_witness as assay


def _evidence(
    chunk_id: str,
    source_id: str,
    text: str,
    *,
    role: str = "user",
) -> dict[str, Any]:
    created_at = "2026-09-06T00:00:00+00:00"
    rendered = f"[{created_at} | {role}] {text}"
    return {
        "evidence_id": chunk_id,
        "chunk_id": chunk_id,
        "turn_id": f"turn-{chunk_id}",
        "source_id": source_id,
        "role": role,
        "created_at": created_at,
        "route": "test",
        "score": 1.0,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
    }


def _arm(
    question: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    evidence = [copy.deepcopy(dict(row)) for row in rows]
    messages = assay.hot.build_qa_prompt(
        question, [str(row["rendered_text"]) for row in evidence]
    )
    payload = assay.hot._canonical_json_bytes(  # noqa: SLF001
        {"messages": messages}
    )
    prompt_tokens = assay.hot.count_chat_prompt_token_proxy(messages)
    return {
        "selected_evidence": evidence,
        "packed_evidence": copy.deepcopy(evidence),
        "selected_chunk_ids": [str(row["chunk_id"]) for row in evidence],
        "packed_chunk_ids": [str(row["chunk_id"]) for row in evidence],
        "dropped_chunk_ids": [],
        "context_token_proxy": assay.hot._context_token_proxy(  # noqa: SLF001
            [str(row["rendered_text"]) for row in evidence]
        ),
        "prompt_token_proxy": prompt_tokens,
        "prompt_workspace_token_proxy": (
            prompt_tokens + assay.OUTPUT_TOKEN_RESERVE
        ),
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "raw_evidence_only": True,
    }


def _receipts() -> dict[str, str]:
    return {
        "profile_preference": "1" * 64,
        "typed_witness": "2" * 64,
        "activated_turn_links": "3" * 64,
    }


def _excerpt_evidence(
    chunk_id: str,
    source_id: str,
    text: str,
    *,
    route: str,
) -> dict[str, Any]:
    return assay._raw_evidence_row(  # noqa: SLF001
        chunk_id=chunk_id,
        turn_id=f"turn-{chunk_id}",
        source_id=source_id,
        role="user",
        created_at="2026-09-06T00:00:00+00:00",
        raw_text=text,
        route=route,
        excerpt=True,
    )


def test_ordinal_parser_supports_reduced_and_full100() -> None:
    assert assay._parse_ordinals("7, 36,54") == (7, 36, 54)
    assert assay._parse_ordinals("all") == tuple(range(100))
    assert assay._parse_ordinals("full100") == tuple(range(100))
    with pytest.raises(ValueError, match="unique exact integers"):
        assay._parse_ordinals("7,7")
    with pytest.raises(ValueError, match="unique exact integers"):
        assay._normalize_ordinals((True,))


def test_composer_dedups_exact_occurrences_without_displacing_parent_raw() -> None:
    question = "[Question asked at 2026-09-06] What did I do?"
    parent = _arm(
        question,
        (
            _evidence("p1", "source-a", "parent a first"),
            _evidence("p1b", "source-a", "parent a remainder"),
            _evidence("p2", "source-b", "parent b"),
        ),
    )
    profile_excerpt = _excerpt_evidence(
        "p1", "source-a", "shared excerpt", route="profile"
    )
    typed_same_excerpt = _excerpt_evidence(
        "p1", "source-a", "shared excerpt", route="typed"
    )
    typed_novel = _excerpt_evidence(
        "typed", "source-c", "typed exact text", route="typed"
    )
    lane_rows = {
        "profile_preference": (
            profile_excerpt,
        ),
        "typed_witness": (
            typed_same_excerpt,
            typed_novel,
        ),
        "activated_turn_links": (
            _evidence("typed", "source-c", "typed exact text"),
            _evidence("linked", "source-d", "linked novel"),
        ),
    }

    effective, audit = assay.compose_successor_arm(
        parent_arm=parent,
        dated_question=question,
        lane_rows=lane_rows,
        lane_receipts=_receipts(),
    )

    assert audit["mode"] == "successor"
    assert audit["source_conservation_passed"] is True
    assert audit["lane_selected_before_dedup_ids"] == {
        "profile_preference": [profile_excerpt["chunk_id"]],
        "typed_witness": [
            typed_same_excerpt["chunk_id"],
            typed_novel["chunk_id"],
        ],
        "activated_turn_links": ["typed", "linked"],
    }
    assert audit["lane_retained_after_dedup_ids"] == {
        "profile_preference": [profile_excerpt["chunk_id"]],
        "typed_witness": [typed_novel["chunk_id"]],
        "activated_turn_links": ["linked"],
    }
    assert effective["packed_chunk_ids"] == [
        profile_excerpt["chunk_id"],
        typed_novel["chunk_id"],
        "linked",
        "p1",
        "p2",
        "p1b",
    ]
    assert effective["packed_evidence"][0]["raw_text"] == "shared excerpt"
    assert any(
        row["chunk_id"] == "p1" and row["raw_text"] == "parent a first"
        for row in effective["packed_evidence"]
    )
    exclusions = {
        (
            row["evidence_occurrence_sha256"],
            row["retained_lane"],
            row["excluded_lane"],
        )
        for row in audit["dedup_exclusions"]
    }
    assert (
        assay._evidence_occurrence_sha256(profile_excerpt),  # noqa: SLF001
        "profile_preference",
        "typed_witness",
    ) in exclusions
    assert (
        assay._evidence_occurrence_sha256(typed_novel),  # noqa: SLF001
        "typed_witness",
        "activated_turn_links",
    ) in exclusions
    assert not any(
        row["excluded_lane"] == "sealed_v3_parent"
        for row in audit["dedup_exclusions"]
    )
    assert audit["parent_representative_conservation_passed"] is True
    assert audit["post_selection_exact_evidence_occurrence_dedup"] is True
    assert effective["context_token_proxy"] <= 7_000
    assert effective["prompt_workspace_token_proxy"] <= 8_000


def test_binary_composer_is_legacy_equivalent_maximal_and_uses_fewer_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "[Question asked at 2026-09-06] What did I do?"
    parent_row = _evidence("p0", "source-a", "authoritative parent")
    parent = _arm(question, (parent_row,))
    candidates = [copy.deepcopy(parent_row)]
    candidates.extend(
        _evidence(
            f"c{index}",
            "source-a",
            f"candidate {index} " + "word " * 20,
        )
        for index in range(1, 128)
    )
    monkeypatch.setattr(assay, "MAX_CONTEXT_TOKENS", 500)

    effective, audit = assay.compose_successor_arm(
        parent_arm=parent,
        dated_question=question,
        lane_rows={
            "profile_preference": tuple(candidates),
            "typed_witness": (),
            "activated_turn_links": (),
        },
        lane_receipts=_receipts(),
    )

    legacy_calls = 0
    legacy_render = assay.hot._render_and_count_prompt  # noqa: SLF001

    def counted_legacy_render(*args: Any, **kwargs: Any):
        nonlocal legacy_calls
        legacy_calls += 1
        return legacy_render(*args, **kwargs)

    monkeypatch.setattr(
        assay.hot, "_render_and_count_prompt", counted_legacy_render
    )
    legacy, _timings, _ready_at = assay.hot._pack_raw_evidence(  # noqa: SLF001
        candidates,
        prompt_question=question,
        max_context_tokens=assay.MAX_CONTEXT_TOKENS,
        max_prompt_tokens=assay.MAX_PROMPT_TOKENS,
    )

    packing = audit["packing_audit"]
    assert audit["mode"] == "successor"
    assert effective == legacy
    assert 0 < packing["packed_count"] < packing["candidate_count"]
    assert packing["maximal_prefix_boundary_validated"] is True
    assert packing["sampled_monotonicity_validated"] is True
    assert packing["prompt_count_call_count"] < legacy_calls
    assert packing["context_count_call_count"] < legacy_calls
    assert packing["packed_count"] == len(effective["packed_evidence"])
    rejected = candidates[: packing["packed_count"] + 1]
    rejected_texts = [str(row["rendered_text"]) for row in rejected]
    rejected_context = assay.hot._context_token_proxy(rejected_texts)  # noqa: SLF001
    rejected_prompt = assay.hot.build_qa_prompt(question, rejected_texts)
    rejected_prompt_tokens = assay.hot.count_chat_prompt_token_proxy(
        rejected_prompt
    )
    assert (
        rejected_context > assay.MAX_CONTEXT_TOKENS
        or rejected_prompt_tokens + assay.OUTPUT_TOKEN_RESERVE
        > assay.MAX_PROMPT_TOKENS
    )


def test_non_excerpt_cannot_masquerade_as_parent_chunk_with_new_bytes() -> None:
    question = "[Question asked at 2026-09-06] What did I do?"
    parent = _arm(
        question,
        (_evidence("p1", "source-a", "authoritative parent bytes"),),
    )
    masquerader = _evidence("p1", "source-a", "different injected bytes")

    with pytest.raises(
        ValueError,
        match="non-excerpt successor changed bytes for a parent chunk ID",
    ):
        assay.compose_successor_arm(
            parent_arm=parent,
            dated_question=question,
            lane_rows={
                "profile_preference": (),
                "typed_witness": (),
                "activated_turn_links": (masquerader,),
            },
            lane_receipts=_receipts(),
        )


def test_authenticated_legacy_projection_collision_preserves_excerpt_and_raw() -> None:
    question = "[Question asked at 2026-09-06] What did I do?"
    projection = _evidence("p1", "source-a", "I rearranged the furniture.")
    projection["route"] = "activated_assertion_projection"
    projection["assertion_fact_receipt_sha256"] = "a" * 64
    parent = _arm(question, (projection,))
    hydrated = _evidence(
        "p1",
        "source-a",
        "Earlier context. I rearranged the furniture. Later context.",
    )
    hydrated["route"] = "hot_v3_activated_turn_links"

    effective, audit = assay.compose_successor_arm(
        parent_arm=parent,
        dated_question=question,
        lane_rows={
            "profile_preference": (),
            "typed_witness": (),
            "activated_turn_links": (hydrated,),
        },
        lane_receipts=_receipts(),
        variant=assay.AssayVariant(
            repair_legacy_parent_projection_collisions=True
        ),
    )

    assert effective["packed_chunk_ids"][0] == "p1"
    assert len(effective["packed_chunk_ids"]) == 2
    repaired = effective["packed_evidence"][1]
    assert repaired["chunk_id"] != "p1"
    assert repaired["backing_chunk_id"] == "p1"
    assert repaired["raw_text"] == "I rearranged the furniture."
    assert repaired["excerpt_occurrence"] is True
    assert repaired["legacy_parent_projection_identity_repaired"] is True
    assert audit["legacy_parent_projection_collision_repair_enabled"] is True
    assert audit["legacy_parent_projection_identity_repairs"] == [
        {
            "assertion_fact_receipt_sha256": "a" * 64,
            "backing_chunk_id": "p1",
            "legacy_parent_occurrence_sha256": (
                assay._evidence_occurrence_sha256(projection)  # noqa: SLF001
            ),
            "repaired_occurrence_id": repaired["chunk_id"],
        }
    ]


def test_compact_parent_without_turn_id_dedups_exact_hydrated_chunk() -> None:
    question = "[Question asked at 2026-09-06] What did I do?"
    parent_row = _evidence("p1", "source-a", "authoritative parent bytes")
    compact_parent_row = copy.deepcopy(parent_row)
    del compact_parent_row["turn_id"]
    parent = _arm(question, (compact_parent_row,))
    hydrated = copy.deepcopy(parent_row)
    hydrated["route"] = "hot_v3_activated_turn_links"

    effective, audit = assay.compose_successor_arm(
        parent_arm=parent,
        dated_question=question,
        lane_rows={
            "profile_preference": (),
            "typed_witness": (),
            "activated_turn_links": (hydrated,),
        },
        lane_receipts=_receipts(),
    )

    assert effective["packed_chunk_ids"] == ["p1"]
    assert effective["packed_evidence"][0]["raw_text"] == (
        "authoritative parent bytes"
    )
    assert audit["parent_representative_conservation_passed"] is True
    assert any(
        row["excluded_lane"] == "sealed_v3_parent"
        and row["retained_lane"] == "activated_turn_links"
        for row in audit["dedup_exclusions"]
    )


def test_source_loss_fails_back_to_byte_semantic_exact_parent() -> None:
    question = "[Question asked at 2026-09-06] What did I do?"
    parent = _arm(
        question,
        (
            _evidence("p1", "source-a", "parent a"),
            _evidence("p2", "source-b", "parent b"),
        ),
    )
    oversized = _evidence("huge", "source-new", "word " * 8_000)
    lane_rows = {
        "profile_preference": (oversized,),
        "typed_witness": (),
        "activated_turn_links": (),
    }

    effective, audit = assay.compose_successor_arm(
        parent_arm=parent,
        dated_question=question,
        lane_rows=lane_rows,
        lane_receipts=_receipts(),
    )

    assert audit["mode"] == "exact_parent_fallback"
    assert audit["source_conservation_passed"] is False
    assert len(audit["missing_parent_source_id_sha256s"]) == 2
    assert effective == parent
    assert effective["provider_payload_sha256"] == parent[
        "provider_payload_sha256"
    ]


def _parent_row(
    ordinal: int,
    question: str,
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "arms": {"a3_protected_union": _arm(question, evidence)},
        "ordinal": ordinal,
        "provider_packet": {"receipt_sha256": "e" * 64},
        "question_id": f"q-{ordinal}",
    }


def _profile_result(
    *,
    selected_chunk: str | None = None,
    source_id: str = "source-a",
) -> object:
    status = (
        "selected"
        if selected_chunk is not None
        else "not_recommendation_or_preference"
    )
    if selected_chunk is None:
        candidates: tuple[object, ...] = ()
        bindings: tuple[object, ...] = ()
    else:
        span = SimpleNamespace(
            chunk_id=selected_chunk,
            turn_id=f"turn-{selected_chunk}",
        )
        candidates = (
            SimpleNamespace(
                role="user",
                created_at="2026-09-06T00:00:00+00:00",
                quote=f"profile {selected_chunk}",
            ),
        )
        bindings = (
            SimpleNamespace(span=span, source_id=source_id),
        )
    return SimpleNamespace(
        audit=SimpleNamespace(status=status),
        candidates=candidates,
        local_bindings=bindings,
        receipt_sha256="4" * 64,
    )


def _typed_result() -> object:
    return SimpleNamespace(
        receipt=SimpleNamespace(receipt_sha256="5" * 64),
        selected_before_dedup=(),
        status="not_applicable",
    )


def _link_result() -> object:
    return SimpleNamespace(
        receipt=SimpleNamespace(receipt_sha256="6" * 64),
        selected_before_dedup=(),
        status="selected",
    )


def _synthetic_runtime(
    rows: Mapping[int, Mapping[str, Any]],
    *,
    calls: dict[str, list[Any]],
) -> assay.RuntimeHooks:
    namespace = SimpleNamespace(namespace_id="namespace-a")
    population_rows = tuple(
        SimpleNamespace(
            source=SimpleNamespace(
                packet=SimpleNamespace(question_id=row["question_id"])
            ),
            namespace=namespace,
        )
        for row in rows.values()
    )
    context = SimpleNamespace(
        population=SimpleNamespace(rows=population_rows)
    )

    def load_parent(_root: Path, *, ordinals: Sequence[int]):
        calls["load_parent"].append(tuple(ordinals))
        return {value: copy.deepcopy(dict(rows[value])) for value in ordinals}, (
            "a" * 64
        )

    def load_context(*_args: object, **_kwargs: object):
        calls["load_context"].append(True)
        return context

    def build_full(_context: object, built_namespace: object):
        calls["build_full"].append(built_namespace.namespace_id)
        physical = tuple(
            SimpleNamespace(chunk_id=f"p-{ordinal}") for ordinal in rows
        )
        return SimpleNamespace(rows=physical), {
            "cache_build_ns": 10,
            "window_index_build_ns": 20,
        }

    def select_profile(_index: object, question: str):
        calls["profile"].append(question)
        if "Q7" in question:
            return _profile_result(selected_chunk="p-7")
        return _profile_result()

    def build_typed(index: object):
        calls["build_typed"].append(id(index))
        return ("typed", id(index))

    def query_typed(
        typed_index: object,
        question: str,
        *,
        protected_chunk_ids: Sequence[str],
    ):
        calls["typed"].append(
            (typed_index, question, tuple(protected_chunk_ids))
        )
        return _typed_result()

    def build_links(index: object):
        calls["build_links"].append(id(index))
        return ("links", id(index))

    def query_links(
        link_index: object,
        seeds: Sequence[str],
        *,
        parent_chunk_ids: Sequence[str],
    ):
        calls["links"].append(
            (link_index, tuple(seeds), tuple(parent_chunk_ids))
        )
        return _link_result()

    return assay.RuntimeHooks(
        load_parent=load_parent,
        load_context=load_context,
        build_full_index=build_full,
        profile_applicable=lambda question: "Q7" in question,
        select_profile=select_profile,
        build_typed_index=build_typed,
        query_typed=query_typed,
        build_link_index=build_links,
        query_links=query_links,
    )


def _call_log() -> dict[str, list[Any]]:
    return {
        name: []
        for name in (
            "load_parent",
            "load_context",
            "build_full",
            "profile",
            "build_typed",
            "typed",
            "build_links",
            "links",
        )
    }


def test_reduced_construct_reuses_namespace_indexes_and_separates_timings(
    tmp_path: Path,
) -> None:
    q7 = "[Question asked at 2026-09-06] Q7?"
    q36 = "[Question asked at 2026-09-06] Q36?"
    rows = {
        7: _parent_row(
            7,
            q7,
            (
                _evidence("p-7", "source-a", "parent seven"),
                # This ID is intentionally absent from the physical index.
                _evidence("opaque-7", "source-meta", "sealed metadata"),
            ),
        ),
        36: _parent_row(
            36,
            q36,
            (_evidence("p-36", "source-b", "parent thirty six"),),
        ),
    }
    calls = _call_log()
    output = tmp_path / "out"
    ticks = itertools.count(step=10)

    construction_sha, runtime_sha = assay.construct(
        v3_root=tmp_path / "v3",
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path / "stores",
        output_root=output,
        ordinals=(36, 7),
        hooks=_synthetic_runtime(rows, calls=calls),
        clock=lambda: next(ticks),
    )

    construction, _ = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.CONSTRUCTION_NAME
    )
    runtime, _ = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.RUNTIME_NAME
    )
    assert construction["ordinals"] == [7, 36]
    assert construction["question_count"] == 2
    assert construction["new_provider_calls"] == 0
    assert construction["model_calls"] == 0
    assert not any(key.endswith("_ns") for key in construction["aggregate"])
    assert all(
        "parent_compact_row_sha256" in row
        and row["parent_provider_packet_receipt_sha256"] == "e" * 64
        for row in construction["questions"]
    )
    assay.assert_gold_blind(construction)
    assert calls["load_parent"] == [(7, 36)]
    assert calls["load_context"] == [True]
    assert calls["build_full"] == ["namespace-a"]
    assert len(calls["build_typed"]) == 1
    assert len(calls["build_links"]) == 1
    assert len(calls["profile"]) == 1
    assert all(row[2] == () for row in calls["typed"])
    # Specialist evidence seeds Q7. Q36 falls back to one physical parent.
    assert calls["links"][0][1:] == (
        ("p-7",),
        ("p-7", "opaque-7"),
    )
    assert calls["links"][1][1:] == (("p-36",), ("p-36",))
    assert runtime["resident_namespace_count"] == 1
    assert len(runtime["cold_setup"]["namespace_index_timings"]) == 1
    assert len(runtime["question_timings"]) == 2
    assert set(runtime["question_timings"][0]) >= {
        "profile_preference_ns",
        "profile_applicability_ns",
        "typed_witness_ns",
        "activated_turn_links_ns",
        "compose_ns",
        "compose_prompt_ready_ns",
        "compose_post_ready_validation_audit_ns",
        "total_ns",
    }
    assert set(runtime["warm_aggregate"]) >= {
        "compose_prompt_ready_mean_ns",
        "compose_prompt_ready_p95_ns",
        "compose_post_ready_validation_audit_mean_ns",
        "compose_post_ready_validation_audit_p95_ns",
    }

    second_calls = _call_log()
    second_construction_sha, second_runtime_sha = assay.construct(
        v3_root=tmp_path / "v3",
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path / "stores",
        output_root=tmp_path / "second-out",
        ordinals=(7, 36),
        hooks=_synthetic_runtime(rows, calls=second_calls),
        clock=lambda counter=itertools.count(step=100): next(counter),
    )
    assert second_construction_sha == construction_sha
    assert second_runtime_sha != runtime_sha


def test_replay_reconstructs_exact_semantic_bytes_with_zero_calls(
    tmp_path: Path,
) -> None:
    question = "[Question asked at 2026-09-06] Q7?"
    rows = {
        7: _parent_row(
            7,
            question,
            (_evidence("p-7", "source-a", "parent seven"),),
        )
    }
    calls = _call_log()
    output = tmp_path / "out"
    assay.construct(
        v3_root=tmp_path / "v3",
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path / "stores",
        output_root=output,
        ordinals=(7,),
        hooks=_synthetic_runtime(rows, calls=calls),
        clock=lambda counter=itertools.count(): next(counter),
    )
    parent_loads: list[tuple[int, ...]] = []

    def parent_loader(_root: Path, *, ordinals: Sequence[int]):
        parent_loads.append(tuple(ordinals))
        return {value: copy.deepcopy(rows[value]) for value in ordinals}, (
            "a" * 64
        )

    replay_sha = assay.replay(
        v3_root=tmp_path / "v3",
        output_root=output,
        parent_loader=parent_loader,
    )

    replayed, loaded_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.REPLAY_NAME
    )
    construction, construction_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.CONSTRUCTION_NAME
    )
    assert loaded_sha == replay_sha
    assert replayed["construction_sha256"] == construction_sha
    assert replayed["status"] == "exact_semantic_bytes_reconstructed"
    assert replayed["gold_loaded"] is False
    assert replayed["model_calls"] == 0
    assert replayed["new_provider_calls"] == 0
    assert parent_loads == [(7,)]
    assert replayed["questions"][0]["provider_payload_sha256"] == (
        construction["questions"][0]["effective_arm"][
            "provider_payload_sha256"
        ]
    )


def test_score_is_a_separate_posthoc_dataset_join(tmp_path: Path) -> None:
    q7 = "[Question asked at 2026-09-06] Q7?"
    q36 = "[Question asked at 2026-09-06] Q36?"
    rows = {
        7: _parent_row(
            7, q7, (_evidence("p-7", "source-a", "parent seven"),)
        ),
        36: _parent_row(
            36,
            q36,
            (_evidence("p-36", "source-b", "parent thirty six"),),
        ),
    }
    calls = _call_log()
    output = tmp_path / "out"
    assay.construct(
        v3_root=tmp_path / "v3",
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path / "stores",
        output_root=output,
        ordinals=(7, 36),
        hooks=_synthetic_runtime(rows, calls=calls),
        clock=lambda counter=itertools.count(): next(counter),
    )
    assert not (output / assay.SCORE_NAME).exists()

    questions = [
        SimpleNamespace(
            answer=f"missing answer {ordinal}",
            dated_question=(
                q7
                if ordinal == 7
                else q36
                if ordinal == 36
                else f"[Question asked at 2026-09-06] Q{ordinal}?"
            ),
            evidence_sources=(),
            question_id=f"q-{ordinal}",
        )
        for ordinal in range(100)
    ]
    phases: list[str] = []

    def population_loader(_dataset: Path, _split: Path):
        phases.append("gold_population")
        return "samples", (), {
            "population_identity_sha256": assay.typed.EXPECTED_POPULATION_SHA256
        }

    def parent_loader(_root: Path, *, ordinals: Sequence[int]):
        phases.append("sealed_parent")
        return {value: rows[value] for value in ordinals}, "a" * 64

    assay.score(
        dataset=tmp_path / "dataset.json",
        split_manifest=tmp_path / "split.json",
        v3_root=tmp_path / "v3",
        output_root=output,
        population_loader=population_loader,
        question_flattener=lambda _samples: questions,
        parent_loader=parent_loader,
    )

    score, _ = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.SCORE_NAME
    )
    construction, _ = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.CONSTRUCTION_NAME
    )
    assert construction["gold_loaded"] is False
    assay.assert_gold_blind(construction)
    assert score["gold_loaded"] is True
    assert score["aggregate"]["question_count"] == 2
    assert [row["ordinal"] for row in score["questions"]] == [7, 36]
    assert phases == ["gold_population", "sealed_parent"]
