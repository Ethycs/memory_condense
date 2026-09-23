import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import assay_hot_retrieval_1m as assay


def _evidence(chunk_id: str, text: str) -> dict[str, object]:
    rendered = f"[2026-01-01T00:00:00+00:00 | user] {text}"
    return {
        "evidence_id": chunk_id,
        "chunk_id": chunk_id,
        "turn_id": f"turn-{chunk_id}",
        "source_id": f"source-{chunk_id}",
        "role": "user",
        "created_at": "2026-01-01T00:00:00+00:00",
        "route": "bm25",
        "score": "1",
        "raw_text": text,
        "raw_text_sha256": assay.quote_sha256(text),
        "rendered_text": rendered,
        "rendered_text_sha256": assay.quote_sha256(rendered),
    }


def test_raw_packet_is_provider_ready_and_keeps_exact_selected_prefix() -> None:
    evidence = [_evidence("a", "alpha fact"), _evidence("b", "beta fact")]

    packet, timings, provider_ready_at_ns = assay._pack_raw_evidence(  # noqa: SLF001
        evidence,
        prompt_question="What is the fact?",
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )

    assert packet["selected_chunk_ids"] == ["a", "b"]
    assert packet["packed_chunk_ids"] == ["a", "b"]
    assert packet["dropped_chunk_ids"] == []
    assert packet["raw_evidence_only"] is True
    assert packet["provider_messages"][-1]["content"].count("alpha fact") == 1
    assert packet["provider_messages"][-1]["content"].count("beta fact") == 1
    assert all(value >= 0 for value in timings.values())
    assert provider_ready_at_ns > 0
    assay._validate_arm_payload(  # noqa: SLF001
        packet,
        prompt_question="What is the fact?",
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )


def test_raw_packet_drops_only_a_ranked_tail_when_budget_is_exhausted() -> None:
    evidence = [_evidence("a", "alpha fact"), _evidence("b", "beta fact")]
    first_only, _, _ = assay._pack_raw_evidence(  # noqa: SLF001
        evidence[:1],
        prompt_question="What is the fact?",
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )

    packet, _, _ = assay._pack_raw_evidence(  # noqa: SLF001
        evidence,
        prompt_question="What is the fact?",
        max_context_tokens=first_only["context_token_proxy"],
        max_prompt_tokens=8_000,
    )

    assert packet["packed_chunk_ids"] == ["a"]
    assert packet["dropped_chunk_ids"] == ["b"]


def test_common_all_fit_packet_counts_the_complete_prompt_once(monkeypatch) -> None:
    evidence = [_evidence(str(index), f"fact {index}") for index in range(40)]
    calls = {"context": 0, "render": 0, "prompt": 0}
    real_context = assay._context_token_proxy  # noqa: SLF001
    real_render = assay.build_qa_prompt
    real_prompt = assay.count_chat_prompt_token_proxy

    def count_context(texts):
        calls["context"] += 1
        return real_context(texts)

    def render(question, texts):
        calls["render"] += 1
        return real_render(question, texts)

    def count_prompt(messages):
        calls["prompt"] += 1
        return real_prompt(messages)

    monkeypatch.setattr(assay, "_context_token_proxy", count_context)
    monkeypatch.setattr(assay, "build_qa_prompt", render)
    monkeypatch.setattr(assay, "count_chat_prompt_token_proxy", count_prompt)

    packet, _, _ = assay._pack_raw_evidence(  # noqa: SLF001
        evidence,
        prompt_question="What are the facts?",
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )

    assert packet["packed_chunk_ids"] == [str(index) for index in range(40)]
    assert calls == {"context": 1, "render": 1, "prompt": 1}


def test_provider_ready_marker_precedes_audit_hash_materialization(monkeypatch) -> None:
    evidence = [_evidence("a", "alpha fact")]
    clock = [0]
    real_sha256 = assay.hashlib.sha256

    def tick() -> int:
        clock[0] += 10
        return clock[0]

    def delayed_sha256(payload=b""):
        clock[0] += 1_000
        return real_sha256(payload)

    monkeypatch.setattr(assay.time, "perf_counter_ns", tick)
    monkeypatch.setattr(assay.hashlib, "sha256", delayed_sha256)

    _packet, _timings, provider_ready_at_ns = assay._pack_raw_evidence(  # noqa: SLF001
        evidence,
        prompt_question="What is the fact?",
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )

    assert provider_ready_at_ns == 40
    assert clock[0] == 1_040


def test_exported_plain_retrieval_query_is_separate_from_dated_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    question = SimpleNamespace(
        question_id="q1",
        question="What happened?",
        dated_question="Current date: 2026-01-03. What happened?",
    )
    sample = SimpleNamespace(questions=(question,), turns=())
    monkeypatch.setattr(assay, "load_original_population", lambda *_args: sample)
    monkeypatch.setattr(
        assay,
        "population_identity_sha256",
        lambda _sample: assay.EXPECTED_POPULATION_SHA256,
    )
    monkeypatch.setattr(
        assay,
        "population_identity_payload",
        lambda _sample: {"fixture": True},
    )
    monkeypatch.setattr(assay, "transcript_tokens", lambda _sample: 1_000_000)

    assay.export_probes(
        dataset=tmp_path / "dataset.json",
        split_manifest=tmp_path / "split.json",
        output_root=tmp_path,
    )

    payload = json.loads((tmp_path / assay.DEFAULT_PROBES_NAME).read_text("utf-8"))
    row = payload["questions"][0]
    assert (
        payload["retrieval_query_form"]
        == "plain_question_with_dated_responder_prompt"
    )
    assert row["retrieval_query"] == question.question
    assert row["prompt_question"] == question.dated_question
    assert row["retrieval_query"] != row["prompt_question"]


def test_cli_keeps_dataset_out_of_compile_run_and_replay(tmp_path: Path) -> None:
    parser = assay._parser()  # noqa: SLF001
    compile_args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "compile-base",
            "--source-selection",
            str(tmp_path / "source.json"),
        ]
    )
    run_args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "run",
            "--source-selection",
            str(tmp_path / "source.json"),
        ]
    )
    replay_args = parser.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "replay",
            "--source-selection",
            str(tmp_path / "source.json"),
        ]
    )

    assert not hasattr(compile_args, "dataset")
    assert not hasattr(run_args, "dataset")
    assert not hasattr(replay_args, "dataset")
    assert run_args.lane_budget == 8
    assert run_args.candidates_per_lane == 96
    assert run_args.repeats == assay.MIN_LATENCY_REPEATS

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-root",
                str(tmp_path),
                "run",
                "--source-selection",
                str(tmp_path / "source.json"),
                "--dataset",
                str(tmp_path / "gold.json"),
            ]
        )


def test_probe_guard_rejects_gold_fields() -> None:
    assay._assert_gold_free_probe_rows(  # noqa: SLF001
        [{"question_id": "q", "retrieval_query": "where?"}]
    )
    with pytest.raises(ValueError, match="forbidden fields"):
        assay._assert_gold_free_probe_rows(  # noqa: SLF001
            [{"question_id": "q", "answer": "leak"}]
        )


def test_temporal_event_search_filters_compiled_features_after_global_rank() -> None:
    class LexicalStub:
        def search(self, query, limit=100):
            assert query == "museums museum"
            assert limit == 3
            return [("assistant", 9.0), ("future", 8.0), ("visit", 7.0)]

    plan = assay.plan_temporal_enumeration(
        "What is the order of the six museums I visited from earliest to latest?"
    )
    metadata = {
        "assistant": {
            "event_first_person": False,
            "event_fixed_completed": False,
            "event_ed_verbs": [],
        },
        "future": {
            "event_first_person": True,
            "event_fixed_completed": False,
            "event_ed_verbs": ["planned"],
        },
        "visit": {
            "event_first_person": True,
            "event_fixed_completed": True,
            "event_ed_verbs": ["visited"],
        },
    }

    hits, elapsed = assay._timed_temporal_events(  # noqa: SLF001
        LexicalStub(),  # type: ignore[arg-type]
        plan,
        metadata,
    )

    assert [hit.chunk_id for hit in hits] == ["visit"]
    assert hits[0].route == "temporal_event"
    assert elapsed >= 0


def test_temporal_event_search_retains_tail_for_post_selection_refill() -> None:
    chunk_ids = [f"event-{index:02d}" for index in range(25)]

    class LexicalStub:
        def search(self, query, limit=100):
            assert query == "museums museum"
            assert limit == len(chunk_ids)
            return [(chunk_id, float(25 - index)) for index, chunk_id in enumerate(chunk_ids)]

    plan = assay.plan_temporal_enumeration(
        "What is the order of the six museums I visited from earliest to latest?"
    )
    metadata = {
        chunk_id: {
            "event_first_person": True,
            "event_fixed_completed": True,
            "event_ed_verbs": ["visited"],
        }
        for chunk_id in chunk_ids
    }

    hits, _elapsed = assay._timed_temporal_events(  # noqa: SLF001
        LexicalStub(),  # type: ignore[arg-type]
        plan,
        metadata,
        candidate_limit=plan.budget + 1,
    )
    union = assay.post_selection_lane_union(
        (
            assay.RankedEvidenceLane("bm25", 1, (hits[0],)),
            assay.RankedEvidenceLane("temporal_event", plan.budget, hits),
        ),
        evidence_id=lambda item: item.chunk_id,
    )

    assert len(hits) == plan.budget + 1
    assert union.lane_selections[1].unfilled_slots == 0
    assert union.lane_selections[1].refilled[-1].chunk_id == chunk_ids[-1]


def test_source_neighborhood_expands_selected_addresses_without_text() -> None:
    metadata = [
        assay.SourceChunkMetadata("before", "source", "turn-0", 0, 0),
        assay.SourceChunkMetadata("seed", "source", "turn-1", 1, 0),
        assay.SourceChunkMetadata("after", "source", "turn-2", 2, 0),
    ]
    index = assay.SourceNeighborhoodIndex(metadata)
    seed = assay.RankedChunkAddress("seed", 1.0, "bm25")

    hits, neighborhood, elapsed = assay._timed_source_neighborhood(  # noqa: SLF001
        index,
        [seed],
    )

    assert [hit.chunk_id for hit in hits] == ["before", "after"]
    assert all(hit.route == "source_neighborhood" for hit in hits)
    assert neighborhood.source_order == ("source",)
    assert [link.direction for link in neighborhood.links] == [
        "predecessor_turn",
        "successor_turn",
    ]
    assert elapsed >= 0


def test_dated_lookback_filters_every_ranked_lane_before_selection() -> None:
    plan = assay.plan_temporal_enumeration(
        "What is the order of concerts I attended in the past two months, "
        "starting from the earliest?"
    )
    window = assay.resolve_temporal_evidence_window(
        plan,
        "[Question asked at 2023/04/22 (Sat) 19:31]\nWhat happened?",
    )
    addresses = (
        assay.RankedChunkAddress("too-old", 3.0, "exact_dense"),
        assay.RankedChunkAddress("in-range", 2.0, "exact_dense"),
        assay.RankedChunkAddress("future", 1.0, "exact_dense"),
    )
    metadata = {
        "too-old": {"created_at": "2023-02-05T08:56:00-07:00"},
        "in-range": {"created_at": "2023-03-17T17:23:00-07:00"},
        "future": {"created_at": "2023-05-01T00:00:00-07:00"},
    }

    retained, excluded, elapsed = assay._filter_temporal_window(  # noqa: SLF001
        addresses,
        window,
        metadata,
    )

    assert [row.chunk_id for row in retained] == ["in-range"]
    assert excluded == ("too-old", "future")
    assert elapsed >= 0
