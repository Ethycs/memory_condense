from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from tools import assay_hot_v3_provider_free_witness as packet_tools
from tools import assay_hot_v4_user_envelope_shadow_full100 as assay
from tools.matched_eval.contracts import identity_sha256


def _evidence(index: int, *, route: str = "parent") -> dict[str, Any]:
    raw = f"memory fact {index}"
    rendered = f"[2026-01-{index + 1:02d}T00:00:00+00:00 | user] {raw}"
    digest = f"{index + 1:064x}"
    return {
        "chunk_id": digest,
        "created_at": f"2026-01-{index + 1:02d}T00:00:00+00:00",
        "evidence_id": digest,
        "raw_text": raw,
        "raw_text_sha256": assay.quote_sha256(raw),
        "rendered_text": rendered,
        "rendered_text_sha256": assay.quote_sha256(rendered),
        "role": "user",
        "route": route,
        "score": 1.0,
        "source_id": "source-a",
        "turn_id": f"turn-{index}",
    }


def _parent_row(ordinal: int) -> dict[str, Any]:
    dated = f"[Question asked at 2026-02-{ordinal + 1:02d}] What happened?"
    packed = _evidence(ordinal)
    arm, _audit = packet_tools._pack_ranked_raw_evidence(  # noqa: SLF001
        [packed],
        prompt_question=dated,
        max_context_tokens=assay.MAX_CONTEXT_TOKENS,
        max_prompt_tokens=assay.MAX_PROMPT_WORKSPACE_TOKENS,
    )
    # The selector must not see this broader, deliberately unsealed candidate.
    arm["selected_evidence"] = [packed, _evidence(20 + ordinal, route="secret")]
    body = {
        "effective_arm": arm,
        "ordinal": ordinal,
        "prompt_question_sha256": assay.quote_sha256(dated),
        "question_id": f"q{ordinal}",
    }
    return {**body, "row_receipt_sha256": identity_sha256(body)}


class _Projected:
    def __init__(self, projection: Mapping[str, Any], **fields: object) -> None:
        self._value = dict(projection)
        for name, value in fields.items():
            setattr(self, name, value)

    def projection(self) -> dict[str, Any]:
        return dict(self._value)


def _fixture_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[assay.RuntimeHooks, assay.ParentArtifacts, dict[str, Any]]:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 2)
    monkeypatch.setattr(
        assay,
        "_implementation_identity",
        lambda: {"format": "fixture-implementation", "sha256": "f" * 64},
    )
    parent_rows = {index: _parent_row(index) for index in range(2)}
    parent = assay.ParentArtifacts(
        construction={},
        construction_sha256=assay.EXPECTED_V4_CONSTRUCTION_SHA256,
        runtime_sha256=assay.EXPECTED_V4_RUNTIME_SHA256,
        replay_sha256=assay.EXPECTED_V4_REPLAY_SHA256,
        rows_by_ordinal=parent_rows,
    )
    namespace = SimpleNamespace(namespace_id="a" * 64)
    population_rows = tuple(
        SimpleNamespace(
            namespace=namespace,
            source=SimpleNamespace(
                packet=SimpleNamespace(
                    question_id=f"q{index}",
                    dated_question=(
                        f"[Question asked at 2026-02-{index + 1:02d}] What happened?"
                    ),
                )
            ),
        )
        for index in range(2)
    )
    context = SimpleNamespace(population=SimpleNamespace(rows=population_rows))
    calls: dict[str, Any] = {
        "build_cache": 0,
        "build_index": 0,
        "compose_caps": [],
        "selector_inputs": [],
    }
    cache = SimpleNamespace(cache_receipt_sha256="b" * 64)
    index = SimpleNamespace(receipt_sha256="c" * 64)
    budget_projection = {
        "max_companion_chunks": 16,
        "max_companion_tokens": 800,
        "max_envelopes": 4,
        "max_turns_per_envelope": 8,
    }

    def build_cache(_context: object, _namespace: object) -> tuple[object, Mapping[str, Any]]:
        calls["build_cache"] += 1
        return cache, {"cache_build_ns": 1}

    def build_index(value: object) -> object:
        assert value is cache
        calls["build_index"] += 1
        return index

    def select(
        value: object, parent_evidence: Sequence[Mapping[str, Any]], *, budget: object
    ) -> object:
        assert value is index
        assert budget.projection() == budget_projection
        assert len(parent_evidence) == 1
        assert parent_evidence[0]["route"] == "parent"
        calls["selector_inputs"].append(tuple(row["chunk_id"] for row in parent_evidence))
        companion_id = f"{50 + int(str(parent_evidence[0]['chunk_id']), 16):064x}"
        return _Projected(
            {
                "diagnostics": [],
                "format": "fixture-selection",
                "selected_companion_chunk_ids": [companion_id],
            },
            receipt_sha256="d" * 64,
            selected_companion_chunk_ids=(companion_id,),
            selected_companion_count=1,
            selected_companion_token_count=3,
        )

    def compose(
        value: object,
        selection: object,
        parent_evidence: Sequence[Mapping[str, Any]],
        measure_packet: object,
        *,
        max_context_tokens: int,
        max_prompt_tokens: int,
    ) -> object:
        assert value is index
        calls["compose_caps"].append((max_context_tokens, max_prompt_tokens))
        companion = _evidence(
            int(selection.selected_companion_chunk_ids[0], 16),
            route="user_envelope_companion",
        )
        companion["chunk_id"] = selection.selected_companion_chunk_ids[0]
        companion["evidence_id"] = companion["chunk_id"]
        packed = tuple([*parent_evidence, companion])
        context_tokens, workspace_tokens = measure_packet(packed)
        return _Projected(
            {
                "admitted_companion_chunk_ids": [companion["chunk_id"]],
                "format": "fixture-composition",
                "packed_chunk_ids": [row["chunk_id"] for row in packed],
            },
            admitted_companion_chunk_ids=(companion["chunk_id"],),
            admitted_companion_count=1,
            admitted_companion_token_count=3,
            context_token_count=context_tokens,
            packed_chunk_ids=tuple(row["chunk_id"] for row in packed),
            packed_evidence=packed,
            prompt_workspace_token_count=workspace_tokens,
            receipt_sha256="e" * 64,
            selected_companion_chunk_ids=(companion["chunk_id"],),
            selected_companion_token_count=3,
        )

    hooks = assay.RuntimeHooks(
        load_parent=lambda _root: parent,
        load_context=lambda *_args, **_kwargs: context,
        build_cache=build_cache,
        build_index=build_index,
        make_budget=lambda: _Projected(budget_projection),
        select=select,
        compose=compose,
    )
    return hooks, parent, calls


def test_construct_and_replay_are_gold_blind_parent_preserving_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks, _parent, calls = _fixture_hooks(monkeypatch)
    output = tmp_path / "fresh-shadow"
    construction_sha, runtime_sha = assay.construct(
        v4_root=tmp_path / "v4",
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path / "stores",
        output_root=output,
        hooks=hooks,
        clock=iter(range(10_000)).__next__,
    )
    construction, loaded_construction_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.CONSTRUCTION_NAME
    )
    runtime, loaded_runtime_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.RUNTIME_NAME
    )
    assert construction_sha == loaded_construction_sha
    assert runtime_sha == loaded_runtime_sha
    assert construction["gold_loaded"] is False
    assert construction["model_calls"] == construction["new_provider_calls"] == 0
    assert construction["aggregate"]["selected_companion_count"] == 2
    assert construction["aggregate"]["admitted_companion_count"] == 2
    assert construction["aggregate"]["all_parent_rows_protected"] is True
    assert construction["aggregate"]["strict_parent_anchor_rejection_count"] == 0
    assert runtime["processed_namespace_count"] == 1
    assert runtime["peak_resident_namespace_count"] == 1
    assert runtime["warm_aggregate"]["packet_materialization_mean_ns"] >= 0
    assert calls["build_cache"] == calls["build_index"] == 1
    assert calls["compose_caps"] == [(7000, 8000), (7000, 8000)]
    assert len(calls["selector_inputs"]) == 2
    for row in construction["questions"]:
        parent_id = row["parent_packed_chunk_ids"][0]
        assert parent_id in row["effective_arm"]["packed_chunk_ids"]
        assert row["effective_arm"]["prompt_workspace_token_proxy"] <= 8000

    replay_sha = assay.replay(
        v4_root=tmp_path / "v4",
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path / "stores",
        output_root=output,
        hooks=hooks,
    )
    replay, loaded_replay_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.REPLAY_NAME
    )
    assert replay_sha == loaded_replay_sha
    assert replay["byte_identical"] is True
    assert replay["gold_loaded"] is False
    # A second lifecycle rebuild still uses exactly one cache/index per namespace.
    assert calls["build_cache"] == calls["build_index"] == 2


def test_construct_requires_a_unique_absent_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks, _parent, _calls = _fixture_hooks(monkeypatch)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(ValueError, match="unique and absent"):
        assay.construct(
            v4_root=tmp_path,
            retrieval_path=tmp_path / "retrieval.json",
            store_root=tmp_path,
            output_root=occupied,
            hooks=hooks,
        )


def test_evaluate_is_the_only_gold_join_and_reports_all_monotone_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks, parent, _calls = _fixture_hooks(monkeypatch)
    output = tmp_path / "shadow"
    assay.construct(
        v4_root=tmp_path,
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path,
        output_root=output,
        hooks=hooks,
        clock=iter(range(10_000)).__next__,
    )
    assay.replay(
        v4_root=tmp_path,
        retrieval_path=tmp_path / "retrieval.json",
        store_root=tmp_path,
        output_root=output,
        hooks=hooks,
    )
    benchmark = tuple(
        SimpleNamespace(
            question_id=f"q{index}",
            dated_question=f"[Question asked at 2026-02-{index + 1:02d}] What happened?",
        )
        for index in range(2)
    )
    gold_reads: list[tuple[Path, Path]] = []

    def load_population(dataset: Path, split: Path) -> tuple[object, object, Mapping[str, Any]]:
        gold_reads.append((dataset, split))
        return benchmark, object(), {
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256
        }

    def score_arm(arm: Mapping[str, Any], _question: object) -> dict[str, Any]:
        improved = len(arm["packed_evidence"]) > 1
        return {
            "all_answer_value_components": improved,
            "all_gold_source_ids_reached": improved,
            "answer_value_component_metric_kind": "fixture",
            "answer_value_component_recall": 1.0 if improved else 0.5,
            "best_f1": 1.0 if improved else 0.5,
            "gold_source_id_recall": 1.0 if improved else 0.5,
            "literal_answer": improved,
            "packed_count": len(arm["packed_evidence"]),
        }

    monkeypatch.setattr(assay.v3, "_score_arm", score_arm)
    digest = assay.evaluate(
        dataset=tmp_path / "gold.json",
        split_manifest=tmp_path / "split.json",
        v4_root=tmp_path,
        output_root=output,
        population_loader=load_population,
        question_flattener=lambda samples: samples,
        parent_loader=lambda _root: parent,
    )
    evaluation, loaded = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.EVALUATION_NAME
    )
    assert digest == loaded
    assert gold_reads == [(tmp_path / "gold.json", tmp_path / "split.json")]
    assert evaluation["gold_loaded"] is True
    assert evaluation["new_provider_calls"] == evaluation["model_calls"] == 0
    aggregate = evaluation["aggregate"]
    assert aggregate["effective_all_gold_source_reach"] == 2
    assert aggregate["effective_literal_answer_hits"] == 2
    assert aggregate["effective_mean_best_f1"] == 1.0
    assert aggregate["effective_mean_answer_value_component_recall"] == 1.0
    assert aggregate["zero_prior_success_regressions"] is True
    assert aggregate["prior_success_regressions"] == {
        "best_f1": 0,
        "component": 0,
        "literal": 0,
        "source": 0,
    }


def test_parent_loader_reads_no_score_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    row = _parent_row(0)
    construction = {
        "format": "memory-condense-hot-v3-ordered-story-construction-v4",
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "question_count": 1,
        "questions": [row],
    }
    runtime = {
        "construction_sha256": "a" * 64,
        "format": "memory-condense-hot-v3-ordered-story-runtime-v4",
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
    }
    replay = {
        "construction_sha256": "a" * 64,
        "format": "memory-condense-hot-v3-ordered-story-replay-v4",
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "question_count": 1,
        "questions": [
            {"ordinal": 0, "row_receipt_sha256": row["row_receipt_sha256"]}
        ],
    }
    artifacts = {
        "construction.json": (construction, "a" * 64),
        "runtime.json": (runtime, "b" * 64),
        "replay.json": (replay, "c" * 64),
    }
    monkeypatch.setattr(assay, "EXPECTED_V4_CONSTRUCTION_SHA256", "a" * 64)
    monkeypatch.setattr(assay, "EXPECTED_V4_RUNTIME_SHA256", "b" * 64)
    monkeypatch.setattr(assay, "EXPECTED_V4_REPLAY_SHA256", "c" * 64)
    opened: list[str] = []

    def read(path: Path) -> tuple[dict[str, Any], str]:
        opened.append(path.name)
        return artifacts[path.name]

    monkeypatch.setattr(assay.hot, "_read_json_artifact", read)
    monkeypatch.setattr(assay.hot, "_validate_arm_payload", lambda *_a, **_k: None)
    loaded = assay._load_parent(Path("sealed"))
    assert loaded.construction_sha256 == "a" * 64
    assert opened == ["construction.json", "runtime.json", "replay.json"]


def test_component_regression_is_tri_state_when_parent_metric_is_undefined() -> None:
    common = {
        "best_f1": 0.5,
        "gold_source_id_recall": 1.0,
        "literal_answer": False,
    }
    parent = {**common, "answer_value_component_recall": None}
    effective = {**common, "answer_value_component_recall": None}
    assert assay._metric_regressions(parent, effective) == {
        "best_f1": False,
        "component": False,
        "literal": False,
        "source": False,
    }

    parent = {**common, "answer_value_component_recall": 1.0}
    assert assay._metric_regressions(parent, effective)["component"] is True
