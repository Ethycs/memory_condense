from __future__ import annotations

import json

import pytest

from tools import assay_episode_segmentation_ablation as assay
from tools.matched_eval.artifacts import read_sealed_json


def test_fixture_is_small_representative_and_has_eight_query_types() -> None:
    chunks = assay._corpus()
    questions = assay._questions()

    assert len(chunks) == 25
    assert {row.source_id for row in chunks} == {"travel", "profile", "support"}
    assert len(questions) == 8
    assert len({row.query_type for row in questions}) == 8
    assert {row.role for row in chunks} == {"system", "user", "assistant"}


def test_user_micro_arm_uses_complete_user_led_exchanges() -> None:
    chunks = assay._corpus()
    episodes = assay._build_user_led(chunks, "artifact-user-micro")

    emitted = [span.chunk_id for episode in episodes for span in episode.evidence]
    assert sorted(emitted) == sorted(row.chunk_id for row in chunks)
    assert len(emitted) == len(set(emitted))
    assert len(episodes) == 13
    for episode in episodes:
        roles = [span.role for span in episode.evidence]
        if episode.boundary_method == "orphan_prelude":
            assert roles == ["system"]
        else:
            assert roles[0] == "user"
            assert roles.count("user") == 1


def test_hybrid_overlay_preserves_micros_and_partitions_them_by_id() -> None:
    chunks = assay._corpus()
    micro = assay._build_user_led(chunks, "artifact-micro")
    hybrid, groups = assay._build_hybrid(chunks, "artifact-macro")
    overlay_index = assay._build_arm("hybrid_overlay", chunks)
    overlay = assay._overlay_payload(overlay_index)

    assert [tuple(span.chunk_id for span in row.evidence) for row in hybrid] == [
        tuple(span.chunk_id for span in row.evidence) for row in micro
    ]
    members = [episode_id for group in groups for episode_id in group.member_episode_ids]
    assert len(members) == len(set(members)) == len(hybrid)
    assert set(members) == {row.episode_id for row in hybrid}
    assert overlay["format"] == assay.FORMAT + "-macro-overlay"
    assert overlay["gold_loaded"] is False
    assert overlay["links"]
    assert all(
        link["left_group_id"] != link["right_group_id"]
        for link in overlay["links"]
    )


def test_overlay_uses_raw_assistant_anchor_and_conditional_budgets() -> None:
    chunks = assay._corpus()
    index = assay._build_arm("hybrid_overlay", chunks)
    questions = {row.question_id: row for row in assay._questions()}

    serial = assay._query(index, questions["q-serial"], chunks)
    code = assay._query(index, questions["q-code"], chunks)
    ordered = assay._query(index, questions["q-order"], chunks)

    assert "pf-a-cobalt" in code["raw_anchor_chunk_ids"]
    assert serial["frontier_expansion"] is False
    assert serial["microepisode_cap"] == assay.SCALAR_MICRO_CAP
    assert len(serial["selected_episode_ids"]) <= assay.SCALAR_MICRO_CAP
    assert ordered["frontier_expansion"] is True
    assert ordered["microepisode_cap"] == assay.FRONTIER_MICRO_CAP
    assert len(ordered["selected_episode_ids"]) <= assay.FRONTIER_MICRO_CAP
    assert set(serial["raw_anchor_chunk_ids"]) <= set(serial["selected_chunk_ids"])
    assert serial["selected_chunk_ids"].index("sp-u-serial") < serial[
        "selected_chunk_ids"
    ].index("sp-a-serial")
    receipt = assay._query(index, questions["q-receipt"], chunks)
    assert receipt["selected_chunk_ids"].index("sp-u-receipt") < receipt[
        "selected_chunk_ids"
    ].index("sp-a-receipt")


def test_construct_is_gold_blind_and_enforces_all_selection_caps(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assay,
        "_gold",
        lambda: (_ for _ in ()).throw(AssertionError("gold loaded during selection")),
    )

    selection_sha, runtime_sha = assay.construct(tmp_path)
    selection = read_sealed_json(tmp_path / assay.SELECTION_NAME)
    runtime = read_sealed_json(tmp_path / assay.RUNTIME_NAME)
    overlay = read_sealed_json(tmp_path / assay.OVERLAY_NAME)

    assert selection.sha256 == selection_sha
    assert runtime.sha256 == runtime_sha
    assert selection.payload["gold_loaded"] is False
    assert selection.payload["macro_overlay_sha256"] == overlay.sha256
    serialized = json.dumps(selection.payload, sort_keys=True)
    assert "target_chunk_ids" not in serialized
    assert "answer_bearing_chunk_ids" not in serialized
    assert "accepted_roles" not in serialized
    for rows in selection.payload["arms"].values():
        assert len(rows) == 8
        for row in rows:
            if row.get("frontier_expansion"):
                assert len(row["selected_episode_ids"]) <= assay.FRONTIER_MICRO_CAP
            else:
                assert len(row["selected_episode_ids"]) <= assay.MAX_EPISODES
            assert len(row["selected_chunk_ids"]) <= assay.MAX_CHUNKS
            assert row["prompt_tokens"] <= assay.MAX_PROMPT_TOKENS


def test_evaluation_joins_gold_only_after_selection_and_replay_is_exact(
    tmp_path,
) -> None:
    selection_sha, _runtime_sha = assay.construct(tmp_path)
    evaluation_sha = assay.evaluate(tmp_path)
    replay_sha = assay.replay(tmp_path)
    evaluation = read_sealed_json(tmp_path / assay.EVALUATION_NAME)
    replay = read_sealed_json(tmp_path / assay.REPLAY_NAME)

    assert len(selection_sha) == len(evaluation_sha) == len(replay_sha) == 64
    assert evaluation.payload["gold_loaded_postseal"] is True
    assert evaluation.payload["selection_sha256"] == selection_sha
    assert replay.payload == {
        "format": assay.FORMAT + "-replay",
        "gold_loaded": False,
        "replay_equal": True,
        "selection_sha256": selection_sha,
    }
    for arm in assay.ARMS:
        means = evaluation.payload["arms"][arm]["means"]
        assert 0.0 <= means["evidence_recall"] <= 1.0
        assert 0.0 <= means["source_recall"] <= 1.0
        assert 0.0 <= means["target_source_evidence_recall"] <= 1.0
        assert 0.0 <= means["role_correctness"] <= 1.0
        assert 0.0 <= means["neighbor_closure"] <= 1.0
        assert means["prompt_tokens"] > 0


def test_runtime_separates_build_first_pass_and_warm_latency(tmp_path) -> None:
    assay.construct(tmp_path)
    runtime = read_sealed_json(tmp_path / assay.RUNTIME_NAME).payload

    assert "episode_build_and_representative_build_are_cold" in runtime["timing_method"]
    for metrics in runtime["arms"].values():
        assert metrics["episode_build_ns"] >= 0
        assert metrics["representative_build_ns"] >= 0
        assert metrics["query_first_pass_median_ns"] >= 0
        assert metrics["query_first_pass_p95_ns"] >= 0
        assert metrics["query_warm_median_ns"] >= 0
        assert metrics["query_warm_p95_ns"] >= 0
        assert metrics["query_samples"] == 8 * assay.WARM_REPETITIONS
