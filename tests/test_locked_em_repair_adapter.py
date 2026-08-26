from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

import tools._locked_em_repair_adapter as adapter
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.fast_em_fact_memory import (
    build_em_fact_answer_prompt,
    episodic_neighborhood,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    answer_recall_guarded_cumulative_stage,
)
from tests.test_recall_guarded_cumulative_final_answer import (
    _Runtime,
    _digest,
    _synthetic_merged_retrieval,
)


def _sealed_population(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    retrieval = _synthetic_merged_retrieval()
    receipts = (
        {"receipt_sha256": "c" * 64},
        {"receipt_sha256": "9" * 64},
    )
    from memory_condense.eval import recall_guarded_cumulative_final_answer

    monkeypatch.setattr(
        recall_guarded_cumulative_final_answer,
        "merged_question_store_receipts",
        lambda _value: receipts,
    )
    retrieval_sha = _digest(retrieval)
    runtime = _Runtime(retrieval, retrieval_sha)
    baseline = answer_recall_guarded_cumulative_stage(
        retrieval,
        retrieval_sha256=retrieval_sha,
        runtime=runtime,
    )
    return retrieval, retrieval_sha, baseline, _digest(baseline)


def _publish_fixture(path: Path, value: object) -> str:
    raw = canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
        newline="",
    )
    return digest


def test_projects_validated_s0_s1_and_binds_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval, retrieval_sha, baseline, baseline_sha = _sealed_population(monkeypatch)

    population = adapter.build_locked_em_repair_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
        baseline_final_answers=baseline,
        baseline_final_answers_sha256=baseline_sha,
    )

    assert population.question_count == 2
    assert population.retrieval_sha256 == retrieval_sha
    assert population.baseline_final_answers_sha256 == baseline_sha
    assert len(population.binding_sha256) == 64
    first = population.rows[0]
    assert first.question.stage_ids == (
        "causal_graph_coverage_predecessor",
        "direct_episode_additions",
    )
    assert first.baseline.text == "Miss Bee Providore"
    assert first.baseline.text_sha256 == baseline["questions"][0]["answer"]["sha256"]
    root, delta = episodic_neighborhood(first.question)
    assert [row.evidence_id for row in root] == ["e0"]
    assert [row.evidence_id for row in delta] == ["e1"]

    compression = parse_fact_compression(
        first.question,
        '{"facts":[{"text":"Miss Bee Providore was chosen.",'
        '"citations":[{"evidence_alias":"E001",'
        '"quote":"Miss Bee Providore"}]}]}',
    )
    prompt = build_em_fact_answer_prompt(
        first.question,
        compression,
        arm="facts",
        policy="v2",
    )
    assert prompt.root_evidence_ids == ("e0",)
    assert prompt.selected_neighborhood_evidence_ids == ()
    assert prompt.dropped_neighborhood_evidence_ids == ("e1",)
    assert "Miss Bee Providore was chosen" in prompt.messages[1].content


def test_whole_population_preflight_is_provider_free_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval, retrieval_sha, baseline, baseline_sha = _sealed_population(monkeypatch)
    population = adapter.build_locked_em_repair_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
        baseline_final_answers=baseline,
        baseline_final_answers_sha256=baseline_sha,
    )

    result = adapter.preflight_locked_em_repair_population(population)

    assert result["question_count"] == 2
    assert result["memory_policy"] == "v2"
    assert result["answer_arm"] == "facts"
    assert result["root_evidence_rows"] == {
        "minimum": 1,
        "mean": 1.0,
        "maximum": 1,
        "total": 2,
    }
    assert result["post_selection_em_delta_rows"]["mean"] == 1.0
    assert result["post_selection_em_delta_rows"]["zero_delta_questions"] == 0
    prompts = result["compression_prompt_population"]
    assert prompts["logical_prompt_count"] == prompts["unique_prompt_count"] == 2
    assert prompts["max_prompt_token_proxy"] == 8_000
    assert result["compression_prompt_token_proxy"]["maximum"] < 8_000
    assert result["provider_calls"] == result["writes"] == 0
    assert result["gold_loaded"] is False
    assert result["dependent_answer_prompts_preflighted"] is False


def test_gold_analysis_field_is_rejected_before_sealed_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _synthetic_merged_retrieval()
    retrieval["questions"][0]["category"] = "multi-session"
    monkeypatch.setattr(
        adapter,
        "validate_final_answer_artifact",
        lambda *_args, **_kwargs: pytest.fail("gold firewall ran too late"),
    )

    with pytest.raises(adapter.LockedEMRepairAdapterError, match="forbidden"):
        adapter.build_locked_em_repair_population(
            retrieval,
            retrieval_sha256=_digest(retrieval),
            baseline_final_answers={},
            baseline_final_answers_sha256=_digest({}),
        )


def test_tampered_baseline_is_rejected_by_historical_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval, retrieval_sha, baseline, _baseline_sha = _sealed_population(monkeypatch)
    changed = copy.deepcopy(baseline)
    changed["questions"][0]["answer"]["text"] = "Another restaurant"

    with pytest.raises(ValueError, match="answer|binding|seal"):
        adapter.build_locked_em_repair_population(
            retrieval,
            retrieval_sha256=retrieval_sha,
            baseline_final_answers=changed,
            baseline_final_answers_sha256=_digest(changed),
        )


def test_file_loader_requires_canonical_bytes_digest_and_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval, _retrieval_sha, baseline, _baseline_sha = _sealed_population(monkeypatch)
    retrieval_path = tmp_path / "retrieval.json"
    baseline_path = tmp_path / "final-answers.json"
    retrieval_sha = _publish_fixture(retrieval_path, retrieval)
    baseline_sha = _publish_fixture(baseline_path, baseline)

    population = adapter.load_locked_em_repair_population(
        retrieval_path,
        expected_retrieval_sha256=retrieval_sha,
        baseline_final_answers_path=baseline_path,
        expected_baseline_final_answers_sha256=baseline_sha,
    )
    assert population.question_count == 2

    baseline_path.with_name("final-answers.json.sha256").write_text(
        f"{'0' * 64}  final-answers.json\n",
        encoding="ascii",
        newline="",
    )
    with pytest.raises(adapter.LockedEMRepairAdapterError, match="sidecar"):
        adapter.load_locked_em_repair_population(
            retrieval_path,
            expected_retrieval_sha256=retrieval_sha,
            baseline_final_answers_path=baseline_path,
            expected_baseline_final_answers_sha256=baseline_sha,
        )
