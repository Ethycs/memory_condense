from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.mem0_eval.typed_common_parent_compare import (
    CERTIFIED_V3_ADAPTER,
    EXACT_ACCOUNTING,
    MEM0_TYPED_ADAPTER,
    TERMINAL_V2_ADAPTER_STATUS,
    build_comparison_payload,
    build_terminal_v2_score_plane,
    load_certified_v3_treatment_score_plane,
    load_verified_score_plane,
    publish_score_plane,
    load_verified_v3_treatment_score_plane,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANSWER_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-reconciliation-v3"
)
JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/"
    "locked-specialist-final-reconciliation-sol-judge-v3"
)
ANSWER_RUN = ANSWER_ROOT / "locked-specialist-final-reconciliation-v3.json"
ANSWER_REPLAY = ANSWER_ROOT / "locked-specialist-final-reconciliation-replay-v3.json"
PREFLIGHT = JUDGE_ROOT / (
    "locked-specialist-final-reconciliation-sol-judge-preflight-v3.json"
)
JUDGE = JUDGE_ROOT / (
    "locked-specialist-final-reconciliation-semantic-judge-sol-v3.json"
)
JUDGE_REPLAY = JUDGE_ROOT / (
    "locked-specialist-final-reconciliation-semantic-judge-sol-replay-v3.json"
)
SCORE = JUDGE_ROOT / "locked-specialist-final-reconciliation-score-v3.json"
SCORE_REPLAY = JUDGE_ROOT / (
    "locked-specialist-final-reconciliation-score-replay-v3.json"
)


def _sha(path: Path) -> str:
    return read_sealed_json(path).sha256


def _reseal_plane(payload: dict[str, object]) -> None:
    rows = payload["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        unsigned = dict(row)
        unsigned.pop("score_plane_row_sha256", None)
        row["score_plane_row_sha256"] = identity_sha256(unsigned)
    correct = sum(bool(row["correct"]) for row in rows)
    payload["correct"] = correct
    payload["accuracy"] = correct / 100


def test_certified_v3_adapter_and_full100_comparator_rejects_forged_mem0(tmp_path):
    treatment_payload = load_certified_v3_treatment_score_plane(
        answer_run_path=ANSWER_RUN,
        answer_replay_path=ANSWER_REPLAY,
        expected_answer_run_sha256=_sha(ANSWER_RUN),
        expected_answer_replay_sha256=_sha(ANSWER_REPLAY),
        judge_output_root=JUDGE_ROOT,
        expected_preflight_sha256=_sha(PREFLIGHT),
        expected_judge_sha256=_sha(JUDGE),
        expected_judge_replay_sha256=_sha(JUDGE_REPLAY),
        expected_score_sha256=_sha(SCORE),
        expected_score_replay_sha256=_sha(SCORE_REPLAY),
    )
    assert treatment_payload["adapter_id"] == CERTIFIED_V3_ADAPTER
    assert treatment_payload["question_count"] == 100
    assert treatment_payload["correct"] == 89
    assert treatment_payload["exact_accounting"] == EXACT_ACCOUNTING
    treatment, treatment_replay = publish_score_plane(
        tmp_path,
        treatment_payload,
        arm_role="treatment",
    )
    verified_treatment = load_verified_v3_treatment_score_plane(
        treatment.path,
        treatment.sha256,
        treatment_replay.path,
        treatment_replay.sha256,
        answer_run_path=ANSWER_RUN,
        answer_replay_path=ANSWER_REPLAY,
        expected_answer_run_sha256=_sha(ANSWER_RUN),
        expected_answer_replay_sha256=_sha(ANSWER_REPLAY),
        judge_output_root=JUDGE_ROOT,
        expected_preflight_sha256=_sha(PREFLIGHT),
        expected_judge_sha256=_sha(JUDGE),
        expected_judge_replay_sha256=_sha(JUDGE_REPLAY),
        expected_score_sha256=_sha(SCORE),
        expected_score_replay_sha256=_sha(SCORE_REPLAY),
    )
    assert verified_treatment.artifact.sha256 == verified_treatment.replay.sha256
    assert len(verified_treatment.rows) == 100

    fixture_judge, _ = publish_sealed_json(
        tmp_path / "fixture-mem0-judge.json",
        {"format": "fixture-mem0-judge-v1", "question_count": 100},
    )
    fixture_judge_replay, _ = publish_sealed_json(
        tmp_path / "fixture-mem0-judge-replay.json",
        fixture_judge.payload,
    )
    fixture_score, _ = publish_sealed_json(
        tmp_path / "fixture-mem0-score.json",
        {"format": "fixture-mem0-score-v1", "question_count": 100},
    )
    fixture_score_replay, _ = publish_sealed_json(
        tmp_path / "fixture-mem0-score-replay.json",
        fixture_score.payload,
    )
    mem0_payload = copy.deepcopy(treatment_payload)
    mem0_payload["adapter_id"] = MEM0_TYPED_ADAPTER
    mem0_payload["arm_role"] = "mem0"
    # Syntactically bind a forged usage receipt; only the strict Mem0 adapter
    # can turn a real journal-derived receipt into a comparison capability.
    mem0_payload["usage_attestation_sha256"] = "c" * 64
    mem0_payload["source_artifacts"]["judge_run_sha256"] = fixture_judge.sha256
    mem0_payload["source_artifacts"]["judge_replay_sha256"] = (
        fixture_judge_replay.sha256
    )
    mem0_payload["source_artifacts"]["score_run_sha256"] = fixture_score.sha256
    mem0_payload["source_artifacts"]["score_replay_sha256"] = (
        fixture_score_replay.sha256
    )
    first_incorrect = next(
        row for row in mem0_payload["rows"] if row["correct"] is False
    )
    first_incorrect["correct"] = True
    first_incorrect["judge_row_sha256"] = hashlib.sha256(
        b"fixture-mem0-improvement"
    ).hexdigest()
    _reseal_plane(mem0_payload)
    mem0, mem0_replay = publish_score_plane(
        tmp_path,
        mem0_payload,
        arm_role="mem0",
    )
    # The retired generic reader must not certify a self-sealed flipped row
    # backed only by unrelated shallow files.
    with pytest.raises(MatchedEvalContractError, match="generic score-plane"):
        load_verified_score_plane(
            mem0.path,
            mem0.sha256,
            mem0_replay.path,
            mem0_replay.sha256,
        )
    with pytest.raises(MatchedEvalContractError, match="strict arm-specific"):
        build_comparison_payload(verified_treatment, mem0)

    partial_payload = copy.deepcopy(mem0.payload)
    partial_payload["rows"] = partial_payload["rows"][:-1]
    partial_payload["question_count"] = 99
    with pytest.raises(MatchedEvalContractError, match="not full100"):
        publish_score_plane(
            tmp_path / "partial",
            partial_payload,
            arm_role="mem0",
        )

    with pytest.raises(MatchedEvalContractError, match="envelope"):
        publish_score_plane(
            tmp_path / "wrong-role",
            treatment_payload,
            arm_role="mem0",
        )

    with pytest.raises(MatchedEvalContractError, match=TERMINAL_V2_ADAPTER_STATUS):
        build_terminal_v2_score_plane()
