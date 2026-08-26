from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import memory_condense.eval.linear_case_ledger as ledger_module
from memory_condense.eval.linear_case_ledger import (
    CAV_ANSWER_FORMAT,
    HEBBIAN_ANSWER_FORMAT,
    SYNTHESIS_FORMAT,
    SYNTHESIS_SCORE_FORMAT,
    ArtifactSpec,
    CaseLedgerValidationError,
    augment_case_ledger_posthoc,
    build_gold_blind_case_ledger,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    RETRIEVAL_FORMAT,
    STAGE_IDS,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _quote(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _publish(tmp_path: Path, name: str, payload: dict) -> tuple[Path, str]:
    path = tmp_path / name
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return path, digest


@dataclass(frozen=True)
class _Stage:
    stage_id: str
    prompt_messages_sha256: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Question:
    ordinal: int
    question_id: str
    question_sha256: str
    question: str
    stages: tuple[_Stage, ...]

    def stage(self, stage_id: str) -> _Stage:
        return next(stage for stage in self.stages if stage.stage_id == stage_id)


@dataclass(frozen=True)
class _Artifact:
    raw_sha256: str
    format: str
    campaign_format: str
    population_identity_sha256: str
    stage_ids: tuple[str, ...]
    questions: tuple[_Question, ...]

    @property
    def question_count(self) -> int:
        return len(self.questions)


def _artifact() -> _Artifact:
    memberships = (
        ("e-alpha",),
        ("e-alpha", "e-beta"),
        ("e-alpha", "e-beta", "e-gamma"),
        ("e-alpha", "e-beta", "e-gamma"),
    )
    stages = tuple(
        _Stage(stage_id, _digest(f"source-prompt-{index}"), memberships[index])
        for index, stage_id in enumerate(STAGE_IDS)
    )
    return _Artifact(
        raw_sha256=_digest("retrieval"),
        format=RETRIEVAL_FORMAT,
        campaign_format="fixture-campaign-v1",
        population_identity_sha256=_digest("population"),
        stage_ids=STAGE_IDS,
        questions=(
            _Question(
                ordinal=0,
                question_id="question-0",
                question_sha256=_quote("What was selected?"),
                question="What was selected?",
                stages=stages,
            ),
        ),
    )


@pytest.fixture
def sealed_retrieval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    artifact = _artifact()

    def load(path, *, expected_sha256=None, verify_sidecar=True):
        assert Path(path) == tmp_path / "retrieval.json"
        assert expected_sha256 is None
        assert verify_sidecar is True
        return artifact

    monkeypatch.setattr(ledger_module, "load_fast_retrieval_artifact", load)
    return artifact, ArtifactSpec("retrieval", tmp_path / "retrieval.json")


def _cav_payload(artifact: _Artifact) -> dict:
    stage_id = STAGE_IDS[1]
    predictions = {
        "original": "Alpha",
        "base": "Alpha",
        "treatment": "Alpha and Beta",
    }
    answers = []
    logical_prompts = []
    for logical_ordinal, arm in enumerate(("original", "base", "treatment")):
        prediction = predictions[arm]
        binding = {
            "logical_ordinal": logical_ordinal,
            "question_ordinal": 0,
            "question_id": "question-0",
            "stage_id": stage_id,
            "arm_id": arm,
            "messages_sha256": _digest(f"cav-{arm}-messages"),
        }
        answers.append(
            {
                **binding,
                "prediction": prediction,
                "prediction_sha256": _quote(prediction),
            }
        )
        evidence_ids = list(artifact.questions[0].stage(stage_id).evidence_ids)
        if arm == "treatment":
            evidence_ids.reverse()
        logical_prompts.append({**binding, "evidence_ids": evidence_ids})
    return {
        "format": CAV_ANSWER_FORMAT,
        "gold_fields_present": False,
        "question_count": 1,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "feature_manifest_sha256": _digest("cav-features"),
        "prompt_population": {
            "format": "fixture-cav-prompts-v1",
            "selected_stage_ids": [stage_id],
            "logical_prompts": logical_prompts,
        },
        "completion_batch": {"runtime_identity_sha256": _digest("cav-runtime")},
        "logical_answer_count": len(answers),
        "answers": answers,
    }


def _hebbian_payload(artifact: _Artifact, *, replacement: bool) -> dict:
    stage_id = STAGE_IDS[0]
    base_chunks = ["chunk-alpha", "chunk-tail"]
    h1_chunks = ["chunk-alpha", "chunk-linked"] if replacement else base_chunks
    base_messages = _digest("hebbian-base-messages")
    h1_messages = _digest("hebbian-h1-messages") if replacement else base_messages
    answers = []
    for logical_ordinal, (arm, chunks, messages) in enumerate(
        (
            ("base", base_chunks, base_messages),
            ("h1", h1_chunks, h1_messages),
        )
    ):
        prediction = "Alpha"
        answers.append(
            {
                "logical_ordinal": logical_ordinal,
                "question_ordinal": 0,
                "question_id": "question-0",
                "stage_id": stage_id,
                "arm_id": arm,
                "prediction": prediction,
                "prediction_sha256": _quote(prediction),
                "messages_sha256": messages,
                "chunk_ids": chunks,
            }
        )
    return {
        "format": HEBBIAN_ANSWER_FORMAT,
        "gold_fields_present": False,
        "question_count": 1,
        "experiment_binding": {
            "retrieval_sha256": artifact.raw_sha256,
            "population_identity_sha256": artifact.population_identity_sha256,
            "association_artifact_sha256": _digest("associations"),
        },
        "prompt_population": {
            "question_receipts": [{"catalog_format": "fixture-catalog-v1"}]
        },
        "completion_batch": {
            "runtime_identity_sha256": _digest("hebbian-runtime")
        },
        "logical_answer_count": len(answers),
        "answers": answers,
    }


def _synthesis_payload(
    artifact: _Artifact,
    *,
    policy: str,
    first_answer: str,
) -> dict:
    question = artifact.questions[0]
    answers = (first_answer, "Alpha and Beta", "Alpha and Beta")
    stages = []
    for index, (stage_id, answer) in enumerate(
        zip(STAGE_IDS[1:], answers, strict=True), start=1
    ):
        stages.append(
            {
                "stage_id": stage_id,
                "source_prompt_messages_sha256": question.stage(
                    stage_id
                ).prompt_messages_sha256,
                "prompt_messages_sha256": _digest(f"{policy}-prompt-{index}"),
                "reused_from_stage_id": None,
                "answer": {"text": answer},
            }
        )
    return {
        "format": SYNTHESIS_FORMAT,
        "gold_fields_present": False,
        "gold_blind": True,
        "question_count": 1,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "stage_ids": list(STAGE_IDS[1:]),
        "synthesis_prompt_policy_sha256": _digest(f"synthesis-{policy}"),
        "request_policy_sha256": _digest(f"request-{policy}"),
        "runtime_identity_sha256": _digest("synthesis-runtime"),
        "questions": [
            {"ordinal": 0, "question_id": "question-0", "stages": stages}
        ],
    }


def _synthesis_score(answer_sha256: str, exact: tuple[bool, bool, bool]) -> dict:
    return {
        "format": SYNTHESIS_SCORE_FORMAT,
        "synthesis_sha256": answer_sha256,
        "question_count": 1,
        "questions": [
            {
                "question_id": "question-0",
                "stages": [
                    {
                        "stage_id": stage_id,
                        "exact_match": is_exact,
                        "f1": 1.0 if is_exact else 0.0,
                    }
                    for stage_id, is_exact in zip(
                        STAGE_IDS[1:], exact, strict=True
                    )
                ],
            }
        ],
    }


def _by_relation(ledger: dict, relation: str) -> list[dict]:
    return [row for row in ledger["comparisons"] if row["relation"] == relation]


def test_gold_blind_ledger_emits_layer_progression_and_marks_proxy(
    sealed_retrieval, tmp_path: Path
) -> None:
    artifact, retrieval = sealed_retrieval
    cav_path, cav_sha = _publish(tmp_path, "cav.json", _cav_payload(artifact))
    alias_path, _ = _publish(
        tmp_path, "hebbian-alias.json", _hebbian_payload(artifact, replacement=False)
    )
    replacement_path, _ = _publish(
        tmp_path,
        "hebbian-replacement.json",
        _hebbian_payload(artifact, replacement=True),
    )

    ledger = build_gold_blind_case_ledger(
        retrieval,
        (
            ArtifactSpec("cav", cav_path),
            ArtifactSpec("hebb_alias", alias_path),
            ArtifactSpec("hebb_replace", replacement_path),
        ),
    )

    assert ledger["gold_fields_present"] is False
    assert ledger["gold_artifacts_read"] == 0
    assert ledger["question_count"] == 1
    progression = ledger["questions"][0]["layer_progression"]
    assert [row["stage"] for row in progression[:4]] == ["S0", "S1", "S2", "S3"]
    cav_run = next(run for run in ledger["runs"] if run["run_id"] == "cav")
    assert cav_run["artifact_sha256"] == cav_sha
    assert cav_run["method_role"] == "linking_technique_proxy_readout_ablation"
    assert cav_run["is_retrieval_layer"] is False
    assert cav_run["is_direct_activation_injection"] is False
    cav_comparison = _by_relation(ledger, "cav_linking_text_order_proxy")[0]
    assert cav_comparison["causal_comparable"] is True
    assert cav_comparison["scope"] == "proxy_readout_answer"
    assert cav_comparison["membership_relation"] == "same_set_reordered"

    alias_h1 = next(
        row
        for row in ledger["observations"]
        if row["run_id"] == "hebb_alias" and row["arm_id"] == "h1"
    )
    assert alias_h1["alias_of"] is not None
    replacement = next(
        row
        for row in _by_relation(ledger, "hebbian_tail_replacement")
        if ledger_module._observation_id(
            "hebb_replace", "question-0", "S0", "h1"
        )
        == row["right_observation_id"]
    )
    assert replacement["membership_relation"] == "one_for_one_replacement"


def test_posthoc_labels_causal_progression_but_not_cross_run_delta(
    sealed_retrieval, tmp_path: Path
) -> None:
    artifact, retrieval = sealed_retrieval
    first_path, first_sha = _publish(
        tmp_path,
        "synth-first.json",
        _synthesis_payload(artifact, policy="first", first_answer="Alpha"),
    )
    second_path, second_sha = _publish(
        tmp_path,
        "synth-second.json",
        _synthesis_payload(artifact, policy="second", first_answer="Alpha and Beta"),
    )
    ledger = build_gold_blind_case_ledger(
        retrieval,
        (
            ArtifactSpec("first", first_path),
            ArtifactSpec("second", second_path),
        ),
    )
    cross = _by_relation(ledger, "prompt_contract_mismatch")
    cross = next(row for row in cross if row["scope"] == "cross_run_answer")
    assert cross["causal_comparable"] is False
    assert cross["prompt_contract_match"] is False

    first_score, _ = _publish(
        tmp_path,
        "first-scores.json",
        _synthesis_score(first_sha, (False, True, True)),
    )
    second_score, _ = _publish(
        tmp_path,
        "second-scores.json",
        _synthesis_score(second_sha, (True, True, True)),
    )
    posthoc = augment_case_ledger_posthoc(
        ledger,
        (
            ArtifactSpec("first", first_score),
            ArtifactSpec("second", second_score),
        ),
    )

    causal = next(
        row
        for row in posthoc["comparisons"]
        if row["relation"] == "cumulative_synthesis_stage"
        and row["left_observation_id"]
        == ledger_module._observation_id("first", "question-0", "S1", "answer")
    )
    assert causal["posthoc_outcome"] == "fix"
    assert causal["posthoc_metric_outcomes"] == {
        "exact_match": "fix",
        "f1": "improve",
    }
    cross_posthoc = next(
        row
        for row in posthoc["comparisons"]
        if row["comparison_id"] == cross["comparison_id"]
    )
    assert cross_posthoc["posthoc_outcome"] == "not_comparable"
    assert cross_posthoc["descriptive_outcome"] == "fix"
    assert posthoc["gold_blind_ledger_sha256"] == ledger["ledger_sha256"]


def test_gold_boundary_rejects_scores_fields_and_tampering(
    sealed_retrieval, tmp_path: Path
) -> None:
    artifact, retrieval = sealed_retrieval
    synth_payload = _synthesis_payload(
        artifact, policy="guardrail", first_answer="Alpha"
    )
    synth_path, synth_sha = _publish(tmp_path, "synth.json", synth_payload)
    ledger = build_gold_blind_case_ledger(
        retrieval, (ArtifactSpec("synth", synth_path),)
    )

    score_path, _ = _publish(
        tmp_path,
        "scores.json",
        _synthesis_score(synth_sha, (False, True, True)),
    )
    with pytest.raises(CaseLedgerValidationError, match="score artifacts are forbidden"):
        build_gold_blind_case_ledger(
            retrieval, (ArtifactSpec("score", score_path),)
        )

    poisoned = dict(synth_payload)
    poisoned["gold_answer"] = "Alpha and Beta"
    poisoned_path, _ = _publish(tmp_path, "poisoned.json", poisoned)
    with pytest.raises(CaseLedgerValidationError, match="gold-bearing field"):
        build_gold_blind_case_ledger(
            retrieval, (ArtifactSpec("poisoned", poisoned_path),)
        )

    tampered = copy.deepcopy(ledger)
    tampered["questions"][0]["question"] = "Changed after sealing"
    with pytest.raises(CaseLedgerValidationError, match="does not verify"):
        augment_case_ledger_posthoc(
            tampered, (ArtifactSpec("synth", score_path),)
        )

    wrong_score = _synthesis_score(_digest("wrong-answer-artifact"), (True,) * 3)
    wrong_path, _ = _publish(tmp_path, "wrong-score.json", wrong_score)
    with pytest.raises(CaseLedgerValidationError, match="does not bind run"):
        augment_case_ledger_posthoc(
            ledger, (ArtifactSpec("synth", wrong_path),)
        )
