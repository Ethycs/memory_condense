from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.recall_guarded_cumulative_1m import (
    QUESTION_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    _canonical_json_bytes,
    population_identity_sha256,
)
from memory_condense.eval.recall_guarded_cumulative_synthesis import (
    SYNTHESIS_FORMAT,
    SYNTHESIS_SCORE_FORMAT,
    SYNTHESIS_STAGE_IDS,
    assemble_synthesis_artifact,
    build_synthesis_messages,
    cumulative_novel_evidence,
    normalize_fallback_abstentions,
    parse_model_synthesis,
    score_recall_guarded_synthesis,
    synthesize_question,
    validate_published_retrieval,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


_QUERY = "Which restaurant did I choose?"
_GOLD = "Miss Bee Providore"
_ROOT_TEXT = "The shortlist included Miss Bee Providore and two other restaurants."
_NOVEL_ONE = "I chose Miss Bee Providore for dinner."
_NOVEL_TWO = "The booking at Miss Bee Providore was confirmed."
_NOVEL_THREE = "Later I said the Miss Bee Providore meal was excellent."


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample",
        questions=[
            BenchmarkQuestion(
                question_id="question-1",
                question=_QUERY,
                answer=_GOLD,
                evidence_sources=["source-root", "source-choice"],
            )
        ],
    )


def _evidence(evidence_id: str, source_id: str, text: str) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "text": text,
    }


_ROOT = _evidence("root", "source-root", _ROOT_TEXT)
_N1 = _evidence("novel-1", "source-choice", _NOVEL_ONE)
_N2 = _evidence("novel-2", "source-booking", _NOVEL_TWO)
_N3 = _evidence("novel-3", "source-review", _NOVEL_THREE)


def _messages(stage_id: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer only from the evidence."},
        {
            "role": "user",
            "content": (
                f"Retrieved excerpts for {stage_id}.\n\n"
                f"Question: {_QUERY}\nShort answer:"
            ),
        },
    ]


def _stage(
    stage_id: str,
    evidence: Sequence[Mapping[str, str]],
    projection: str,
) -> dict[str, Any]:
    messages = _messages(stage_id)
    return {
        "stage_id": stage_id,
        "stage_receipt": {
            "evidence_projection_sha256": projection,
            "prompt_messages_sha256": identity_sha256(messages),
            "selected_evidence_ids": [row["evidence_id"] for row in evidence],
        },
        "provider_messages": messages,
        "evidence": [dict(row) for row in evidence],
    }


def _question() -> dict[str, Any]:
    # S3 is an exact no-op over S2, so its sealed projection is identical. That
    # is the real campaign case in which synthesis memoization must apply.
    stages = [
        _stage(STAGE_IDS[0], [_ROOT], "0" * 64),
        _stage(STAGE_IDS[1], [_ROOT, _N1, _N2], "1" * 64),
        _stage(STAGE_IDS[2], [_ROOT, _N1, _N2, _N3], "2" * 64),
        _stage(STAGE_IDS[3], [_ROOT, _N1, _N2, _N3], "2" * 64),
    ]
    return {
        "format": QUESTION_FORMAT,
        "ordinal": 0,
        "question_id": "question-1",
        "question_sha256": "a" * 64,
        "stage_ids": list(STAGE_IDS),
        "provider_calls": 0,
        "stages": stages,
    }


def _retrieval(question: Mapping[str, Any] | None = None) -> dict[str, Any]:
    questions = [dict(question or _question())]
    return {
        "format": RETRIEVAL_FORMAT,
        "gold_fields_present": False,
        "population_identity_sha256": population_identity_sha256(_sample()),
        "stage_ids": list(STAGE_IDS),
        "question_count": len(questions),
        "questions": questions,
    }


def _model_response(
    labels: Sequence[str],
    *,
    citations: Sequence[tuple[str, str]] = (("E001", "Miss Bee Providore"),),
    answer_claim_ids: Sequence[str] = ("C1",),
    claim_citations: bool = True,
    duplicate_claim: bool = False,
    extra: bool = False,
) -> str:
    claim = {
        "claim_id": "C1",
        "text": "The user chose Miss Bee Providore.",
        "citations": [
            {"evidence_alias": alias, "quote": quote}
            for alias, quote in citations
        ]
        if claim_citations
        else [],
    }
    value: dict[str, Any] = {
        "answer": {"text": _GOLD, "claim_ids": list(answer_claim_ids)},
        "claims": [claim, dict(claim)] if duplicate_claim else [claim],
        "evidence_labels": [
            {
                "evidence_alias": alias,
                "role": "decisive" if index == 0 else "supporting",
                "density": "critical" if index == 0 else "high",
                "supports_claim_ids": ["C1"],
            }
            for index, alias in enumerate(labels)
        ],
    }
    if extra:
        value["unexpected"] = True
    return json.dumps(value)


@dataclass(frozen=True)
class _Score:
    candidate_id: str
    inspected: bool = True
    answerability: float = 0.9
    value_evidence_logit: float = 2.0
    direct_log_likelihood: float = -0.1
    indirect_log_likelihood: float = -2.1


class _Runtime:
    def __init__(self, completions: Sequence[str]) -> None:
        self._completions = iter(completions)
        self.identity = {"runtime": "test", "checkpoint_sha256": "c" * 64}
        self.last_completion_report: dict[str, Any] = {}
        self.last_score_report: dict[str, Any] = {}
        self.completion_messages: list[list[dict[str, str]]] = []
        self.score_inputs: list[tuple[str, dict[str, str]]] = []
        self.completion_calls = 0
        self.score_calls = 0

    @property
    def usage(self) -> dict[str, int]:
        return {
            "completion_calls": self.completion_calls,
            "score_calls": self.score_calls,
        }

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        normalized = [dict(row) for row in messages]
        self.completion_messages.append(normalized)
        self.completion_calls += 1
        raw = next(self._completions)
        self.last_completion_report = {
            "call": self.completion_calls,
            "max_new_tokens": max_new_tokens,
            "completion_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        }
        return raw

    def score_candidates(
        self,
        query: str,
        candidates: Mapping[str, str],
    ) -> Mapping[str, _Score]:
        normalized = dict(candidates)
        self.score_inputs.append((query, normalized))
        self.score_calls += 1
        self.last_score_report = {
            "call": self.score_calls,
            "candidate_ids": list(normalized),
        }
        return {candidate_id: _Score(candidate_id) for candidate_id in normalized}


def _successful_part(*, runtime: _Runtime | None = None) -> tuple[dict[str, Any], _Runtime]:
    active = runtime or _Runtime(
        [
            _model_response(["E002", "E003"]),
            _model_response(["E002", "E003", "E004"]),
            _model_response(["E002", "E003", "E004"]),
        ]
    )
    return (
        synthesize_question(
            _question(),
            retrieval_sha256="d" * 64,
            runtime=active,
            max_new_tokens=512,
        ),
        active,
    )


def test_s1_s3_label_population_does_not_memoize_different_prompts() -> None:
    question = _question()
    novel = cumulative_novel_evidence(question)

    assert [row["evidence_id"] for row in novel[STAGE_IDS[1]]] == [
        "novel-1",
        "novel-2",
    ]
    assert [row["evidence_id"] for row in novel[STAGE_IDS[2]]] == [
        "novel-1",
        "novel-2",
        "novel-3",
    ]

    messages, aliases, required = build_synthesis_messages(
        question["stages"][2], root_evidence_ids={"root"}
    )
    assert required == ("E002", "E003", "E004")
    assert aliases["E001"]["evidence_id"] == "root"
    assert "E002,E003,E004" in messages[1]["content"]

    part, runtime = _successful_part()
    s1, s2, s3 = part["stages"]
    assert [row["evidence_id"] for row in s1["evidence_labels"]] == [
        "novel-1",
        "novel-2",
    ]
    assert [row["evidence_id"] for row in s2["evidence_labels"]] == [
        "novel-1",
        "novel-2",
        "novel-3",
    ]
    assert len({row["evidence_id"] for row in s2["evidence_labels"]}) == 3

    # Claims may cite protected S0 evidence, but the normalized citation still
    # carries the exact protected source coordinate and quote hash.
    assert s1["claims"][0]["citations"][0]["evidence_id"] == "root"
    assert s1["claims"][0]["citations"][0]["quote"] == "Miss Bee Providore"

    assert runtime.completion_calls == 3
    assert s2["synthesis_key_sha256"] != s3["synthesis_key_sha256"]
    assert s3["reused_from_stage_id"] is None
    assert s1["synthesis_key_sha256"] != s2["synthesis_key_sha256"]


def test_exact_prompt_and_projection_are_memoized() -> None:
    question = _question()
    question["stages"][3]["provider_messages"] = copy.deepcopy(
        question["stages"][2]["provider_messages"]
    )
    question["stages"][3]["stage_receipt"]["prompt_messages_sha256"] = (
        identity_sha256(question["stages"][3]["provider_messages"])
    )
    runtime = _Runtime(
        [
            _model_response(["E002", "E003"]),
            _model_response(["E002", "E003", "E004"]),
            _model_response(["E002", "E003", "E004"]),
        ]
    )

    part = synthesize_question(
        question,
        retrieval_sha256="d" * 64,
        runtime=runtime,
    )

    s2, s3 = part["stages"][1:]
    assert runtime.completion_calls == 2
    assert s2["synthesis_key_sha256"] == s3["synthesis_key_sha256"]
    assert s3["reused_from_stage_id"] == STAGE_IDS[2]


def test_declared_attribution_fallback_uses_only_sealed_answer_prompts() -> None:
    class _FallbackRuntime(_Runtime):
        def score_candidates(
            self,
            query: str,
            candidates: Mapping[str, str],
        ) -> Mapping[str, _Score]:
            rows = super().score_candidates(query, candidates)
            if self.score_calls != 1:
                return rows
            probabilities = {
                "novel-1": 0.85,
                "novel-2": 0.52,
                "novel-3": 0.20,
            }
            return {
                candidate_id: _Score(
                    candidate_id,
                    answerability=probabilities[candidate_id],
                )
                for candidate_id in candidates
            }

    question = _question()
    runtime = _FallbackRuntime([_GOLD, _GOLD, _GOLD])
    part = synthesize_question(
        question,
        retrieval_sha256="d" * 64,
        runtime=runtime,
        attempt_structured=False,
        allow_attribution_fallback=True,
    )

    # Only the original sealed responder messages reach generation. Despite
    # equal S2/S3 evidence projections, their distinct prompt bytes prevent
    # memoization.
    assert runtime.completion_messages == [
        question["stages"][1]["provider_messages"],
        question["stages"][2]["provider_messages"],
        question["stages"][3]["provider_messages"],
    ]
    assert runtime.completion_calls == 3

    for source_stage, synthesized in zip(
        question["stages"][1:], part["stages"], strict=True
    ):
        expected_evidence = {
            row["evidence_id"]: row["text"] for row in source_stage["evidence"]
        }
        # Attribution is forced-choice scored over the complete stage, including
        # protected S0 evidence, rather than only over episodic additions.
        assert synthesized["attribution_score_report"]["candidate_ids"] == list(
            expected_evidence
        )
        assert (_QUERY, expected_evidence) in runtime.score_inputs

        citations = synthesized["claims"][0]["citations"]
        assert citations
        for citation in citations:
            expected = expected_evidence[citation["evidence_id"]]
            assert citation["quote"].encode("utf-8") == expected.encode("utf-8")
            assert citation["quote_sha256"] == hashlib.sha256(
                expected.encode("utf-8")
            ).hexdigest()

    score_by_id = {
        row["evidence_id"]: row for row in part["episodic_evidence_scores"]
    }
    expected_bands = {
        "novel-1": ("critical", "decisive"),
        "novel-2": ("critical", "decisive"),
        "novel-3": ("high", "supporting"),
    }
    assert {
        evidence_id: (row["evidence_density_band"], row["calibrated"])
        for evidence_id, row in score_by_id.items()
    } == {
        evidence_id: (density, False)
        for evidence_id, (density, _role) in expected_bands.items()
    }
    assert score_by_id["novel-3"]["answerability_band"] == "none"
    assert score_by_id["novel-3"]["evidence_density_band"] == "high"
    assert "causal_density_band" not in score_by_id["novel-3"]
    for synthesized in part["stages"]:
        assert synthesized["synthesis_mode"] == (
            "short_answer_with_forced_choice_attribution"
        )
        for label in synthesized["evidence_labels"]:
            density, role = expected_bands[label["evidence_id"]]
            assert (label["density"], label["role"]) == (density, role)
            assert label["label_origin"] == (
                "uncalibrated_answerability_per_100_tokens_band_v1"
            )


def test_normalize_fallback_abstentions_is_auditable_gold_blind_and_scoreable() -> None:
    retrieval = _retrieval()
    part, _runtime = _successful_part()
    artifact = assemble_synthesis_artifact(
        retrieval,
        retrieval_sha256="d" * 64,
        question_parts=[part],
    )
    abstention = artifact["questions"][0]["stages"][0]
    abstention["synthesis_mode"] = (
        "short_answer_with_forced_choice_attribution"
    )
    abstention["prompt_messages_sha256"] = abstention[
        "source_prompt_messages_sha256"
    ]
    abstention["answer"]["text"] = "I don't know."
    abstention["raw_completion"] = "I don't know."
    abstention["raw_completion_sha256"] = hashlib.sha256(
        abstention["raw_completion"].encode("utf-8")
    ).hexdigest()
    abstention["completion_report"] = {"sealed_completion_report": True}
    abstention["attribution_score_report"] = {
        "sealed_attribution_report": True
    }

    # Exercise the negative branch under the same fallback mode: a real answer
    # must not be rewritten merely because it used forced-choice attribution.
    non_abstention = artifact["questions"][0]["stages"][1]
    non_abstention["synthesis_mode"] = (
        "short_answer_with_forced_choice_attribution"
    )
    non_abstention["prompt_messages_sha256"] = non_abstention[
        "source_prompt_messages_sha256"
    ]
    abstention_before = copy.deepcopy(abstention)
    non_abstention_before = copy.deepcopy(non_abstention)
    source_synthesis_sha256 = "f" * 64
    artifact["question_identity_sha256s"] = [
        identity_sha256(question) for question in artifact["questions"]
    ]

    normalized = normalize_fallback_abstentions(
        artifact,
        source_synthesis_sha256=source_synthesis_sha256,
    )
    rewritten = normalized["questions"][0]["stages"][0]
    untouched = normalized["questions"][0]["stages"][1]

    assert rewritten["answer"] == {"text": "I don't know.", "claim_ids": []}
    assert rewritten["claims"] == []
    assert rewritten["claim_scores"] == []
    assert rewritten["claim_score_reports"] == []
    assert all(
        label["supports_claim_ids"] == []
        for label in rewritten["evidence_labels"]
    )
    assert rewritten["abstention_normalized"] is True

    # Raw model material and runtime reports are immutable evidence. Generated
    # attribution is removed from the active row but retained, hash-bound, for
    # audit rather than silently discarded.
    for key in (
        "raw_completion",
        "raw_completion_sha256",
        "completion_report",
        "attribution_score_report",
    ):
        assert rewritten[key] == abstention_before[key]
    discarded = {
        "claims": abstention_before["claims"],
        "claim_scores": abstention_before["claim_scores"],
        "claim_score_reports": abstention_before["claim_score_reports"],
    }
    assert rewritten["pre_normalization_abstention_attribution"] == {
        **discarded,
        "identity_sha256": identity_sha256(discarded),
    }

    expected_untouched = copy.deepcopy(non_abstention_before)
    expected_untouched["abstention_normalized"] = False
    assert untouched == expected_untouched
    # Normalization is non-mutating as well as gold-free.
    assert artifact["questions"][0]["stages"][0] == abstention_before
    assert normalized["gold_fields_present"] is False
    assert normalized["normalization"] == {
        "kind": "squad_normalized_fallback_abstention_v1",
        "source_synthesis_sha256": source_synthesis_sha256,
        "raw_completions_changed": False,
        "gold_fields_read": False,
        "normalized_stage_rows": 1,
    }

    sample = _sample()
    normalized_sha256 = hashlib.sha256(
        _canonical_json_bytes(normalized)
    ).hexdigest()
    scores = score_recall_guarded_synthesis(
        normalized,
        sample=sample,
        synthesis_sha256=normalized_sha256,
    )
    assert scores["format"] == SYNTHESIS_SCORE_FORMAT
    assert len(scores["questions"][0]["stages"]) == len(SYNTHESIS_STAGE_IDS)
    assert scores["questions"][0]["stages"][0]["exact_match"] is False


@pytest.mark.parametrize(
    ("completion", "message"),
    [
        (
            _model_response(
                ["E002", "E003"],
                citations=(("E999", "Miss Bee Providore"),),
            ),
            "unknown evidence alias",
        ),
        (
            _model_response(
                ["E002", "E003"],
                citations=(
                    ("E001", "Miss Bee Providore"),
                    ("E001", "Miss Bee Providore"),
                ),
            ),
            "duplicate citation",
        ),
        (
            _model_response(["E002"], citations=(("E001", "Miss Bee Providore"),)),
            "label population changed",
        ),
        (
            _model_response(["E002", "E002", "E003"]),
            "evidence labels must be unique",
        ),
        (
            _model_response(["E002", "E003"], claim_citations=False),
            "at least 1 item",
        ),
        (
            _model_response(["E002", "E003"], answer_claim_ids=("C9",)),
            "unknown claim IDs",
        ),
        (
            _model_response(["E002", "E003"], duplicate_claim=True),
            "claim IDs must be unique",
        ),
        (
            _model_response(["E002", "E003"], extra=True),
            "Extra inputs are not permitted",
        ),
        ("not JSON", "contains no JSON object"),
    ],
)
def test_invalid_model_outputs_are_rejected(completion: str, message: str) -> None:
    runtime = _Runtime([completion])

    with pytest.raises(ValueError, match=message):
        synthesize_question(
            _question(),
            retrieval_sha256="d" * 64,
            runtime=runtime,
        )


def test_citation_quote_must_be_an_exact_substring() -> None:
    completion = _model_response(
        ["E002", "E003"],
        citations=(("E001", "miss bee providore"),),
    )

    with pytest.raises(ValueError, match="exact evidence substring"):
        synthesize_question(
            _question(),
            retrieval_sha256="d" * 64,
            runtime=_Runtime([completion]),
        )


def test_parser_rejects_trailing_second_object_instead_of_merging_it() -> None:
    valid = _model_response(["E002", "E003"])
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_model_synthesis(valid + '{"second":true}')


def test_synthesis_runtime_never_receives_gold() -> None:
    sentinel = "GOLD-MUST-NEVER-ENTER-SYNTHESIS"
    runtime = _Runtime(
        [
            _model_response(["E002", "E003"]),
            _model_response(["E002", "E003", "E004"]),
            _model_response(["E002", "E003", "E004"]),
        ]
    )
    synthesize_question(
        _question(),
        retrieval_sha256="d" * 64,
        runtime=runtime,
    )

    provider_visible = json.dumps(runtime.completion_messages, sort_keys=True)
    scorer_visible = json.dumps(runtime.score_inputs, sort_keys=True)
    assert sentinel not in provider_visible
    assert sentinel not in scorer_visible
    # The gold-bearing BenchmarkSample is accepted only by the separate
    # score_recall_guarded_synthesis entry point.
    assert "gold" not in synthesize_question.__annotations__


def test_posthoc_answer_metrics_and_artifact_bindings() -> None:
    retrieval = _retrieval()
    validate_published_retrieval(retrieval)
    part, _runtime = _successful_part()
    artifact = assemble_synthesis_artifact(
        retrieval,
        retrieval_sha256="d" * 64,
        question_parts=[part],
    )
    assert artifact["format"] == SYNTHESIS_FORMAT
    assert artifact["retrieval_sha256"] == "d" * 64
    assert artifact["population_identity_sha256"] == population_identity_sha256(
        _sample()
    )
    assert artifact["stage_ids"] == list(SYNTHESIS_STAGE_IDS)
    assert artifact["unique_synthesis_calls"] == 3
    assert artifact["authoritative_runtime_usage"]["completion_calls"] == 3
    assert len(artifact["synthesis_implementation_sha256"]) == 64
    assert len(artifact["synthesis_prompt_policy_sha256"]) == 64
    assert len(artifact["runtime_identity_sha256"]) == 64
    assert len(artifact["question_identity_sha256s"]) == 1
    assert artifact["questions"][0]["bound_stages"][0]["evidence"]

    sample = _sample()
    synthesis_sha256 = hashlib.sha256(
        _canonical_json_bytes(artifact)
    ).hexdigest()
    scores = score_recall_guarded_synthesis(
        artifact,
        sample=sample,
        synthesis_sha256=synthesis_sha256,
    )

    assert scores["format"] == SYNTHESIS_SCORE_FORMAT
    assert scores["synthesis_sha256"] == synthesis_sha256
    assert scores["retrieval_sha256"] == "d" * 64
    assert scores["population_identity_sha256"] == population_identity_sha256(
        sample
    )
    assert scores["gold_loaded_posthoc"] is True
    assert scores["independent_llm_judge_calls"] == 0
    assert [row["stage_id"] for row in scores["stage_aggregates"]] == list(
        SYNTHESIS_STAGE_IDS
    )
    assert all(row["mean_f1"] == 1.0 for row in scores["stage_aggregates"])
    assert all(
        stage["exact_match"] is True
        for stage in scores["questions"][0]["stages"]
    )
    assert all(
        row["mean_answer_value_component_recall"] is None
        for row in scores["stage_aggregates"]
    )

    wrong_order = copy.deepcopy(artifact)
    wrong_order["questions"][0]["question_id"] = "another-question"
    wrong_order["question_identity_sha256s"] = [
        identity_sha256(question) for question in wrong_order["questions"]
    ]
    with pytest.raises(ValueError, match="order differs"):
        score_recall_guarded_synthesis(
            wrong_order,
            sample=sample,
            synthesis_sha256=hashlib.sha256(
                _canonical_json_bytes(wrong_order)
            ).hexdigest(),
        )


def test_posthoc_rejects_population_and_citation_tampering() -> None:
    retrieval = _retrieval()
    part, _runtime = _successful_part()
    artifact = assemble_synthesis_artifact(
        retrieval,
        retrieval_sha256="d" * 64,
        question_parts=[part],
    )
    artifact_sha = hashlib.sha256(_canonical_json_bytes(artifact)).hexdigest()
    changed_sample = _sample().model_copy(
        update={
            "questions": [
                _sample().questions[0].model_copy(
                    update={"question": "Which venue did I choose?"}
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="population identities differ"):
        score_recall_guarded_synthesis(
            artifact,
            sample=changed_sample,
            synthesis_sha256=artifact_sha,
        )

    tampered = copy.deepcopy(artifact)
    citation = tampered["questions"][0]["stages"][0]["claims"][0][
        "citations"
    ][0]
    citation["quote"] = "miss bee providore"
    citation["quote_sha256"] = hashlib.sha256(
        citation["quote"].encode("utf-8")
    ).hexdigest()
    tampered["question_identity_sha256s"] = [
        identity_sha256(question) for question in tampered["questions"]
    ]
    with pytest.raises(ValueError, match="exact evidence"):
        score_recall_guarded_synthesis(
            tampered,
            sample=_sample(),
            synthesis_sha256=hashlib.sha256(
                _canonical_json_bytes(tampered)
            ).hexdigest(),
        )


def test_posthoc_reports_multi_value_answer_and_claim_components() -> None:
    retrieval = _retrieval()
    part, _runtime = _successful_part()
    artifact = assemble_synthesis_artifact(
        retrieval,
        retrieval_sha256="d" * 64,
        question_parts=[part],
    )
    sample = _sample().model_copy(
        update={
            "questions": [
                _sample().questions[0].model_copy(
                    update={"answer": "Miss Bee Providore, dinner"}
                )
            ]
        }
    )
    scores = score_recall_guarded_synthesis(
        artifact,
        sample=sample,
        synthesis_sha256=hashlib.sha256(
            _canonical_json_bytes(artifact)
        ).hexdigest(),
    )

    for stage in scores["questions"][0]["stages"]:
        assert stage["answer_value_components_expected"] == 2
        assert stage["answer_value_components_found"] == 1
        assert stage["answer_value_component_recall"] == 0.5
        assert stage["claim_value_component_recall"] == 0.5
    for aggregate in scores["stage_aggregates"]:
        assert aggregate["answer_value_component_questions"] == 1
        assert aggregate["mean_answer_value_component_recall"] == 0.5
