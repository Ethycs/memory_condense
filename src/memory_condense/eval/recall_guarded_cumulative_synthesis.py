"""Gold-blind episodic evidence scoring and cited S1--S3 synthesis.

This public facade orchestrates generation and post-hoc scoring.  Strict
schemas/input transforms live in the adjacent contracts module, while durable
artifact validation/publication lives in the artifacts module.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._recall_guarded_cumulative_synthesis_artifacts import (
    _validate_assembled_synthesis,
    assemble_synthesis_artifact,
    normalize_fallback_abstentions,
)
from memory_condense.eval._recall_guarded_cumulative_synthesis_contracts import (
    ANSWER_REUSE_FORMAT,
    ANSWER_REUSE_RULE,
    ANSWERABILITY_BAND_THRESHOLDS,
    EVIDENCE_DENSITY_PER_100_TOKEN_THRESHOLDS,
    EvidenceDensity,
    EvidenceRole,
    ModelSynthesis,
    SYNTHESIS_FORMAT,
    SYNTHESIS_PROMPT_POLICY,
    SYNTHESIS_PROMPT_POLICY_SHA256,
    SYNTHESIS_QUESTION_FORMAT,
    SYNTHESIS_SCORE_FORMAT,
    SYNTHESIS_STAGE_IDS,
    SynthesisRuntime,
    _dump,
    _evidence_rows,
    _projection_sha,
    _runtime_identity,
    _score_row,
    _sha256_text,
    _stage_receipt_sha256,
    _usage_delta,
    _validated_synthesis,
    build_synthesis_messages,
    cumulative_novel_evidence,
    extract_stage_question,
    parse_model_synthesis,
    validate_published_retrieval,
)
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
)
from memory_condense.eval.benchmark import exact_match, f1_score
from memory_condense.eval.recall_guarded_cumulative_1m import (
    _canonical_json_bytes,
    population_identity_sha256,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.ingest.loader import BenchmarkSample


def _rescore_claims(
    *,
    runtime: SynthesisRuntime,
    question: str,
    synthesis: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = synthesis["claims"]
    if not claims:
        return [], []
    claim_texts = {str(row["claim_id"]): str(row["text"]) for row in claims}
    answer_rows = runtime.score_candidates(question, claim_texts)
    answer_report = _dump(runtime.last_score_report)
    normalized: list[dict[str, Any]] = []
    support_reports: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim["claim_id"])
        if claim_id not in answer_rows:
            raise ValueError("claim answer-value scorer omitted a claim")
        quotes = "\n".join(str(row["quote"]) for row in claim["citations"])
        support = runtime.score_candidates(
            f"Claim to verify:\n{claim['text']}", {claim_id: quotes}
        )
        support_report = _dump(runtime.last_score_report)
        if claim_id not in support:
            raise ValueError("citation-support scorer omitted a claim")
        normalized.append(
            {
                "claim_id": claim_id,
                "answer_value": _score_row(
                    answer_rows[claim_id], text=claim["text"]
                ),
                "citation_support_proxy": _score_row(
                    support[claim_id], text=quotes
                ),
            }
        )
        support_reports.append(
            {"claim_id": claim_id, "score_report": support_report}
        )
    return normalized, [
        {"kind": "answer_value", "score_report": answer_report},
        *support_reports,
    ]


def _forced_choice_attribution_fallback(
    *,
    runtime: SynthesisRuntime,
    stage: Mapping[str, Any],
    question: str,
    root_evidence_ids: set[str],
    episodic_scores: Mapping[str, Mapping[str, Any]],
    max_new_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Build a mechanically grounded card when structured generation fails."""

    provider_messages = stage.get("provider_messages")
    if not isinstance(provider_messages, list):
        raise ValueError("fallback stage has no sealed provider messages")
    answer = runtime.complete(
        provider_messages,
        max_new_tokens=min(256, max_new_tokens),
    ).strip()
    completion_report = _dump(runtime.last_completion_report)
    if not answer:
        raise ValueError("fallback responder produced an empty answer")

    rows = _evidence_rows(stage)
    evidence_text = {row["evidence_id"]: row["text"] for row in rows}
    attribution = runtime.score_candidates(question, evidence_text)
    attribution_report = _dump(runtime.last_score_report)
    if set(attribution) != set(evidence_text):
        raise ValueError("fallback attribution changed the evidence population")
    ranked = sorted(
        rows,
        key=lambda row: (
            int(bool(attribution[row["evidence_id"]].inspected)),
            float(attribution[row["evidence_id"]].answerability),
            -rows.index(row),
        ),
        reverse=True,
    )
    is_abstention = exact_match(answer, "I don't know")
    if is_abstention:
        selected: list[dict[str, str]] = []
    else:
        selected = [
            row
            for row in ranked
            if attribution[row["evidence_id"]].inspected
            and float(attribution[row["evidence_id"]].answerability) >= 0.50
        ][:3]
        if not selected and ranked:
            selected = ranked[:1]

    claim_ids = [] if is_abstention else ["C1"]
    claims = []
    if claim_ids:
        claims = [
            {
                "claim_id": "C1",
                "text": answer,
                "citations": [
                    {
                        "evidence_id": row["evidence_id"],
                        "source_id": row["source_id"],
                        "evidence_text_sha256": _sha256_text(row["text"]),
                        "quote": row["text"],
                        "quote_sha256": quote_sha256(row["text"]),
                    }
                    for row in selected
                ],
            }
        ]
    role_by_band: dict[str, EvidenceRole] = {
        "critical": "decisive",
        "high": "supporting",
        "medium": "context",
        "low": "redundant",
        "none": "irrelevant",
    }
    selected_ids = {row["evidence_id"] for row in selected}
    labels: list[dict[str, Any]] = []
    for row in rows:
        evidence_id = row["evidence_id"]
        if evidence_id in root_evidence_ids:
            continue
        score = episodic_scores.get(evidence_id)
        if not isinstance(score, Mapping):
            raise ValueError("fallback has no causal score for episodic evidence")
        band = str(score["evidence_density_band"])
        labels.append(
            {
                "evidence_id": evidence_id,
                "source_id": row["source_id"],
                "evidence_text_sha256": _sha256_text(row["text"]),
                "role": role_by_band[band],
                "density": band,
                "supports_claim_ids": (
                    ["C1"] if evidence_id in selected_ids and claim_ids else []
                ),
                "label_origin": (
                    "uncalibrated_answerability_per_100_tokens_band_v1"
                ),
            }
        )
    return (
        {
            "answer": {"text": answer, "claim_ids": claim_ids},
            "claims": claims,
            "evidence_labels": labels,
        },
        answer,
        completion_report,
        attribution_report,
    )


def synthesize_question(
    question: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    runtime: SynthesisRuntime,
    max_new_tokens: int = 2048,
    allow_attribution_fallback: bool = False,
    attempt_structured: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score and synthesize one sealed retrieval question without gold."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    synthesis_implementation = implementation_sha256()
    runtime_identity, runtime_identity_sha = _runtime_identity(runtime)
    usage_before = _dump(runtime.usage)
    request_policy = {
        "attempt_structured": bool(attempt_structured),
        "allow_attribution_fallback": bool(allow_attribution_fallback),
        "max_new_tokens": int(max_new_tokens),
    }
    request_policy_sha = identity_sha256(request_policy)
    stages = question.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError("question must contain S0 through S3")
    stage_maps = [stage for stage in stages if isinstance(stage, Mapping)]
    if len(stage_maps) != 4:
        raise ValueError("question stage must be an object")
    root_rows = _evidence_rows(stage_maps[0])
    root_ids = {row["evidence_id"] for row in root_rows}
    final_rows = _evidence_rows(stage_maps[-1])
    novel_rows = [row for row in final_rows if row["evidence_id"] not in root_ids]
    query = extract_stage_question(stage_maps[0])
    candidates = {row["evidence_id"]: row["text"] for row in novel_rows}
    if progress:
        progress(f"scoring {len(candidates)} cumulative episodic evidence items")
    scored = runtime.score_candidates(query, candidates)
    score_report = _dump(runtime.last_score_report)
    if set(scored) != set(candidates):
        raise ValueError("causal scorer changed the episodic evidence population")
    evidence_scores = []
    for row in novel_rows:
        score = _score_row(scored[row["evidence_id"]], text=row["text"])
        if not score["inspected"]:
            raise ValueError("causal scorer left episodic evidence uninspected")
        evidence_scores.append(
            {
                "evidence_id": row["evidence_id"],
                "source_id": row["source_id"],
                "evidence_text_sha256": _sha256_text(row["text"]),
                **score,
            }
        )
    episodic_scores = {
        str(row["evidence_id"]): row for row in evidence_scores
    }

    cached: dict[str, dict[str, Any]] = {}
    stage_outputs: list[dict[str, Any]] = []
    previous_evidence_ids = {row["evidence_id"] for row in root_rows}
    for stage in stage_maps[1:]:
        stage_id = str(stage["stage_id"])
        projection = _projection_sha(stage)
        messages, by_alias, novel_aliases = build_synthesis_messages(
            stage, root_evidence_ids=root_ids
        )
        source_prompt_sha = identity_sha256(stage["provider_messages"])
        structured_prompt_sha = identity_sha256(messages)
        key = identity_sha256(
            {
                "question_sha256": question.get("question_sha256"),
                "evidence_projection_sha256": projection,
                "source_prompt_messages_sha256": source_prompt_sha,
                "structured_prompt_messages_sha256": structured_prompt_sha,
                "runtime_identity_sha256": runtime_identity_sha,
                "synthesis_prompt_policy_sha256": (
                    SYNTHESIS_PROMPT_POLICY_SHA256
                ),
                "request_policy_sha256": request_policy_sha,
            }
        )
        if key in cached:
            reused = dict(cached[key])
            reused["stage_id"] = stage_id
            reused["reused_from_stage_id"] = cached[key]["stage_id"]
            reused["source_stage_receipt_sha256"] = _stage_receipt_sha256(stage)
            stage_outputs.append(reused)
            previous_evidence_ids = {
                row["evidence_id"] for row in _evidence_rows(stage)
            }
            continue
        if progress:
            progress(
                f"synthesizing {stage_id} with {len(novel_aliases)} episodic labels"
            )
        structured_attempt: dict[str, Any] | None = None
        attribution_report: dict[str, Any] | None = None
        if not attempt_structured:
            if not allow_attribution_fallback:
                raise ValueError(
                    "skipping structured generation requires the declared fallback"
                )
            if progress:
                progress(
                    f"using short-answer/forced-choice attribution for {stage_id}"
                )
            (
                normalized,
                raw,
                completion_report,
                attribution_report,
            ) = _forced_choice_attribution_fallback(
                runtime=runtime,
                stage=stage,
                question=query,
                root_evidence_ids=root_ids,
                episodic_scores=episodic_scores,
                max_new_tokens=max_new_tokens,
            )
            synthesis_mode = "short_answer_with_forced_choice_attribution"
            effective_prompt_sha256 = source_prompt_sha
        else:
            raw = runtime.complete(messages, max_new_tokens=max_new_tokens)
            completion_report = _dump(runtime.last_completion_report)
            effective_prompt_sha256 = structured_prompt_sha
            synthesis_mode = "structured_generation"
            try:
                parsed = parse_model_synthesis(raw)
                normalized = _validated_synthesis(
                    parsed,
                    by_alias=by_alias,
                    novel_aliases=novel_aliases,
                )
            except ValueError as exc:
                if not allow_attribution_fallback:
                    raise
                structured_attempt = {
                    "valid": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_completion": raw,
                    "raw_completion_sha256": _sha256_text(raw),
                    "completion_report": completion_report,
                    "prompt_messages_sha256": structured_prompt_sha,
                }
                if progress:
                    progress(
                        f"structured parse failed for {stage_id}; using declared "
                        "short-answer/forced-choice attribution fallback"
                    )
                (
                    normalized,
                    raw,
                    completion_report,
                    attribution_report,
                ) = _forced_choice_attribution_fallback(
                    runtime=runtime,
                    stage=stage,
                    question=query,
                    root_evidence_ids=root_ids,
                    episodic_scores=episodic_scores,
                    max_new_tokens=max_new_tokens,
                )
                synthesis_mode = (
                    "short_answer_with_forced_choice_attribution"
                )
                effective_prompt_sha256 = source_prompt_sha
        current_evidence_ids = [
            row["evidence_id"] for row in _evidence_rows(stage)
        ]
        newly_added_ids = [
            evidence_id
            for evidence_id in current_evidence_ids
            if evidence_id not in root_ids
            and evidence_id not in previous_evidence_ids
        ]
        current_labels = list(normalized["evidence_labels"])
        labels_by_id = {
            str(label["evidence_id"]): label for label in current_labels
        }
        answer_reuse: dict[str, Any] | None = None
        if (
            stage_outputs
            and all(
                labels_by_id[evidence_id]["density"] == "none"
                and labels_by_id[evidence_id]["role"] == "irrelevant"
                and labels_by_id[evidence_id]["supports_claim_ids"] == []
                for evidence_id in newly_added_ids
            )
        ):
            previous = stage_outputs[-1]
            generated_answer = copy.deepcopy(normalized["answer"])
            generated_claims = copy.deepcopy(normalized["claims"])
            generated_labels = copy.deepcopy(current_labels)
            previous_labels_by_id = {
                str(label["evidence_id"]): label
                for label in previous["evidence_labels"]
            }
            effective_labels = [
                copy.deepcopy(
                    previous_labels_by_id.get(
                        str(label["evidence_id"]),
                        label,
                    )
                )
                for label in generated_labels
            ]
            new_labels = [
                labels_by_id[evidence_id] for evidence_id in newly_added_ids
            ]
            reuse_body = {
                "format": ANSWER_REUSE_FORMAT,
                "rule": ANSWER_REUSE_RULE,
                "from_stage_id": previous["stage_id"],
                "new_evidence_ids": newly_added_ids,
                "new_evidence_labels_sha256": identity_sha256(new_labels),
                "generated_answer": generated_answer,
                "generated_claims": generated_claims,
                "generated_evidence_labels": generated_labels,
                "generated_answer_sha256": identity_sha256(generated_answer),
                "generated_claims_sha256": identity_sha256(generated_claims),
                "generated_evidence_labels_sha256": identity_sha256(
                    generated_labels
                ),
                "source_answer_sha256": identity_sha256(previous["answer"]),
                "source_claims_sha256": identity_sha256(previous["claims"]),
                "source_evidence_labels_sha256": identity_sha256(
                    previous["evidence_labels"]
                ),
                "effective_evidence_labels_sha256": identity_sha256(
                    effective_labels
                ),
            }
            answer_reuse = {
                **reuse_body,
                "receipt_sha256": identity_sha256(reuse_body),
            }
            normalized["answer"] = copy.deepcopy(previous["answer"])
            normalized["claims"] = copy.deepcopy(previous["claims"])
            normalized["evidence_labels"] = effective_labels
        claim_scores, claim_reports = _rescore_claims(
            runtime=runtime,
            question=query,
            synthesis=normalized,
        )
        row = {
            "stage_id": stage_id,
            "reused_from_stage_id": None,
            "synthesis_key_sha256": key,
            "evidence_projection_sha256": projection,
            "source_stage_receipt_sha256": _stage_receipt_sha256(stage),
            "source_prompt_messages_sha256": source_prompt_sha,
            "prompt_messages_sha256": effective_prompt_sha256,
            "structured_prompt_messages_sha256": structured_prompt_sha,
            "runtime_identity_sha256": runtime_identity_sha,
            "synthesis_implementation_sha256": synthesis_implementation,
            "synthesis_prompt_policy_sha256": SYNTHESIS_PROMPT_POLICY_SHA256,
            "request_policy": dict(request_policy),
            "request_policy_sha256": request_policy_sha,
            "raw_completion": raw,
            "raw_completion_sha256": _sha256_text(raw),
            "completion_report": completion_report,
            "synthesis_mode": synthesis_mode,
            "structured_attempt": structured_attempt,
            "attribution_score_report": attribution_report,
            "monotonic_answer_reuse": answer_reuse,
            "episodic_evidence_count": len(novel_aliases),
            **normalized,
            "claim_scores": claim_scores,
            "claim_score_reports": claim_reports,
        }
        cached[key] = row
        stage_outputs.append(dict(row))
        previous_evidence_ids = set(current_evidence_ids)
    if implementation_sha256() != synthesis_implementation:
        raise RuntimeError("synthesis implementation changed during the question")
    usage_after = _dump(runtime.usage)
    return {
        "format": SYNTHESIS_QUESTION_FORMAT,
        "retrieval_sha256": retrieval_sha256,
        "ordinal": int(question["ordinal"]),
        "question_id": str(question["question_id"]),
        "question_sha256": str(question["question_sha256"]),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha,
        "synthesis_implementation_sha256": synthesis_implementation,
        "synthesis_prompt_policy": dict(SYNTHESIS_PROMPT_POLICY),
        "synthesis_prompt_policy_sha256": SYNTHESIS_PROMPT_POLICY_SHA256,
        "request_policy": request_policy,
        "request_policy_sha256": request_policy_sha,
        "gold_fields_present": False,
        "episodic_scoring_population": "unique evidence in S3 minus S0",
        "episodic_evidence_count": len(novel_rows),
        "episodic_evidence_scores": evidence_scores,
        "episodic_score_report": score_report,
        "stages": stage_outputs,
        "runtime_usage_before": usage_before,
        "runtime_usage_delta": _usage_delta(usage_before, usage_after),
        "runtime_usage_cumulative": usage_after,
    }


def _source_metrics(selected: set[str], expected: set[str]) -> dict[str, Any]:
    overlap = len(selected & expected)
    return {
        "selected_count": len(selected),
        "expected_count": len(expected),
        "overlap_count": overlap,
        "precision": (
            overlap / len(selected)
            if selected
            else (1.0 if not expected else 0.0)
        ),
        "recall": overlap / len(expected) if expected else 1.0,
    }


def score_recall_guarded_synthesis(
    synthesis: Mapping[str, Any],
    *,
    sample: BenchmarkSample,
    synthesis_sha256: str,
) -> dict[str, Any]:
    """Read gold only after synthesis and score answers/citations by stage."""

    _validate_assembled_synthesis(synthesis)
    observed_synthesis_sha = hashlib.sha256(
        _canonical_json_bytes(synthesis)
    ).hexdigest()
    if synthesis_sha256 != observed_synthesis_sha:
        raise ValueError("synthesis SHA-256 does not match its canonical bytes")
    observed_population_sha = population_identity_sha256(sample)
    if synthesis.get("population_identity_sha256") != observed_population_sha:
        raise ValueError("synthesis and benchmark population identities differ")
    questions = synthesis.get("questions")
    if not isinstance(questions, list) or len(questions) != len(sample.questions):
        raise ValueError("synthesis and benchmark question populations differ")
    by_stage: dict[str, list[dict[str, Any]]] = {
        stage_id: [] for stage_id in SYNTHESIS_STAGE_IDS
    }
    scored_questions: list[dict[str, Any]] = []
    for source, gold in zip(questions, sample.questions, strict=True):
        if source.get("question_id") != gold.question_id:
            raise ValueError("synthesis question order differs from benchmark")
        evidence_scores = {
            row["evidence_id"]: row
            for row in source["episodic_evidence_scores"]
        }
        stage_rows: list[dict[str, Any]] = []
        for stage in source["stages"]:
            answer = str(stage["answer"]["text"])
            citations = [
                citation
                for claim in stage["claims"]
                for citation in claim["citations"]
            ]
            cited_sources = {str(row["source_id"]) for row in citations}
            expected_sources = set(gold.evidence_sources)
            dense_ids = {
                str(row["evidence_id"])
                for row in stage["evidence_labels"]
                if row["density"] in {"critical", "high"}
            }
            dense_sources = {
                str(row["source_id"])
                for row in stage["evidence_labels"]
                if row["evidence_id"] in dense_ids
            }
            label_counts = Counter(
                str(row["density"]) for row in stage["evidence_labels"]
            )
            role_counts = Counter(
                str(row["role"]) for row in stage["evidence_labels"]
            )
            causal = [
                evidence_scores[row["evidence_id"]]
                for row in stage["evidence_labels"]
            ]
            claim_scores = stage.get("claim_scores", [])
            answer_components = answer_value_component_coverage(
                gold.answer,
                len(expected_sources),
                [answer],
            )
            claim_texts = [str(claim["text"]) for claim in stage["claims"]]
            claim_components = answer_value_component_coverage(
                gold.answer,
                len(expected_sources),
                claim_texts,
            )
            row = {
                "stage_id": stage["stage_id"],
                "answer_sha256": quote_sha256(answer),
                "f1": f1_score(answer, gold.answer),
                "exact_match": exact_match(answer, gold.answer),
                "answer_value_components_expected": (
                    None
                    if answer_components is None
                    else answer_components.expected
                ),
                "answer_value_components_found": (
                    None
                    if answer_components is None
                    else answer_components.found
                ),
                "answer_value_component_recall": (
                    None
                    if answer_components is None
                    else answer_components.recall
                ),
                "all_answer_value_components": (
                    None
                    if answer_components is None
                    else answer_components.all_components
                ),
                "answer_value_component_hit_mask": (
                    []
                    if answer_components is None
                    else list(answer_components.hit_mask)
                ),
                "answer_value_metric_kind": (
                    ""
                    if answer_components is None
                    else answer_components.metric_kind
                ),
                "claim_value_component_recall": (
                    None
                    if claim_components is None
                    else claim_components.recall
                ),
                "all_claim_value_components": (
                    None
                    if claim_components is None
                    else claim_components.all_components
                ),
                "claim_count": len(stage["claims"]),
                "citation_count": len(citations),
                "citation_quote_grounding_rate": 1.0,
                "cited_expected_source": _source_metrics(
                    cited_sources, expected_sources
                ),
                "critical_high_expected_source": _source_metrics(
                    dense_sources, expected_sources
                ),
                "density_counts": dict(sorted(label_counts.items())),
                "role_counts": dict(sorted(role_counts.items())),
                "mean_causal_answerability": (
                    sum(float(item["answerability"]) for item in causal)
                    / len(causal)
                    if causal
                    else 0.0
                ),
                "mean_causal_answerability_per_100_tokens": (
                    sum(
                        float(item["answerability_per_100_tokens"])
                        for item in causal
                    )
                    / len(causal)
                    if causal
                    else 0.0
                ),
                "mean_claim_answerability": (
                    sum(
                        float(item["answer_value"]["answerability"])
                        for item in claim_scores
                    )
                    / len(claim_scores)
                    if claim_scores
                    else 0.0
                ),
                "mean_citation_support_proxy": (
                    sum(
                        float(
                            item["citation_support_proxy"]["answerability"]
                        )
                        for item in claim_scores
                    )
                    / len(claim_scores)
                    if claim_scores
                    else 0.0
                ),
            }
            by_stage[str(stage["stage_id"])].append(row)
            stage_rows.append(row)
        scored_questions.append(
            {"question_id": gold.question_id, "stages": stage_rows}
        )
    aggregates: list[dict[str, Any]] = []
    for stage_id, rows in by_stage.items():
        component_rows = [
            row
            for row in rows
            if row["answer_value_component_recall"] is not None
        ]
        claim_component_rows = [
            row
            for row in rows
            if row["claim_value_component_recall"] is not None
        ]
        aggregates.append(
            {
                "stage_id": stage_id,
                "questions": len(rows),
                "exact_matches": sum(
                    bool(row["exact_match"]) for row in rows
                ),
                "mean_f1": (
                    sum(float(row["f1"]) for row in rows) / len(rows)
                ),
                "answer_value_component_questions": len(component_rows),
                "mean_answer_value_component_recall": (
                    None
                    if not component_rows
                    else sum(
                        float(row["answer_value_component_recall"])
                        for row in component_rows
                    )
                    / len(component_rows)
                ),
                "all_answer_value_component_hits": sum(
                    row["all_answer_value_components"] is True
                    for row in component_rows
                ),
                "mean_claim_value_component_recall": (
                    None
                    if not claim_component_rows
                    else sum(
                        float(row["claim_value_component_recall"])
                        for row in claim_component_rows
                    )
                    / len(claim_component_rows)
                ),
                "mean_causal_answerability": (
                    sum(
                        float(row["mean_causal_answerability"])
                        for row in rows
                    )
                    / len(rows)
                ),
                "mean_claim_answerability": (
                    sum(
                        float(row["mean_claim_answerability"])
                        for row in rows
                    )
                    / len(rows)
                ),
                "mean_citation_support_proxy": (
                    sum(
                        float(row["mean_citation_support_proxy"])
                        for row in rows
                    )
                    / len(rows)
                ),
                "mean_cited_expected_source_recall": (
                    sum(
                        float(row["cited_expected_source"]["recall"])
                        for row in rows
                    )
                    / len(rows)
                ),
            }
        )
    return {
        "format": SYNTHESIS_SCORE_FORMAT,
        "synthesis_sha256": synthesis_sha256,
        "retrieval_sha256": synthesis.get("retrieval_sha256"),
        "population_identity_sha256": synthesis.get(
            "population_identity_sha256"
        ),
        "gold_scoring_population_sha256": identity_sha256(
            [
                {
                    "question_id": question.question_id,
                    "answer_sha256": quote_sha256(question.answer),
                    "evidence_source_ids": list(question.evidence_sources),
                }
                for question in sample.questions
            ]
        ),
        "question_count": len(scored_questions),
        "gold_loaded_posthoc": True,
        "independent_llm_judge_calls": 0,
        "same_model_scores_calibrated": False,
        "questions": scored_questions,
        "stage_aggregates": aggregates,
    }


__all__ = [
    "ANSWERABILITY_BAND_THRESHOLDS",
    "EVIDENCE_DENSITY_PER_100_TOKEN_THRESHOLDS",
    "EvidenceDensity",
    "EvidenceRole",
    "ModelSynthesis",
    "SYNTHESIS_FORMAT",
    "SYNTHESIS_PROMPT_POLICY",
    "SYNTHESIS_PROMPT_POLICY_SHA256",
    "SYNTHESIS_QUESTION_FORMAT",
    "SYNTHESIS_SCORE_FORMAT",
    "SYNTHESIS_STAGE_IDS",
    "assemble_synthesis_artifact",
    "build_synthesis_messages",
    "cumulative_novel_evidence",
    "extract_stage_question",
    "normalize_fallback_abstentions",
    "parse_model_synthesis",
    "score_recall_guarded_synthesis",
    "synthesize_question",
    "validate_published_retrieval",
]
