#!/usr/bin/env python3
"""Build a provider-free retrieval-topology and reasoning-operator ledger.

This is a post-hoc diagnostic.  It deliberately reads LongMemEval gold source
labels and reference answers only after the retrieval, answer, and semantic
judge artifacts have been sealed.  It never calls a model or mutates an input
artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    contains_answer,
)
from memory_condense.eval.locked_split import (
    load_split_manifest,
    select_locked_split,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample, load_benchmark


FORMAT = "memory-condense-locked-retrieval-style-ledger-v1"
TAXONOMY_FORMAT = "memory-condense-retrieval-style-taxonomy-v1"
STAGES = (
    ("S0", "causal_graph_coverage_predecessor"),
    ("S1", "direct_episode_additions"),
    ("S2", "representative_episode_additions"),
    ("S3", "artifact_global_closure_additions"),
)
DEFAULT_ROOT = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826"
)
DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)

_INTERVAL_RE = re.compile(
    r"how (?:many (?:days|weeks|months)|long|much time)", re.IGNORECASE
)
_NUMERIC_JOIN_RE = re.compile(
    r"how many|\btotal\b|how much|percentage|older|faster|more expensive|"
    r"higher percentage|discount",
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--answers", type=Path, default=DEFAULT_ROOT / "final-answers.json"
    )
    parser.add_argument(
        "--judge",
        type=Path,
        default=DEFAULT_ROOT / "final-answer-semantic-judge-sol.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_ROOT / "retrieval-style-ledger.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_ROOT / "retrieval-style-ledger.csv",
    )
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_sealed_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=_reject_constant)
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise ValueError(f"artifact is not a canonical JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"artifact SHA-256 sidecar is missing or invalid: {path}")
    return value, digest


def _publish_bytes(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
    else:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists():
        if sidecar.read_bytes() != sidecar_payload:
            raise FileExistsError(f"refusing to replace another digest: {sidecar}")
    else:
        sidecar.write_bytes(sidecar_payload)
    return digest


def _unique_strings(values: Sequence[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _source_geometry(
    sample: BenchmarkSample, question: BenchmarkQuestion
) -> dict[str, Any]:
    ordered_sources = _unique_strings(sample.turn_source_ids)
    expected_sources = _unique_strings(question.evidence_sources)
    missing = [source for source in expected_sources if source not in ordered_sources]
    if missing:
        raise ValueError(
            f"question {question.question_id} labels absent sources: {missing!r}"
        )
    positions = [ordered_sources.index(source) for source in expected_sources]
    count = len(positions)
    if count == 0:
        span = 0
        gaps = 0
        arity = "unlabeled"
        locality = "unlabeled"
        topology = "unlabeled"
    elif count == 1:
        span = 1
        gaps = 0
        arity = "point"
        locality = "point"
        topology = "point"
    else:
        span = max(positions) - min(positions) + 1
        gaps = span - len(set(positions))
        arity = "pair" if count == 2 else "fanout"
        locality = "local" if gaps == 0 else "dispersed"
        topology = (
            f"local_{arity}" if locality == "local" else "dispersed_join"
        )
    return {
        "expected_source_ids": expected_sources,
        "expected_source_positions": positions,
        "expected_source_count": count,
        "source_arity": arity,
        "source_locality": locality,
        "source_span_sessions": span,
        "intervening_source_count": gaps,
        "retrieval_topology": topology,
    }


def _answer_operator(question: BenchmarkQuestion) -> str:
    reference = question.answer.casefold()
    if "_abs" in question.question_id or (
        "information provided is not enough" in reference
    ):
        return "insufficient_evidence"
    if question.category == "single-session-preference":
        return "preference_synthesis"
    if question.category == "knowledge-update":
        return "state_update"
    if question.category == "temporal-reasoning":
        if _INTERVAL_RE.search(question.question):
            return "temporal_interval"
        return "temporal_order_select"
    if question.category == "multi-session":
        if _NUMERIC_JOIN_RE.search(question.question):
            return "numeric_aggregate_compare"
        return "set_or_list_join"
    return "direct_lookup"


def _is_abstention(value: str) -> bool:
    return value.strip().casefold().rstrip(".! ") == "i don't know"


def _stage_metrics(
    stage: Mapping[str, Any],
    *,
    expected_source_ids: Sequence[str],
    reference_answer: str,
) -> dict[str, Any]:
    evidence = stage.get("evidence")
    receipt = stage.get("stage_receipt")
    if not isinstance(evidence, list) or not isinstance(receipt, Mapping):
        raise ValueError("retrieval stage omitted evidence or receipt")
    texts = [str(item["text"]) for item in evidence]
    retrieved_sources = {
        str(item["source_id"])
        for item in evidence
        if item.get("source_id") is not None
    }
    expected = set(expected_source_ids)
    found = len(expected & retrieved_sources)
    components = answer_value_component_coverage(
        reference_answer, len(expected), texts
    )
    component_recall = None if components is None else components.recall
    return {
        "stage_id": str(stage["stage_id"]),
        "evidence_count": len(evidence),
        "added_evidence_count": len(receipt.get("added_evidence_ids", ())),
        "admission_status": str(receipt.get("admission_status", "")),
        "context_token_proxy": int(receipt.get("context_token_proxy", 0)),
        "prompt_token_proxy": int(receipt.get("prompt_token_proxy", 0)),
        "retrieved_expected_source_count": found,
        "evidence_source_recall": (
            None if not expected else found / len(expected)
        ),
        "any_evidence_source": bool(expected and found),
        "all_evidence_sources": bool(expected and found == len(expected)),
        "literal_answer_present": contains_answer(texts, reference_answer),
        "answer_value_component_recall": component_recall,
        "all_answer_value_components": (
            None if component_recall is None else component_recall == 1.0
        ),
    }


def _first_stage(rows: Sequence[Mapping[str, Any]], field: str) -> str:
    for alias, row in zip((alias for alias, _stage in STAGES), rows, strict=True):
        if row[field] is True:
            return alias
    return "unreached"


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _group_summary(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        questions = len(group)
        correct = sum(row["semantic_correct"] is True for row in group)
        abstained = sum(row["abstained"] is True for row in group)
        s1_all = [row for row in group if row["stages"][1]["all_evidence_sources"]]
        s1_literal = [
            row for row in group if row["stages"][1]["literal_answer_present"]
        ]
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "questions": questions,
                "semantic_correct": correct,
                "semantic_accuracy": correct / questions,
                "abstained": abstained,
                "abstention_rate": abstained / questions,
                "s1_any_source": sum(
                    row["stages"][1]["any_evidence_source"] for row in group
                ),
                "s1_all_sources": len(s1_all),
                "s1_literal_answer": len(s1_literal),
                "correct_given_s1_all_sources": sum(
                    row["semantic_correct"] is True for row in s1_all
                ),
                "accuracy_given_s1_all_sources": _ratio(
                    sum(row["semantic_correct"] is True for row in s1_all),
                    len(s1_all),
                ),
                "correct_given_s1_literal_answer": sum(
                    row["semantic_correct"] is True for row in s1_literal
                ),
                "accuracy_given_s1_literal_answer": _ratio(
                    sum(row["semantic_correct"] is True for row in s1_literal),
                    len(s1_literal),
                ),
            }
        )
    return result


def _transition_summary(
    rows: Sequence[Mapping[str, Any]], left: int, right: int
) -> dict[str, Any]:
    prior = [row["stages"][left] for row in rows]
    later = [row["stages"][right] for row in rows]
    new_any = sum(
        not a["any_evidence_source"] and b["any_evidence_source"]
        for a, b in zip(prior, later, strict=True)
    )
    new_all = sum(
        not a["all_evidence_sources"] and b["all_evidence_sources"]
        for a, b in zip(prior, later, strict=True)
    )
    new_literal = sum(
        not a["literal_answer_present"] and b["literal_answer_present"]
        for a, b in zip(prior, later, strict=True)
    )
    eligible_any = sum(not row["any_evidence_source"] for row in prior)
    eligible_all = sum(not row["all_evidence_sources"] for row in prior)
    eligible_literal = sum(not row["literal_answer_present"] for row in prior)
    return {
        "transition": f"S{left}->S{right}",
        "questions": len(rows),
        "added_evidence_rows": sum(row["added_evidence_count"] for row in later),
        "new_expected_sources": sum(
            b["retrieved_expected_source_count"]
            - a["retrieved_expected_source_count"]
            for a, b in zip(prior, later, strict=True)
        ),
        "eligible_any_source_misses": eligible_any,
        "new_any_source_hits": new_any,
        "any_source_rescue_rate": _ratio(new_any, eligible_any),
        "eligible_all_source_misses": eligible_all,
        "new_all_source_hits": new_all,
        "all_source_rescue_rate": _ratio(new_all, eligible_all),
        "eligible_literal_misses": eligible_literal,
        "new_literal_answer_hits": new_literal,
        "literal_rescue_rate": _ratio(new_literal, eligible_literal),
        "any_source_regressions": sum(
            a["any_evidence_source"] and not b["any_evidence_source"]
            for a, b in zip(prior, later, strict=True)
        ),
        "all_source_regressions": sum(
            a["all_evidence_sources"] and not b["all_evidence_sources"]
            for a, b in zip(prior, later, strict=True)
        ),
        "literal_regressions": sum(
            a["literal_answer_present"] and not b["literal_answer_present"]
            for a, b in zip(prior, later, strict=True)
        ),
    }


def _build_rows(
    *,
    samples: Sequence[BenchmarkSample],
    retrieval: Mapping[str, Any],
    answers: Mapping[str, Any],
    judge: Mapping[str, Any],
) -> list[dict[str, Any]]:
    retrieval_rows = list(retrieval.get("questions", ()))
    answer_rows = {row["question_id"]: row for row in answers.get("questions", ())}
    judge_rows = {row["question_id"]: row for row in judge.get("questions", ())}
    ordered_ids = [sample.questions[0].question_id for sample in samples]
    if [row.get("question_id") for row in retrieval_rows] != ordered_ids:
        raise ValueError("retrieval and locked validation question order differ")
    if set(answer_rows) != set(ordered_ids) or set(judge_rows) != set(ordered_ids):
        raise ValueError("answer or judge question population differs")

    result: list[dict[str, Any]] = []
    for ordinal, (sample, retrieval_row) in enumerate(
        zip(samples, retrieval_rows, strict=True)
    ):
        question = sample.questions[0]
        question_id = question.question_id
        answer_row = answer_rows[question_id]
        judge_row = judge_rows[question_id]
        geometry = _source_geometry(sample, question)
        namespaced_sources = [
            f"{sample.sample_id}::{source}"
            for source in geometry["expected_source_ids"]
        ]
        raw_stages = list(retrieval_row.get("stages", ()))
        if [row.get("stage_id") for row in raw_stages] != [
            stage_id for _alias, stage_id in STAGES
        ]:
            raise ValueError(f"question {question_id} changed the S0-S3 ladder")
        stages = [
            _stage_metrics(
                stage,
                expected_source_ids=namespaced_sources,
                reference_answer=question.answer,
            )
            for stage in raw_stages
        ]
        prediction = str(answer_row["answer"]["text"])
        correct = judge_row.get("correct")
        if type(correct) is not bool:
            raise ValueError(f"question {question_id} has no binary judge verdict")
        abstained = _is_abstention(prediction)
        s1 = stages[1]
        if correct:
            boundary = "semantic_correct"
        elif not s1["any_evidence_source"]:
            boundary = "no_labeled_source_in_s1"
        elif not s1["all_evidence_sources"]:
            boundary = "partial_labeled_sources_in_s1"
        elif s1["literal_answer_present"]:
            boundary = "all_sources_and_literal_answer_in_s1"
        else:
            boundary = "all_sources_without_literal_answer_in_s1"
        operator = _answer_operator(question)
        result.append(
            {
                "ordinal": ordinal,
                "question_id": question_id,
                "question": question.question,
                "question_date": question.question_date,
                "reference_answer": question.answer,
                "prediction": prediction,
                "benchmark_category": question.category,
                **geometry,
                "negative_reference": operator == "insufficient_evidence",
                "answer_operator": operator,
                "semantic_correct": correct,
                "abstained": abstained,
                "incorrect_abstention": abstained and not correct,
                "s1_failure_boundary": boundary,
                "first_any_source_stage": _first_stage(
                    stages, "any_evidence_source"
                ),
                "first_all_sources_stage": _first_stage(
                    stages, "all_evidence_sources"
                ),
                "first_literal_answer_stage": _first_stage(
                    stages, "literal_answer_present"
                ),
                "first_all_value_components_stage": _first_stage(
                    stages, "all_answer_value_components"
                ),
                "stages": stages,
            }
        )
    return result


def _taxonomy() -> dict[str, Any]:
    body = {
        "format": TAXONOMY_FORMAT,
        "retrieval_topology": {
            "point": "exactly one labeled answer source",
            "local_pair": "two labeled sources with no intervening session",
            "local_fanout": "three or more labeled sources in one contiguous block",
            "dispersed_join": "two or more labeled sources separated by other sessions",
            "unlabeled": "no labeled answer source",
        },
        "source_geometry": (
            "first-occurrence session order after the LongMemEval loader's "
            "chronological sort, measured inside each original sample"
        ),
        "answer_operator_priority": [
            "insufficient_evidence: _abs ID or insufficiency-form reference",
            "preference_synthesis: single-session-preference category",
            "state_update: knowledge-update category",
            "temporal_interval: temporal category plus interval regex",
            "temporal_order_select: remaining temporal category",
            "numeric_aggregate_compare: multi-session plus numeric regex",
            "set_or_list_join: remaining multi-session category",
            "direct_lookup: remainder",
        ],
        "first_stage_metrics": [
            "any_evidence_source",
            "all_evidence_sources",
            "literal_answer_present",
            "all_answer_value_components",
        ],
        "caveats": [
            "This is a gold-bearing post-hoc diagnostic, not a deployable router.",
            "Answer-session labels do not identify the decisive turn; all-source coverage is not proof that the needed fact was packed.",
            "Literal-answer and value-component hits are conservative surface proxies, not semantic sufficiency judgments.",
            "Operator labels are deterministic benchmark-category and regex buckets, not model-generated annotations.",
        ],
    }
    body["taxonomy_sha256"] = hashlib.sha256(
        _canonical_json_bytes(body)
    ).hexdigest()
    return body


def _build_artifact(
    *,
    rows: list[dict[str, Any]],
    dataset_sha256: str,
    split_sha256: str,
    retrieval_sha256: str,
    answers_sha256: str,
    judge_sha256: str,
    population_identity_sha256: str,
) -> dict[str, Any]:
    transitions = [
        _transition_summary(rows, left, left + 1) for left in range(3)
    ]
    transition_by_topology = []
    for topology in sorted({row["retrieval_topology"] for row in rows}):
        selected = [row for row in rows if row["retrieval_topology"] == topology]
        for left in range(3):
            transition_by_topology.append(
                {
                    "retrieval_topology": topology,
                    **_transition_summary(selected, left, left + 1),
                }
            )
    artifact: dict[str, Any] = {
        "format": FORMAT,
        "status": "provider_free_gold_bearing_posthoc_diagnostic",
        "provider_calls": 0,
        "question_count": len(rows),
        "bindings": {
            "dataset_sha256": dataset_sha256,
            "split_manifest_sha256": split_sha256,
            "population_identity_sha256": population_identity_sha256,
            "retrieval_sha256": retrieval_sha256,
            "answers_sha256": answers_sha256,
            "judge_sha256": judge_sha256,
        },
        "taxonomy": _taxonomy(),
        "summary": {
            "semantic_correct": sum(row["semantic_correct"] for row in rows),
            "semantic_accuracy": sum(row["semantic_correct"] for row in rows)
            / len(rows),
            "abstained": sum(row["abstained"] for row in rows),
            "incorrect_abstentions": sum(
                row["incorrect_abstention"] for row in rows
            ),
            "topology_counts": dict(
                sorted(Counter(row["retrieval_topology"] for row in rows).items())
            ),
            "source_arity_counts": dict(
                sorted(Counter(row["source_arity"] for row in rows).items())
            ),
            "operator_counts": dict(
                sorted(Counter(row["answer_operator"] for row in rows).items())
            ),
            "first_any_source_stage": dict(
                sorted(Counter(row["first_any_source_stage"] for row in rows).items())
            ),
            "first_all_sources_stage": dict(
                sorted(
                    Counter(row["first_all_sources_stage"] for row in rows).items()
                )
            ),
            "first_literal_answer_stage": dict(
                sorted(
                    Counter(row["first_literal_answer_stage"] for row in rows).items()
                )
            ),
            "s1_failure_boundaries": dict(
                sorted(Counter(row["s1_failure_boundary"] for row in rows).items())
            ),
            "by_topology": _group_summary(rows, ("retrieval_topology",)),
            "by_source_arity": _group_summary(rows, ("source_arity",)),
            "by_operator": _group_summary(rows, ("answer_operator",)),
            "by_topology_and_operator": _group_summary(
                rows, ("retrieval_topology", "answer_operator")
            ),
            "by_category": _group_summary(rows, ("benchmark_category",)),
            "stage_transitions": transitions,
            "stage_transitions_by_topology": transition_by_topology,
        },
        "questions": rows,
    }
    artifact["analysis_sha256"] = hashlib.sha256(
        _canonical_json_bytes(artifact)
    ).hexdigest()
    return artifact


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stage_columns = [
        f"{alias}_{metric}"
        for alias, _stage_id in STAGES
        for metric in (
            "evidence_count",
            "added_evidence_count",
            "retrieved_expected_source_count",
            "evidence_source_recall",
            "any_evidence_source",
            "all_evidence_sources",
            "literal_answer_present",
            "answer_value_component_recall",
            "admission_status",
        )
    ]
    fields = [
        "ordinal",
        "question_id",
        "benchmark_category",
        "retrieval_topology",
        "source_arity",
        "source_locality",
        "expected_source_count",
        "source_span_sessions",
        "intervening_source_count",
        "expected_source_ids",
        "expected_source_positions",
        "answer_operator",
        "negative_reference",
        "semantic_correct",
        "abstained",
        "incorrect_abstention",
        "s1_failure_boundary",
        "first_any_source_stage",
        "first_all_sources_stage",
        "first_literal_answer_stage",
        "first_all_value_components_stage",
        *stage_columns,
        "question_date",
        "question",
        "reference_answer",
        "prediction",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for source in rows:
        row = {key: source.get(key) for key in fields}
        row["expected_source_ids"] = json.dumps(
            source["expected_source_ids"], ensure_ascii=False, separators=(",", ":")
        )
        row["expected_source_positions"] = json.dumps(
            source["expected_source_positions"], separators=(",", ":")
        )
        for (alias, _stage_id), stage in zip(STAGES, source["stages"], strict=True):
            for metric in (
                "evidence_count",
                "added_evidence_count",
                "retrieved_expected_source_count",
                "evidence_source_recall",
                "any_evidence_source",
                "all_evidence_sources",
                "literal_answer_present",
                "answer_value_component_recall",
                "admission_status",
            ):
                row[f"{alias}_{metric}"] = stage[metric]
        writer.writerow(row)
    return handle.getvalue().encode("utf-8")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    retrieval, retrieval_sha = _read_sealed_json(args.retrieval.resolve())
    answers, answers_sha = _read_sealed_json(args.answers.resolve())
    judge, judge_sha = _read_sealed_json(args.judge.resolve())
    if answers.get("retrieval_sha256") != retrieval_sha or (
        judge.get("retrieval_sha256") != retrieval_sha
    ):
        raise ValueError("answer or judge retrieval binding changed")
    if judge.get("final_answer_artifact_sha256") != answers_sha:
        raise ValueError("judge no longer binds the answer artifact")
    population_sha = str(retrieval.get("population_identity_sha256", ""))
    if answers.get("population_identity_sha256") != population_sha or (
        judge.get("population_identity_sha256") != population_sha
    ):
        raise ValueError("retrieval, answer, and judge populations differ")
    if retrieval.get("question_count") != 100 or retrieval.get("provider_calls") != 0:
        raise ValueError("retrieval is not the provider-free locked 100Q artifact")
    if tuple(retrieval.get("stage_ids", ())) != tuple(
        stage_id for _alias, stage_id in STAGES
    ):
        raise ValueError("retrieval stage ladder changed")

    dataset = args.dataset.resolve()
    split = args.split_manifest.resolve()
    manifest = load_split_manifest(split)
    selected = select_locked_split(
        load_benchmark(dataset, "longmemeval"),
        dataset_path=dataset,
        manifest=manifest,
        split="validation",
    )
    samples = selected[:100]
    if len(samples) != 100 or any(len(sample.questions) != 1 for sample in samples):
        raise ValueError("locked validation population is not 100 one-question samples")
    rows = _build_rows(
        samples=samples,
        retrieval=retrieval,
        answers=answers,
        judge=judge,
    )
    artifact = _build_artifact(
        rows=rows,
        dataset_sha256=_file_sha256(dataset),
        split_sha256=_file_sha256(split),
        retrieval_sha256=retrieval_sha,
        answers_sha256=answers_sha,
        judge_sha256=judge_sha,
        population_identity_sha256=population_sha,
    )
    json_sha = _publish_bytes(args.output_json.resolve(), _canonical_json_bytes(artifact))
    csv_sha = _publish_bytes(args.output_csv.resolve(), _csv_bytes(rows))
    return artifact, json_sha, csv_sha


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result, json_sha, csv_sha = run(args)
    transitions = result["summary"]["stage_transitions"]
    print(
        f"Published {result['question_count']}Q provider-free style ledger: "
        f"JSON={json_sha}; CSV={csv_sha}",
        flush=True,
    )
    print(
        "; ".join(
            f"{row['transition']} added={row['added_evidence_rows']} "
            f"any/all/literal fixes={row['new_any_source_hits']}/"
            f"{row['new_all_source_hits']}/{row['new_literal_answer_hits']}"
            for row in transitions
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
