"""Run a resumable, provider-free retrieval campaign over persisted histories.

The runner treats the ten-history native-spine battery as an artifact graph:

* persisted snapshots replace reingestion;
* authored questions and exact references replace regeneration;
* production answer packets supply the already-computed baseline routes;
* the development archive supplies one fixed rank-16 attention basis;
* compatible Qwen summary/question features are content-addressed and reused;
* TF-IDF is rebuilt locally from stored summaries because it is cheap.

Qwen receives stored summaries and question strings only. Raw turns are opened
locally after ranking solely to measure exact support coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    expected_prefix_checkpoint_sha256,
)
from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, read_sealed_json, save
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from tools.analyze_attention_feature_basis import _basis
from tools.assay_attention_feature_retrieval import (
    _normalize,
    _ordered,
    _reciprocal_rank_scores,
    _section_covers,
)
from tools.evaluate_dialogue_neighborhood import source_turns
from tools.validate_attention_feature_holdout import _supports


FORMAT = "memory-condense-attention-tfidf-artifact-campaign-v1"
FEATURE_FORMAT = "memory-condense-attention-value-feature-cache-v1"
BASIS_FORMAT = "memory-condense-attention-rank16-basis-cache-v1"
RESULT_FORMAT = "memory-condense-attention-tfidf-history-result-v1"
SCORING_METHOD = "exact-section-support-union-v1"
ATTENTION_LAYER = 4
ATTENTION_WIDTH = 4096
BASIS_RANK = 16
DEFAULT_LIMITS = (8, 16)
DEFAULT_CAMPAIGN = Path("eval_results/native-spine-ten100-20260922-r1")
DEFAULT_DEVELOPMENT = Path("eval_results/attention-feature-retrieval-20261009-r1")
DEFAULT_LEGACY_HOLDOUT = Path("eval_results/attention-feature-holdout-20261009-r1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_histories(value: str, available: int = 10) -> tuple[int, ...]:
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("history selection contains an empty item")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("history range ends before it starts")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    if not selected or min(selected) < 1 or max(selected) > available:
        raise ValueError(f"histories must lie in [1, {available}]")
    return tuple(sorted(selected))


def _encoder_identity() -> dict[str, Any]:
    return {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": expected_prefix_checkpoint_sha256(6),
        "prefix_layers": 6,
        "attention_layer": ATTENTION_LAYER,
        "representation": "mean-pooled concatenated per-head A@V before W_O",
        "inference_dtype": "float16",
        "stored_dtype": "float16",
        "width": ATTENTION_WIDTH,
        "raw_inputs_to_qwen": False,
    }


def _feature_input_fingerprint(
    section_rows: Sequence[tuple[str, str]],
    question_rows: Sequence[tuple[str, str]],
    encoder_identity: dict[str, Any] | None = None,
) -> str:
    """Bind expensive features only to their exact ordered model inputs."""
    return identity_sha256(
        {
            "format": FEATURE_FORMAT,
            "encoder": encoder_identity or _encoder_identity(),
            "sections": [
                {"id": section_id, "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest()}
                for section_id, summary in section_rows
            ],
            "questions": [
                {"id": question_id, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                for question_id, text in question_rows
            ],
        }
    )


def _array_receipt(path: Path, value: np.ndarray | None = None) -> dict[str, Any]:
    array = np.load(path, allow_pickle=False, mmap_mode="r") if value is None else value
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
    }


def _validated_array(path: Path, receipt: dict[str, Any]) -> np.ndarray:
    if not path.is_file() or _sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"cached array digest changed: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if list(array.shape) != receipt.get("shape") or str(array.dtype) != receipt.get("dtype"):
        raise ValueError(f"cached array metadata changed: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"cached array contains non-finite values: {path}")
    return array


def _history_data(history: Path) -> dict[str, Any]:
    application = history / "application"
    turns = source_turns(application)
    snapshot = native_spine_store.load(application / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    section_ids = tuple(section.section_id for section in sections)
    section_position = {section_id: index for index, section_id in enumerate(section_ids)}
    if len(section_position) != len(section_ids):
        raise ValueError(f"duplicate section IDs in {history}")

    question_path = history / "questions" / "questions.json"
    reference_path = history / "questions" / "references.json"
    question_file = read(question_path)
    questions = tuple(question_file["questions"])
    if tuple(int(question["ordinal"]) for question in questions) != tuple(range(len(questions))):
        raise ValueError(f"questions are not in ordinal order: {history}")
    if len({question["question_id"] for question in questions}) != len(questions):
        raise ValueError(f"duplicate question IDs in {history}")

    references = read(reference_path)
    supports = _supports(turns, references)
    if set(supports) != {question["question_id"] for question in questions}:
        raise ValueError(f"question/reference population differs: {history}")

    baseline: dict[str, np.ndarray] = {}
    baseline_ids: dict[str, list[str]] = {}
    answer_hashes = []
    for question in questions:
        ordinal = int(question["ordinal"])
        answer_path = history / "answers" / f"{ordinal:03}.response.json"
        response = read(answer_path)
        ids = [row["section"]["section_id"] for row in response["hydration"]["sections"]]
        baseline_ids[question["question_id"]] = ids
        baseline[question["question_id"]] = np.asarray(
            [section_position[section_id] for section_id in ids if section_id in section_position],
            dtype=np.int64,
        )
        answer_hashes.append((answer_path.name, _sha256_file(answer_path)))

    section_rows = tuple((section.section_id, section.summary) for section in sections)
    question_rows = tuple((question["question_id"], question["question"]) for question in questions)
    feature_fingerprint = _feature_input_fingerprint(section_rows, question_rows)
    score_fingerprint = identity_sha256(
        {
            "format": SCORING_METHOD,
            "snapshot": dict(snapshot.receipt),
            "questions_sha256": _sha256_file(question_path),
            "references_sha256": _sha256_file(reference_path),
            "answer_packets": answer_hashes,
            "baseline_section_ids": baseline_ids,
        }
    )
    return {
        "path": history,
        "name": history.name,
        "turns": turns,
        "snapshot": snapshot,
        "sections": sections,
        "section_ids": section_ids,
        "section_position": section_position,
        "questions": questions,
        "supports": supports,
        "baseline": baseline,
        "feature_fingerprint": feature_fingerprint,
        "score_fingerprint": score_fingerprint,
    }


def _basis_identity(development: Path) -> tuple[str, dict[str, Any]]:
    source = development / "attention-values-fp16.npy"
    report = read(development / "report.json")
    source_sha256 = _sha256_file(source)
    declared = report.get("artifacts", {}).get("attention_values_fp16_sha256")
    if declared is not None and declared != source_sha256:
        raise ValueError("development attention feature digest differs from its report")
    payload = {
        "format": BASIS_FORMAT,
        "source": str(development.resolve()),
        "source_sha256": source_sha256,
        "source_shape": list(np.load(source, allow_pickle=False, mmap_mode="r").shape),
        "centered": False,
        "rank": BASIS_RANK,
    }
    return identity_sha256(payload), payload


def _load_or_build_basis(development: Path, output: Path) -> tuple[np.ndarray, str, str]:
    fingerprint, payload = _basis_identity(development)
    folder = output / "basis" / fingerprint
    manifest_path = folder / "manifest.json"
    components_path = folder / "components-fp32.npy"
    if manifest_path.is_file():
        manifest = read(manifest_path)
        if manifest.get("fingerprint") != fingerprint or manifest.get("basis") != payload:
            raise ValueError(f"basis cache manifest is inconsistent: {manifest_path}")
        components = _validated_array(components_path, manifest["components"])
        if components.shape != (BASIS_RANK, ATTENTION_WIDTH) or components.dtype != np.float32:
            raise ValueError("cached attention basis has the wrong shape or dtype")
        return np.asarray(components), fingerprint, "reused"

    development_features = np.load(
        development / "attention-values-fp16.npy", allow_pickle=False
    ).astype(np.float32)
    _mean, all_components, _eigenvalues = _basis(
        development_features, centered=False, maximum=BASIS_RANK
    )
    components = np.asarray(all_components[:BASIS_RANK], dtype=np.float32)
    if components.shape != (BASIS_RANK, ATTENTION_WIDTH):
        raise ValueError("development archive did not produce the declared rank-16 basis")
    _atomic_npy(components_path, components)
    save(
        manifest_path,
        {
            "format": BASIS_FORMAT,
            "fingerprint": fingerprint,
            "basis": payload,
            "components": _array_receipt(components_path),
        },
    )
    return components, fingerprint, "built"


def _feature_paths(output: Path, fingerprint: str) -> tuple[Path, Path, Path]:
    folder = output / "features" / fingerprint
    return (
        folder / "manifest.json",
        folder / "attention-values-fp16.npy",
        folder / "attention-query-values-fp16.npy",
    )


def _read_feature_cache(
    output: Path, data: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    manifest_path, document_path, query_path = _feature_paths(
        output, data["feature_fingerprint"]
    )
    if not manifest_path.is_file():
        return None
    manifest = read(manifest_path)
    if (
        manifest.get("format") != FEATURE_FORMAT
        or manifest.get("fingerprint") != data["feature_fingerprint"]
        or manifest.get("encoder") != _encoder_identity()
    ):
        raise ValueError(f"feature cache manifest is inconsistent: {manifest_path}")
    documents = _validated_array(document_path, manifest["documents"])
    queries = _validated_array(query_path, manifest["queries"])
    expected_documents = (len(data["sections"]), ATTENTION_WIDTH)
    expected_queries = (len(data["questions"]), ATTENTION_WIDTH)
    if documents.shape != expected_documents or queries.shape != expected_queries:
        raise ValueError("cached attention feature population changed")
    return documents, queries, manifest


def _publish_feature_cache(
    output: Path,
    data: dict[str, Any],
    documents: np.ndarray,
    queries: np.ndarray,
    provenance: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path, document_path, query_path = _feature_paths(
        output, data["feature_fingerprint"]
    )
    documents = np.asarray(documents, dtype=np.float16)
    queries = np.asarray(queries, dtype=np.float16)
    if documents.shape != (len(data["sections"]), ATTENTION_WIDTH):
        raise ValueError("document attention features have the wrong shape")
    if queries.shape != (len(data["questions"]), ATTENTION_WIDTH):
        raise ValueError("query attention features have the wrong shape")
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        raise ValueError("attention features contain non-finite values")
    _atomic_npy(document_path, documents)
    _atomic_npy(query_path, queries)
    manifest = {
        "format": FEATURE_FORMAT,
        "fingerprint": data["feature_fingerprint"],
        "encoder": _encoder_identity(),
        "sections": len(data["sections"]),
        "questions": len(data["questions"]),
        "documents": _array_receipt(document_path),
        "queries": _array_receipt(query_path),
        "provenance": provenance,
    }
    save(manifest_path, manifest)
    return (
        np.load(document_path, allow_pickle=False, mmap_mode="r"),
        np.load(query_path, allow_pickle=False, mmap_mode="r"),
        manifest,
    )


def _legacy_is_adoptable(legacy: Path, data: dict[str, Any]) -> bool:
    try:
        report = read(legacy / "report.json")
        archive = Path(report["holdout_archive"]).resolve()
        documents = np.load(
            legacy / "attention-values-fp16.npy", allow_pickle=False, mmap_mode="r"
        )
        queries = np.load(
            legacy / "attention-query-values-fp16.npy", allow_pickle=False, mmap_mode="r"
        )
        return bool(
            report.get("format") == "memory-condense-attention-feature-holdout-v1"
            and archive == data["path"].resolve()
            and int(report["sections"]) == len(data["sections"])
            and int(report["questions"]) == len(data["questions"])
            and documents.shape == (len(data["sections"]), ATTENTION_WIDTH)
            and queries.shape == (len(data["questions"]), ATTENTION_WIDTH)
            and documents.dtype == np.float16
            and queries.dtype == np.float16
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def _adopt_legacy_features(
    legacy: Path, output: Path, data: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not _legacy_is_adoptable(legacy, data):
        raise ValueError("legacy attention artifacts do not match this history")
    manifest_path, document_path, query_path = _feature_paths(
        output, data["feature_fingerprint"]
    )
    _atomic_copy(legacy / "attention-values-fp16.npy", document_path)
    _atomic_copy(legacy / "attention-query-values-fp16.npy", query_path)
    report_artifact = read_sealed_json(legacy / "report.json")
    documents = np.load(document_path, allow_pickle=False, mmap_mode="r")
    queries = np.load(query_path, allow_pickle=False, mmap_mode="r")
    manifest = {
        "format": FEATURE_FORMAT,
        "fingerprint": data["feature_fingerprint"],
        "encoder": _encoder_identity(),
        "sections": len(data["sections"]),
        "questions": len(data["questions"]),
        "documents": _array_receipt(document_path, documents),
        "queries": _array_receipt(query_path, queries),
        "provenance": {
            "kind": "adopted_legacy_assay",
            "source": str(legacy.resolve()),
            "source_report_sha256": report_artifact.sha256,
            "validation": "archive path, population, shape, dtype, and local artifact digests",
            "limitation": "the legacy report predates ordered-input fingerprints",
        },
    }
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        raise ValueError("legacy attention features contain non-finite values")
    save(manifest_path, manifest)
    return documents, queries, manifest


def _encode_features(
    output: Path,
    data: dict[str, Any],
    attention_runtime: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    attention_runtime.prepare()
    encoder = attention_runtime.scorer.linker.encoder
    with (
        attention_runtime.gate,
        torch.cuda.stream(attention_runtime.stream),
        torch.inference_mode(),
    ):
        started = time.perf_counter()
        documents = encoder.encode_attention_values(
            [section.summary for section in data["sections"]],
            layers=(ATTENTION_LAYER,),
            batch_size=8,
        )[ATTENTION_LAYER].numpy()
        attention_runtime.stream.synchronize()
        document_s = time.perf_counter() - started
        started = time.perf_counter()
        queries = encoder.encode_attention_values(
            [question["question"] for question in data["questions"]],
            layers=(ATTENTION_LAYER,),
            batch_size=8,
        )[ATTENTION_LAYER].numpy()
        attention_runtime.stream.synchronize()
        query_s = time.perf_counter() - started
    return _publish_feature_cache(
        output,
        data,
        documents,
        queries,
        {
            "kind": "encoded",
            "history": str(data["path"].resolve()),
            "qwen_document_encode_s": document_s,
            "qwen_query_encode_s": query_s,
            "provider_calls": 0,
            "raw_inputs_to_qwen": False,
        },
    )


def _qwen_tfidf(
    documents: Sequence[str], queries: Sequence[str], tokenizer: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    def tokens(text: str) -> list[str]:
        return [str(token) for token in tokenizer.encode(text, add_special_tokens=False)]

    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer=tokens,
        lowercase=False,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True,
    )
    document_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(queries)
    scores = (document_matrix @ query_matrix.T).toarray().astype(np.float32).T
    return scores, {
        "tokenizer": "pinned local Qwen3-8B token IDs",
        "vocabulary": len(vectorizer.vocabulary_),
        "document_nonzero_cells": int(document_matrix.nnz),
        "query_nonzero_cells": int(query_matrix.nnz),
        "elapsed_s": time.perf_counter() - started,
        "provider_calls": 0,
    }


def _rankings(
    *,
    section_ids: Sequence[str],
    question_ids: Sequence[str],
    attention_scores: np.ndarray | None,
    tfidf_scores: np.ndarray | None,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    if tfidf_scores is not None:
        result["tfidf"] = {
            question_id: _ordered(tfidf_scores[row], section_ids)
            for row, question_id in enumerate(question_ids)
        }
    if attention_scores is not None:
        result["attention_rank16"] = {
            question_id: _ordered(attention_scores[row], section_ids)
            for row, question_id in enumerate(question_ids)
        }
    if attention_scores is not None and tfidf_scores is not None:
        result["rrf_equal"] = {}
        result["rrf_attention_2x"] = {}
        for row, question_id in enumerate(question_ids):
            attention_rrf = _reciprocal_rank_scores(attention_scores[row], section_ids)
            tfidf_rrf = _reciprocal_rank_scores(tfidf_scores[row], section_ids)
            result["rrf_equal"][question_id] = _ordered(
                attention_rrf + tfidf_rrf, section_ids
            )
            result["rrf_attention_2x"][question_id] = _ordered(
                2.0 * attention_rrf + tfidf_rrf, section_ids
            )
    return result


_COUNT_KEYS = (
    "questions",
    "support_total",
    "baseline_supports",
    "candidate_supports",
    "union_supports",
    "novel_supports_over_baseline",
    "baseline_complete_questions",
    "candidate_complete_questions",
    "union_complete_questions",
    "newly_complete_questions",
)


def _empty_counts() -> dict[str, int]:
    return {key: 0 for key in _COUNT_KEYS}


def _rates(counts: dict[str, Any]) -> dict[str, Any]:
    result = dict(counts)
    supports = max(int(result["support_total"]), 1)
    questions = max(int(result["questions"]), 1)
    result.update(
        {
            "baseline_support_recall": result["baseline_supports"] / supports,
            "candidate_support_recall": result["candidate_supports"] / supports,
            "union_support_recall": result["union_supports"] / supports,
            "baseline_complete_rate": result["baseline_complete_questions"] / questions,
            "candidate_complete_rate": result["candidate_complete_questions"] / questions,
            "union_complete_rate": result["union_complete_questions"] / questions,
        }
    )
    return result


def _measure(
    data: dict[str, Any], rankings: dict[str, np.ndarray], limit: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = _empty_counts()
    categories: dict[str, dict[str, int]] = {}
    novel_rows = []
    sections = data["sections"]
    for question in data["questions"]:
        qid = question["question_id"]
        baseline = data["baseline"][qid]
        selected = rankings[qid][:limit]
        baseline_values = tuple(
            _section_covers(support, sections, baseline) for support in data["supports"][qid]
        )
        selected_values = tuple(
            _section_covers(support, sections, selected) for support in data["supports"][qid]
        )
        union_values = tuple(old or new for old, new in zip(baseline_values, selected_values))
        row = {
            "questions": 1,
            "support_total": len(union_values),
            "baseline_supports": sum(baseline_values),
            "candidate_supports": sum(selected_values),
            "union_supports": sum(union_values),
            "novel_supports_over_baseline": sum(union_values) - sum(baseline_values),
            "baseline_complete_questions": int(all(baseline_values)),
            "candidate_complete_questions": int(all(selected_values)),
            "union_complete_questions": int(all(union_values)),
            "newly_complete_questions": int(all(union_values) and not all(baseline_values)),
        }
        category = str(question.get("category", "uncategorized"))
        bucket = categories.setdefault(category, _empty_counts())
        for key in _COUNT_KEYS:
            counts[key] += int(row[key])
            bucket[key] += int(row[key])

        novel = []
        for index, (old, new) in enumerate(zip(baseline_values, selected_values)):
            if not new or old:
                continue
            support = data["supports"][qid][index]
            selected_ids = [
                sections[int(position)].section_id
                for position in selected
                if _section_covers(support, sections, np.asarray([position]))
            ]
            novel.append({**support, "selected_section_ids": selected_ids})
        if novel:
            novel_rows.append(
                {
                    "question_id": qid,
                    "question": question["question"],
                    "category": category,
                    "novel_supports": novel,
                    "newly_complete": bool(row["newly_complete_questions"]),
                }
            )
    metrics = _rates(counts)
    metrics["limit"] = limit
    metrics["categories"] = {name: _rates(value) for name, value in categories.items()}
    return metrics, novel_rows


def _aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, dict[str, dict[str, int]]] = {}
    for result in results:
        for arm, limits in result["arms"].items():
            arm_bucket = arms.setdefault(arm, {})
            for limit, measurement in limits.items():
                bucket = arm_bucket.setdefault(limit, _empty_counts())
                for key in _COUNT_KEYS:
                    bucket[key] += int(measurement[key])
    return {
        arm: {limit: _rates(counts) for limit, counts in limits.items()}
        for arm, limits in arms.items()
    }


def _result_fingerprint(
    data: dict[str, Any], basis_fingerprint: str | None, mode: str, limits: Sequence[int]
) -> str:
    return identity_sha256(
        {
            "format": RESULT_FORMAT,
            "scoring_method": SCORING_METHOD,
            "score_input": data["score_fingerprint"],
            "feature_input": data["feature_fingerprint"] if mode != "tfidf" else None,
            "basis": basis_fingerprint if mode != "tfidf" else None,
            "mode": mode,
            "limits": list(limits),
        }
    )


def run(
    *,
    campaign: Path,
    output: Path,
    histories: Sequence[int],
    mode: str,
    limits: Sequence[int],
    development: Path,
    legacy_features: Path | None,
    assets: RuntimeAssets,
    dry_run: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True) if not dry_run else None
    basis_fingerprint = None
    components = None
    basis_status = "not_requested"
    if mode != "tfidf":
        basis_fingerprint, _basis_payload = _basis_identity(development)
        if not dry_run:
            components, basis_fingerprint, basis_status = _load_or_build_basis(
                development, output
            )

    runtime: LocalRuntime | None = None
    attention_runtime = None
    tokenizer = None
    results = []
    plans = []
    try:
        for ordinal in histories:
            history = campaign / f"history-{ordinal:02}"
            started = time.perf_counter()
            data = _history_data(history)
            result_fingerprint = _result_fingerprint(data, basis_fingerprint, mode, limits)
            result_path = output / "results" / data["name"] / f"{result_fingerprint}.json"
            cached_features = (
                _read_feature_cache(output, data) if mode != "tfidf" and output.exists() else None
            )
            legacy_adoptable = bool(
                mode != "tfidf"
                and cached_features is None
                and legacy_features is not None
                and _legacy_is_adoptable(legacy_features, data)
            )
            plan = {
                "history": data["name"],
                "sections": len(data["sections"]),
                "questions": len(data["questions"]),
                "result_cached": result_path.is_file(),
                "attention_features": (
                    "not_requested"
                    if mode == "tfidf"
                    else "reused"
                    if cached_features is not None
                    else "adopt_legacy"
                    if legacy_adoptable
                    else "encode_missing"
                ),
                "reingest": False,
                "regenerate_questions": False,
                "rerun_production_baseline": False,
            }
            plans.append(plan)
            print(json.dumps({"phase": "plan", **plan}), flush=True)
            if dry_run:
                continue
            if result_path.is_file():
                results.append(read(result_path))
                continue

            feature_manifest = None
            attention_scores = None
            if mode != "tfidf":
                if cached_features is None and legacy_adoptable:
                    cached_features = _adopt_legacy_features(
                        legacy_features, output, data  # type: ignore[arg-type]
                    )
                if cached_features is None:
                    if runtime is None:
                        runtime = LocalRuntime(
                            output / "runtime", assets=assets
                        )
                        attention_runtime = runtime.attention(output / "attention-cache")
                    cached_features = _encode_features(output, data, attention_runtime)
                document_features, query_features, feature_manifest = cached_features
                assert components is not None
                document_codes = _normalize(
                    np.asarray(document_features, dtype=np.float32) @ components.T
                )
                query_codes = _normalize(
                    np.asarray(query_features, dtype=np.float32) @ components.T
                )
                attention_scores = query_codes @ document_codes.T

            tfidf_scores = None
            tfidf_receipt = None
            if mode != "attention":
                if tokenizer is None:
                    if attention_runtime is not None and attention_runtime.scorer is not None:
                        tokenizer = attention_runtime.scorer.linker.encoder.tokenizer
                    else:
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(
                            assets.qwen, local_files_only=True
                        )
                tfidf_scores, tfidf_receipt = _qwen_tfidf(
                    [section.summary for section in data["sections"]],
                    [question["question"] for question in data["questions"]],
                    tokenizer,
                )

            all_rankings = _rankings(
                section_ids=data["section_ids"],
                question_ids=tuple(
                    question["question_id"] for question in data["questions"]
                ),
                attention_scores=attention_scores,
                tfidf_scores=tfidf_scores,
            )
            arm_results = {}
            details = {}
            for arm, arm_rankings in all_rankings.items():
                arm_results[arm] = {}
                details[arm] = {}
                for limit in limits:
                    measurement, novel = _measure(data, arm_rankings, limit)
                    arm_results[arm][str(limit)] = measurement
                    details[arm][str(limit)] = novel

            result = {
                "format": RESULT_FORMAT,
                "history": data["name"],
                "history_path": str(history.resolve()),
                "fingerprint": result_fingerprint,
                "score_input_fingerprint": data["score_fingerprint"],
                "feature_input_fingerprint": data["feature_fingerprint"],
                "provider_calls": 0,
                "history_reingestions": 0,
                "question_generations": 0,
                "production_baseline_calls": 0,
                "raw_inputs_to_qwen": False,
                "sections": len(data["sections"]),
                "questions": len(data["questions"]),
                "basis_fingerprint": basis_fingerprint,
                "basis_status": basis_status,
                "feature_cache": (
                    None
                    if feature_manifest is None
                    else {
                        "manifest": str(
                            _feature_paths(output, data["feature_fingerprint"])[0].resolve()
                        ),
                        "provenance": feature_manifest["provenance"],
                    }
                ),
                "tfidf": tfidf_receipt,
                "arms": arm_results,
                "novel_supports": details,
                "elapsed_s": time.perf_counter() - started,
            }
            artifact = save(result_path, result)
            results.append(artifact.payload)
            print(
                json.dumps(
                    {
                        "phase": "history_complete",
                        "history": data["name"],
                        "result": str(result_path),
                        "elapsed_s": result["elapsed_s"],
                    }
                ),
                flush=True,
            )
    finally:
        if runtime is not None:
            runtime.close()

    if dry_run:
        return {
            "format": FORMAT,
            "dry_run": True,
            "mode": mode,
            "basis_fingerprint": basis_fingerprint,
            "plans": plans,
        }

    aggregate = {
        "format": FORMAT,
        "campaign": str(campaign.resolve()),
        "histories": [f"history-{ordinal:02}" for ordinal in histories],
        "mode": mode,
        "limits": list(limits),
        "provider_calls": 0,
        "history_reingestions": 0,
        "question_generations": 0,
        "production_baseline_calls": 0,
        "raw_inputs_to_qwen": False,
        "basis_fingerprint": basis_fingerprint,
        "artifact_policy": {
            "snapshots": "reuse persisted native-spine snapshots",
            "questions": "reuse authored questions and exact references",
            "baseline": "reuse production hydration packets",
            "qwen": "reuse exact content-addressed summary/question features",
            "tfidf": "recompile cheaply from stored summaries",
        },
        "aggregate": _aggregate(results),
        "history_results": [
            {
                "history": result["history"],
                "fingerprint": result["fingerprint"],
                "elapsed_s": result["elapsed_s"],
            }
            for result in results
        ],
    }
    aggregate_fingerprint = identity_sha256(
        {
            "format": FORMAT,
            "results": [result["fingerprint"] for result in results],
        }
    )
    report_path = output / "reports" / f"{aggregate_fingerprint}.json"
    save(report_path, aggregate)
    print(json.dumps({"phase": "complete", "report": str(report_path)}), flush=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--histories",
        default="1-10",
        help="comma-separated ordinals and inclusive ranges, for example 1-3,7",
    )
    parser.add_argument(
        "--mode", choices=("all", "attention", "tfidf"), default="all"
    )
    parser.add_argument("--limits", default="8,16")
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument(
        "--legacy-features",
        type=Path,
        default=DEFAULT_LEGACY_HOLDOUT,
        help="optional prior history-01 feature assay to adopt after validation",
    )
    parser.add_argument("--assets", type=Path, default=Path(".cache"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    histories = _parse_histories(args.histories)
    limits = tuple(sorted({int(value) for value in args.limits.split(",")}))
    if not limits or min(limits) < 1:
        parser.error("--limits must contain positive integers")
    report = run(
        campaign=args.campaign,
        output=args.output,
        histories=histories,
        mode=args.mode,
        limits=limits,
        development=args.development,
        legacy_features=args.legacy_features,
        assets=RuntimeAssets.resolve(args.assets),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
