"""Screen Qwen-token TF-IDF as a rarity correction for attention-code retrieval."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.config import RuntimeAssets
from tools.analyze_attention_feature_basis import _basis
from tools.assay_attention_feature_retrieval import (
    ORIGINAL,
    REFERENCE,
    _normalize,
    _ordered,
    _reciprocal_rank_scores,
    _score_arm,
)
from tools.characterize_native_experiments import cases
from tools.evaluate_dialogue_neighborhood import source_turns
from tools.validate_attention_feature_holdout import (
    DEVELOPMENT,
    HOLDOUT,
    _measure,
    _supports,
)


HOLDOUT_FEATURES = Path("eval_results/attention-feature-holdout-20261009-r1")


def _qwen_tfidf(documents, queries):
    tokenizer = AutoTokenizer.from_pretrained(
        RuntimeAssets.resolve(".cache").qwen, local_files_only=True
    )

    def tokens(text: str) -> list[str]:
        return [str(token) for token in tokenizer.encode(text, add_special_tokens=False)]

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
        "vocabulary": len(vectorizer.vocabulary_),
        "document_nonzero_cells": int(document_matrix.nnz),
        "query_nonzero_cells": int(query_matrix.nnz),
    }


def _rankings(*, attention_scores, tfidf_scores, section_ids, question_ids):
    result = {name: {} for name in ("tfidf", "attention_rank16", "rrf_equal", "rrf_attention_2x")}
    for row, question_id in enumerate(question_ids):
        attention_rrf = _reciprocal_rank_scores(attention_scores[row], section_ids)
        tfidf_rrf = _reciprocal_rank_scores(tfidf_scores[row], section_ids)
        result["tfidf"][question_id] = _ordered(tfidf_scores[row], section_ids)
        result["attention_rank16"][question_id] = _ordered(
            attention_scores[row], section_ids
        )
        result["rrf_equal"][question_id] = _ordered(
            attention_rrf + tfidf_rrf, section_ids
        )
        result["rrf_attention_2x"][question_id] = _ordered(
            2.0 * attention_rrf + tfidf_rrf, section_ids
        )
    return result


def _development(components):
    source = read(ORIGINAL / "source.json")
    directory = Path(source["directory"])
    turns = source_turns(directory)
    snapshot = native_spine_store.load(directory / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    questions = tuple(cases())
    references = read(REFERENCE)
    documents = np.load(DEVELOPMENT / "attention-values-fp16.npy", allow_pickle=False).astype(np.float32)
    queries = np.load(DEVELOPMENT / "attention-query-values-fp16.npy", allow_pickle=False).astype(np.float32)
    attention_scores = _normalize(queries @ components.T) @ _normalize(
        documents @ components.T
    ).T
    started = time.perf_counter()
    tfidf_scores, tfidf_receipt = _qwen_tfidf(
        [section.summary for section in sections],
        [question["question"] for question in questions],
    )
    tfidf_receipt["elapsed_s"] = time.perf_counter() - started
    rankings = _rankings(
        attention_scores=attention_scores,
        tfidf_scores=tfidf_scores,
        section_ids=tuple(section.section_id for section in sections),
        question_ids=tuple(question["id"] for question in questions),
    )
    return {
        "sections": len(sections),
        "questions": len(questions),
        "tfidf": tfidf_receipt,
        "arms": {
            arm: {
                str(limit): _score_arm(
                    rankings=arm_rankings,
                    sections=sections,
                    questions=questions,
                    reference=references,
                    limit=limit,
                )[0]
                for limit in (8, 16)
            }
            for arm, arm_rankings in rankings.items()
        },
    }


def _holdout(components):
    application = HOLDOUT / "application"
    turns = source_turns(application)
    snapshot = native_spine_store.load(application / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    questions = tuple(read(HOLDOUT / "questions" / "questions.json")["questions"])
    supports = _supports(turns, read(HOLDOUT / "questions" / "references.json"))
    documents = np.load(HOLDOUT_FEATURES / "attention-values-fp16.npy", allow_pickle=False).astype(np.float32)
    queries = np.load(HOLDOUT_FEATURES / "attention-query-values-fp16.npy", allow_pickle=False).astype(np.float32)
    attention_scores = _normalize(queries @ components.T) @ _normalize(
        documents @ components.T
    ).T
    started = time.perf_counter()
    tfidf_scores, tfidf_receipt = _qwen_tfidf(
        [section.summary for section in sections],
        [question["question"] for question in questions],
    )
    tfidf_receipt["elapsed_s"] = time.perf_counter() - started
    section_ids = tuple(section.section_id for section in sections)
    rankings = _rankings(
        attention_scores=attention_scores,
        tfidf_scores=tfidf_scores,
        section_ids=section_ids,
        question_ids=tuple(question["question_id"] for question in questions),
    )
    section_position = {section_id: index for index, section_id in enumerate(section_ids)}
    arm_results = {}
    for arm, arm_rankings in rankings.items():
        arm_results[arm] = {}
        for limit in (8, 16):
            metrics, novel_questions = _measure(
                rankings=arm_rankings,
                sections=sections,
                section_position=section_position,
                questions=questions,
                supports=supports,
                limit=limit,
            )
            metrics["novel_questions"] = novel_questions
            arm_results[arm][str(limit)] = metrics
    return {
        "sections": len(sections),
        "questions": len(questions),
        "tfidf": tfidf_receipt,
        "arms": arm_results,
    }


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    development = np.load(DEVELOPMENT / "attention-values-fp16.npy", allow_pickle=False).astype(np.float32)
    _mean, all_components, _eigenvalues = _basis(
        development, centered=False, maximum=16
    )
    components = all_components[:16]
    report = {
        "format": "memory-condense-qwen-token-tfidf-attention-assay-v1",
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "tfidf_tokenizer": "pinned local Qwen3-8B tokenizer IDs",
        "development": _development(components),
        "holdout": _holdout(components),
        "limitations": [
            "TF-IDF is compiled independently for each stored-summary corpus.",
            "Fusion weights are fixed controls, not tuned production policies.",
            "This screen reranks whole summaries; it does not yet IDF-weight A@V source-token transport.",
        ],
    }
    save(output / "report.json", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
