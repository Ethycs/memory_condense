"""Reconstruct a token-level explanation for one held-out attention-code hit."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from tools.analyze_attention_feature_basis import _basis
from tools.assay_attention_feature_retrieval import _normalize, _ordered, _section_covers
from tools.evaluate_dialogue_neighborhood import source_turns
from tools.validate_attention_feature_holdout import (
    DEVELOPMENT,
    HOLDOUT,
    PRIMARY_LIMIT,
    WIDTH,
    _supports,
    _token_observable,
)


def run(validation: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    application = HOLDOUT / "application"
    turns = source_turns(application)
    snapshot = native_spine_store.load(application / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    section_ids = tuple(section.section_id for section in sections)
    questions = tuple(read(HOLDOUT / "questions" / "questions.json")["questions"])
    supports = _supports(turns, read(HOLDOUT / "questions" / "references.json"))

    development = np.load(
        DEVELOPMENT / "attention-values-fp16.npy", allow_pickle=False
    ).astype(np.float32)
    _mean, all_components, _eigenvalues = _basis(
        development, centered=False, maximum=WIDTH
    )
    components = all_components[:WIDTH]
    document_features = np.load(
        validation / "attention-values-fp16.npy", allow_pickle=False
    ).astype(np.float32)
    query_features = np.load(
        validation / "attention-query-values-fp16.npy", allow_pickle=False
    ).astype(np.float32)
    document_codes = _normalize(document_features @ components.T)
    query_codes = _normalize(query_features @ components.T)

    chosen = None
    for row, question in enumerate(questions):
        ranking = _ordered(document_codes @ query_codes[row], section_ids)[:PRIMARY_LIMIT]
        for support in supports[question["question_id"]]:
            for position in ranking:
                singleton = np.asarray([position], dtype=np.int64)
                if _section_covers(support, sections, singleton):
                    chosen = (row, question, int(position), support)
                    break
            if chosen:
                break
        if chosen:
            break
    if chosen is None:
        raise RuntimeError("the held-out attention code retrieved no exact support")

    row, question, position, support = chosen
    runtime = LocalRuntime(output / "runtime", assets=RuntimeAssets.resolve(".cache"))
    try:
        attention_runtime = runtime.attention(output / "attention-cache")
        attention_runtime.prepare()
        import torch

        with (
            attention_runtime.gate,
            torch.cuda.stream(attention_runtime.stream),
            torch.inference_mode(),
        ):
            observable = _token_observable(
                encoder=attention_runtime.scorer.linker.encoder,
                question=question["question"],
                summary=sections[position].summary,
                query_feature=query_features[row],
                components=components,
            )
            attention_runtime.stream.synchronize()
    finally:
        runtime.close()

    report = {
        "format": "memory-condense-attention-token-observable-v1",
        "question_id": question["question_id"],
        "section_id": sections[position].section_id,
        "exact_support": support,
        "raw_inputs_to_qwen": False,
        "observable": observable,
        "interpretation": (
            "Each signed token term is its exact contribution to the rank-16 "
            "document-question cosine. Terms sum to the reported cosine up to "
            "floating-point error."
        ),
    }
    save(output / "report.json", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.validation, args.output)


if __name__ == "__main__":
    main()
