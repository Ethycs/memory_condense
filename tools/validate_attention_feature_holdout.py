"""Validate the predeclared rank-16 attention code on a separate 100Q archive.

The SVD basis is fitted only on the prior development archive's stored summary
features. Qwen receives held-out stored summaries and questions, never raw
transcript bodies. Exact raw spans are opened only to score support coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from memory_condense.persistence import native_spine_store
from memory_condense.runtime.artifacts import read, save
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from memory_condense.modeling.qwen_prefix import gqa_key_group_index
from tools.analyze_attention_feature_basis import _basis
from tools.assay_attention_feature_retrieval import _normalize, _ordered, _section_covers
from tools.evaluate_dialogue_neighborhood import source_turns


DEVELOPMENT = Path("eval_results/attention-feature-retrieval-20261009-r1")
HOLDOUT = Path("eval_results/native-spine-ten100-20260922-r1/history-01")
WIDTH = 16
PRIMARY_LIMIT = 8


def _supports(turns, references: dict) -> dict[str, tuple[dict, ...]]:
    by_source: dict[str, list[object]] = {}
    for turn in turns:
        by_source.setdefault(turn.source_id, []).append(turn)
    result = {}
    for reference in references["references"]:
        source_id = "native-source-" + reference["source"]["source"]["occurrence_id"]
        rows = []
        for support in reference["supports"]:
            quote = support["quote"]
            source_turns_for_reference = by_source.get(source_id, ())
            turn_index = int(support["turn_index"])
            if not 0 <= turn_index < len(source_turns_for_reference):
                raise ValueError(
                    f"support for {reference['question_id']} has an invalid turn index"
                )
            turn = source_turns_for_reference[turn_index]
            if quote not in turn.text:
                raise ValueError(
                    f"support for {reference['question_id']} is absent from its source turn"
                )
            start = turn.text.index(quote)
            rows.append(
                {
                    "turn_id": turn.turn_id,
                    "start_char": start,
                    "end_char": start + len(quote),
                    "quote": quote,
                }
            )
        result[reference["question_id"]] = tuple(rows)
    return result


def _baseline_positions(question_ordinal: int, section_position: dict[str, int]) -> np.ndarray:
    response = read(HOLDOUT / "answers" / f"{question_ordinal:03}.response.json")
    ids = [row["section"]["section_id"] for row in response["hydration"]["sections"]]
    return np.asarray(
        [section_position[section_id] for section_id in ids if section_id in section_position],
        dtype=np.int64,
    )


def _measure(
    *, rankings, sections, section_position, questions, supports, limit: int
) -> tuple[dict, list[dict]]:
    baseline_hits = candidate_hits = union_hits = 0
    baseline_complete = candidate_complete = union_complete = 0
    total = 0
    rows = []
    for question in questions:
        qid = question["question_id"]
        baseline = _baseline_positions(int(question["ordinal"]), section_position)
        selected = rankings[qid][:limit]
        baseline_values = tuple(
            _section_covers(support, sections, baseline) for support in supports[qid]
        )
        selected_values = tuple(
            _section_covers(support, sections, selected) for support in supports[qid]
        )
        union_values = tuple(a or b for a, b in zip(baseline_values, selected_values))
        total += len(union_values)
        baseline_hits += sum(baseline_values)
        candidate_hits += sum(selected_values)
        union_hits += sum(union_values)
        baseline_complete += all(baseline_values)
        candidate_complete += all(selected_values)
        union_complete += all(union_values)
        novel = [
            dict(supports[qid][index], selected_section_ids=[
                sections[int(position)].section_id
                for position in selected
                if _section_covers(supports[qid][index], sections, np.asarray([position]))
            ])
            for index, (old, new) in enumerate(zip(baseline_values, selected_values))
            if new and not old
        ]
        if novel:
            rows.append(
                {
                    "question_id": qid,
                    "question": question["question"],
                    "novel_supports": novel,
                    "newly_complete": all(union_values) and not all(baseline_values),
                }
            )
    return (
        {
            "limit": limit,
            "questions": len(questions),
            "support_total": total,
            "baseline_supports": baseline_hits,
            "candidate_supports": candidate_hits,
            "union_supports": union_hits,
            "novel_supports_over_baseline": union_hits - baseline_hits,
            "baseline_complete_questions": baseline_complete,
            "candidate_complete_questions": candidate_complete,
            "union_complete_questions": union_complete,
            "newly_complete_questions": union_complete - baseline_complete,
        },
        rows,
    )


def _token_observable(
    *, encoder, question: str, summary: str, query_feature: np.ndarray, components: np.ndarray
) -> dict:
    """Decompose rank-16 cosine similarity into exact source-token terms."""
    import torch

    capture = encoder.capture(summary, layer=4)
    attention = capture.attention[0].float()
    values = capture.values[0].float()
    key_groups = gqa_key_group_index(
        torch,
        query_heads=encoder.config.num_attention_heads,
        key_value_heads=encoder.config.num_key_value_heads,
        device=encoder.device,
    )
    expanded_values = values[key_groups]
    token_vectors = torch.einsum("hij,hjd->jhd", attention, expanded_values)
    token_vectors = token_vectors / attention.shape[-2]
    token_vectors = token_vectors.reshape(token_vectors.shape[0], -1).cpu().numpy()
    document_feature = token_vectors.sum(axis=0)
    document_code = document_feature @ components.T
    query_code = query_feature @ components.T
    denominator = float(np.linalg.norm(document_code) * np.linalg.norm(query_code))
    token_scores = (token_vectors @ components.T) @ query_code / denominator
    cosine = float(document_code @ query_code / denominator)

    encoded = encoder.tokenizer(summary, return_offsets_mapping=True)
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(token_ids) != len(token_scores):
        raise RuntimeError("token offsets do not match captured attention positions")
    rows = []
    for position, (token_id, offset, score) in enumerate(
        zip(token_ids, offsets, token_scores)
    ):
        start, end = map(int, offset)
        rows.append(
            {
                "position": position,
                "token_id": int(token_id),
                "token": encoder.tokenizer.convert_ids_to_tokens(int(token_id)),
                "text": summary[start:end],
                "start_char": start,
                "end_char": end,
                "cosine_contribution": float(score),
            }
        )
    rows.sort(key=lambda row: (-abs(row["cosine_contribution"]), row["position"]))
    return {
        "question": question,
        "summary": summary,
        "rank16_cosine": cosine,
        "token_contribution_sum": float(token_scores.sum()),
        "reconstruction_error": abs(float(token_scores.sum()) - cosine),
        "top_tokens_by_absolute_contribution": rows[:16],
    }


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    application = HOLDOUT / "application"
    turns = source_turns(application)
    snapshot = native_spine_store.load(application / "native-spine.sqlite", turns=turns)
    sections = tuple(snapshot.semantic.sections)
    section_ids = tuple(section.section_id for section in sections)
    section_position = {section_id: index for index, section_id in enumerate(section_ids)}
    question_file = read(HOLDOUT / "questions" / "questions.json")
    questions = tuple(question_file["questions"])
    references = read(HOLDOUT / "questions" / "references.json")
    supports = _supports(turns, references)

    development = np.load(
        DEVELOPMENT / "attention-values-fp16.npy", allow_pickle=False
    ).astype(np.float32)
    _mean, all_components, _eigenvalues = _basis(
        development, centered=False, maximum=WIDTH
    )
    components = all_components[:WIDTH]

    runtime = LocalRuntime(output / "runtime", assets=RuntimeAssets.resolve(".cache"))
    try:
        attention_runtime = runtime.attention(output / "attention-cache")
        started = time.perf_counter()
        attention_runtime.prepare()
        prepare_s = time.perf_counter() - started
        import torch

        encoder = attention_runtime.scorer.linker.encoder
        with (
            attention_runtime.gate,
            torch.cuda.stream(attention_runtime.stream),
            torch.inference_mode(),
        ):
            started = time.perf_counter()
            document_features = encoder.encode_attention_values(
                [section.summary for section in sections], layers=(4,), batch_size=8
            )[4].numpy().astype(np.float32)
            attention_runtime.stream.synchronize()
            document_s = time.perf_counter() - started
            started = time.perf_counter()
            query_features = encoder.encode_attention_values(
                [question["question"] for question in questions],
                layers=(4,),
                batch_size=8,
            )[4].numpy().astype(np.float32)
            attention_runtime.stream.synchronize()
            query_s = time.perf_counter() - started

            document_codes = _normalize(document_features @ components.T)
            query_codes = _normalize(query_features @ components.T)
            rankings = {
                question["question_id"]: _ordered(
                    document_codes @ query_codes[row], section_ids
                )
                for row, question in enumerate(questions)
            }
            measurements = {}
            details = {}
            for limit in (PRIMARY_LIMIT, 16):
                measurements[str(limit)], details[str(limit)] = _measure(
                    rankings=rankings,
                    sections=sections,
                    section_position=section_position,
                    questions=questions,
                    supports=supports,
                    limit=limit,
                )

            observable = None
            if details[str(PRIMARY_LIMIT)]:
                gain = details[str(PRIMARY_LIMIT)][0]
                section_id = gain["novel_supports"][0]["selected_section_ids"][0]
                position = section_position[section_id]
                question_row = next(
                    index
                    for index, question in enumerate(questions)
                    if question["question_id"] == gain["question_id"]
                )
                observable = _token_observable(
                    encoder=encoder,
                    question=gain["question"],
                    summary=sections[position].summary,
                    query_feature=query_features[question_row],
                    components=components,
                )
    finally:
        runtime.close()

    np.save(output / "attention-values-fp16.npy", document_features.astype(np.float16))
    np.save(output / "attention-query-values-fp16.npy", query_features.astype(np.float16))
    report = {
        "format": "memory-condense-attention-feature-holdout-v1",
        "development_archive": str(DEVELOPMENT),
        "holdout_archive": str(HOLDOUT),
        "provider_calls": 0,
        "raw_inputs_to_qwen": False,
        "predeclared_policy": {"basis": "uncentered_svd", "rank": WIDTH, "limit": PRIMARY_LIMIT},
        "sections": len(sections),
        "questions": len(questions),
        "measurements": measurements,
        "timing": {
            "qwen_prepare_s": prepare_s,
            "qwen_document_encode_s": document_s,
            "qwen_query_encode_s": query_s,
        },
        "token_observable": observable,
        "limitations": [
            "The held-out questions emphasize explicit user facts, unlike the development archive's dialogue-wide engineering synthesis.",
            "This is exact support reachability before packet budgeting and reader generation.",
            "The token observable is reconstructed for one novel hit and is not persisted for every section.",
        ],
    }
    save(output / "report.json", report)
    save(output / "details.json", details)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
