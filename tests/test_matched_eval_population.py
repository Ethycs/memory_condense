from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    identity_sha256 as legacy_identity_sha256,
    quote_sha256,
)
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, canonical_json_bytes
from tools.matched_eval.population import (
    DEFAULT_MAX_PROMPT_TOKENS,
    EXPECTED_QUESTION_COUNT,
    MERGED_QUESTION_FORMAT,
    MERGED_RETRIEVAL_FORMAT,
    SOURCE_STAGE_ID,
    STAGE_IDS,
    MatchedPopulationError,
    load_s0_population,
)
from tools.matched_eval.renderer import RENDERER_ID


REAL_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
REAL_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _seal_receipt(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": legacy_identity_sha256(body)}


def _question(ordinal: int, population_sha: str) -> dict[str, Any]:
    question_id = f"q-{ordinal}"
    raw_question = f"What was choice {ordinal}?"
    dated = f"[Question asked at 2026/08/{ordinal + 1:02d}]\n{raw_question}"
    source_id = f"turn-{ordinal}"
    chunk_id = f"chunk-{ordinal}"
    text = f"Choice {ordinal} was blue."
    excerpt = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "text_sha256": quote_sha256(text),
    }
    evidence_id = legacy_identity_sha256({"kind": "protected_excerpt", **excerpt})
    evidence = [{"evidence_id": evidence_id, "source_id": source_id, "text": text}]
    context = f"[1] {text}"
    messages = [
        {"role": "system", "content": "Answer only from supplied memory."},
        {
            "role": "user",
            "content": (
                f"Retrieved excerpts:\n{context}\n\nQuestion: {dated}\n"
                "Short answer:"
            ),
        },
    ]
    prompt_sha = legacy_identity_sha256(messages)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    excerpt_projection = legacy_identity_sha256([excerpt])
    s0_projection = legacy_identity_sha256(
        {"protected_excerpts": [excerpt], "admitted_atoms": []}
    )
    predecessor = _seal_receipt(
        {
            "format": "memory-condense-causal-coverage-predecessor-v1",
            "protected_chunk_ids": [chunk_id],
            "packed_chunk_ids": [chunk_id],
            "protected_excerpt_projection_sha256": excerpt_projection,
            "protected_context_sha256": quote_sha256(context),
            "prompt_messages_sha256": prompt_sha,
            "prompt_token_proxy": prompt_tokens,
            "retrieval_query_sha256": legacy_identity_sha256({"query": dated}),
            "prompt_question_sha256": legacy_identity_sha256(
                {"prompt_question": dated}
            ),
            "retained_request_token_state_bytes": 0,
        }
    )

    stage_rows: list[dict[str, Any]] = []
    parent_receipt: str | None = None
    for index, stage_id in enumerate(STAGE_IDS):
        receipt = _seal_receipt(
            {
                "format": "memory-condense-cumulative-retrieval-stage-v2",
                "stage_id": stage_id,
                "matched_controls_sha256": SHA_A,
                "method_evidence_sha256": (
                    predecessor["receipt_sha256"]
                    if index == 0
                    else (SHA_B, SHA_C, SHA_D)[index - 1]
                ),
                "parent_stage_receipt_sha256": parent_receipt,
                "parent_evidence_ids": [] if index == 0 else [evidence_id],
                "selected_evidence_ids": [evidence_id],
                "added_evidence_ids": [evidence_id] if index == 0 else [],
                "admission_status": "root" if index == 0 else "no_novel_evidence",
                "evidence_projection_sha256": s0_projection,
                "context_sha256": quote_sha256(context),
                "prompt_messages_sha256": prompt_sha,
                "context_token_proxy": count_tokens(context),
                "max_context_token_proxy": 7_000,
                "prompt_token_proxy": prompt_tokens,
                "max_prompt_token_proxy": DEFAULT_MAX_PROMPT_TOKENS,
                "responder_output_token_reserve": 256,
            }
        )
        stage_rows.append(
            {
                "stage_id": stage_id,
                "stage_receipt": receipt,
                "provider_messages": copy.deepcopy(messages),
                "evidence": copy.deepcopy(evidence),
            }
        )
        parent_receipt = receipt["receipt_sha256"]

    ladder_sha = legacy_identity_sha256(
        {
            "stages": [row["stage_receipt"] for row in stage_rows],
            "format": "memory-condense-cumulative-retrieval-ladder-v1",
        }
    )
    retrieval_receipt = _seal_receipt(
        {
            "format": "memory-condense-recall-guarded-cumulative-v2",
            "predecessor_receipt_sha256": predecessor["receipt_sha256"],
            "ladder_receipt_sha256": ladder_sha,
            "protected_chunk_ids": [chunk_id],
            "protected_evidence_ids": [evidence_id],
            "protected_excerpt_projection_sha256": excerpt_projection,
            "final_evidence_ids": [evidence_id],
            "prompt_messages_sha256": prompt_sha,
            "retained_request_token_state_bytes": 0,
        }
    )
    return {
        "format": MERGED_QUESTION_FORMAT,
        "population_identity_sha256": population_sha,
        "ordinal": ordinal,
        "local_ordinal": ordinal,
        "question_id": question_id,
        "question_id_sha256": legacy_identity_sha256({"question_id": question_id}),
        "question_sha256": quote_sha256(raw_question),
        "dated_question_sha256": quote_sha256(dated),
        "probe_identity_sha256": legacy_identity_sha256(
            {
                "format": "memory-condense-gold-blind-question-probe-v1",
                "ordinal": ordinal,
                "question_id_sha256": legacy_identity_sha256(
                    {"question_id": question_id}
                ),
                "retrieval_query_sha256": quote_sha256(raw_question),
                "prompt_question_sha256": quote_sha256(dated),
            }
        ),
        "stage_ids": list(STAGE_IDS),
        "stages": stage_rows,
        "predecessor_receipt": predecessor,
        "retrieval_receipt": retrieval_receipt,
        "provider_calls": 0,
    }


def _retrieval(count: int = 2) -> dict[str, Any]:
    question_meta = [
        {
            "question_id": f"q-{ordinal}",
            "question_sha256": quote_sha256(f"What was choice {ordinal}?"),
            "dated_question_sha256": quote_sha256(
                f"[Question asked at 2026/08/{ordinal + 1:02d}]\n"
                f"What was choice {ordinal}?"
            ),
        }
        for ordinal in range(count)
    ]
    population_body = {
        "format": "memory-condense-locked-cumulative-1m-100q-population-v1",
        "gold_fields_present": False,
        "question_count": count,
        "ordered_question_id_sha256s": [
            legacy_identity_sha256({"question_id": row["question_id"]})
            for row in question_meta
        ],
        "ordered_question_probe_sha256s": [
            legacy_identity_sha256(
                {
                    "format": "memory-condense-gold-blind-question-probe-v1",
                    "ordinal": ordinal,
                    "question_id_sha256": legacy_identity_sha256(
                        {"question_id": row["question_id"]}
                    ),
                    "retrieval_query_sha256": row["question_sha256"],
                    "prompt_question_sha256": row["dated_question_sha256"],
                }
            )
            for ordinal, row in enumerate(question_meta)
        ],
    }
    population_sha = legacy_identity_sha256(population_body)
    population = {**population_body, "population_identity_sha256": population_sha}
    questions = [_question(ordinal, population_sha) for ordinal in range(count)]
    return {
        "format": MERGED_RETRIEVAL_FORMAT,
        "gold_fields_present": False,
        "provider_calls": 0,
        "question_count": count,
        "stage_ids": list(STAGE_IDS),
        "population_identity": population,
        "population_identity_sha256": population_sha,
        "question_part_sha256s": [
            hashlib.sha256(canonical_json_bytes(row)).hexdigest()
            for row in questions
        ],
        "questions": questions,
    }


def _publish(tmp_path: Path, value: dict[str, Any], name: str = "retrieval.json") -> Path:
    path = tmp_path / name
    publish_sealed_json(path, value)
    return path


def test_loads_sealed_s0_packets_snapshot_and_zero_call_preflight(
    tmp_path: Path,
) -> None:
    population = load_s0_population(
        _publish(tmp_path, _retrieval()), expected_question_count=2
    )

    assert population.question_count == 2
    assert population.snapshot.renderer_id == RENDERER_ID
    assert population.snapshot.source_artifacts[0].sha256 == population.retrieval_sha256
    assert [row.packet.question_id for row in population.rows] == ["q-0", "q-1"]
    assert population.rows[0].packet.stage_id == SOURCE_STAGE_ID
    assert population.rows[0].packet.protected_evidence[0].text == "Choice 0 was blue."
    assert population.rows[0].rendered_prompt.messages[-1]["role"] == "user"
    preflight = population.preflight_projection()
    assert preflight["logical_prompt_count"] == 2
    assert preflight["unique_prompt_count"] == 2
    assert preflight["required_authorized_provider_calls"] == 2
    assert preflight["provider_calls"] == 0
    assert preflight["gold_loaded"] is False
    assert len(population.preflight_sha256) == 64


def test_rejects_resealed_s0_text_when_evidence_projection_is_stale(
    tmp_path: Path,
) -> None:
    retrieval = _retrieval(1)
    retrieval["questions"][0]["stages"][0]["evidence"][0]["text"] = "Tampered."
    retrieval["question_part_sha256s"][0] = hashlib.sha256(
        canonical_json_bytes(retrieval["questions"][0])
    ).hexdigest()

    with pytest.raises(MatchedPopulationError, match="projection"):
        load_s0_population(
            _publish(tmp_path, retrieval), expected_question_count=1
        )


def test_rejects_resealed_question_reordering(tmp_path: Path) -> None:
    retrieval = _retrieval()
    retrieval["questions"].reverse()
    retrieval["question_part_sha256s"] = [
        hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        for row in retrieval["questions"]
    ]

    with pytest.raises(MatchedPopulationError, match="binding|order"):
        load_s0_population(
            _publish(tmp_path, retrieval), expected_question_count=2
        )


def test_gold_firewall_runs_on_sealed_input(tmp_path: Path) -> None:
    retrieval = _retrieval(1)
    retrieval["questions"][0]["reference_answer"] = "blue"
    retrieval["question_part_sha256s"][0] = hashlib.sha256(
        canonical_json_bytes(retrieval["questions"][0])
    ).hexdigest()

    with pytest.raises(MatchedEvalContractError, match="reference_answer"):
        load_s0_population(
            _publish(tmp_path, retrieval), expected_question_count=1
        )


@pytest.mark.skipif(not REAL_RETRIEVAL.is_file(), reason="sealed retrieval absent")
def test_real_sealed_retrieval_projects_100_v2_s0_prompts() -> None:
    population = load_s0_population(
        REAL_RETRIEVAL,
        expected_retrieval_sha256=REAL_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )

    assert population.question_count == EXPECTED_QUESTION_COUNT
    assert population.prompt_population.logical_prompt_count == 100
    assert population.prompt_population.unique_prompt_count == 100
    assert population.preflight_projection()["provider_calls"] == 0
    assert all(row.packet.protected_evidence for row in population.rows)
