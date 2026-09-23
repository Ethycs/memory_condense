import copy
import hashlib
from pathlib import Path

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from tools import assay_hot_retrieval_full100 as assay
from tools import evaluate_hot_retrieval_full100 as evaluate


def _probe_row(ordinal: int) -> dict[str, object]:
    retrieval_query = f"question {ordinal}?"
    prompt_question = f"[Question asked at 2026/09/05]\n{retrieval_query}"
    question_id = f"question-{ordinal:03d}"
    retrieval_query_sha256 = quote_sha256(retrieval_query)
    prompt_question_sha256 = quote_sha256(prompt_question)
    locked_probe_identity_sha256 = identity_sha256(
        {
            "format": assay.QUESTION_PROBE_FORMAT,
            "ordinal": ordinal % 10,
            "question_id_sha256": identity_sha256({"question_id": question_id}),
            "retrieval_query_sha256": retrieval_query_sha256,
            "prompt_question_sha256": prompt_question_sha256,
        }
    )
    row: dict[str, object] = {
        "ordinal": ordinal,
        "shard_offset": (ordinal // 10) * 10,
        "local_ordinal": ordinal % 10,
        "question_id": question_id,
        "retrieval_query": retrieval_query,
        "prompt_question": prompt_question,
        "retrieval_query_sha256": retrieval_query_sha256,
        "prompt_question_sha256": prompt_question_sha256,
        "locked_probe_identity_sha256": locked_probe_identity_sha256,
    }
    row["probe_sha256"] = identity_sha256(row)
    return row


def _probe_artifact() -> dict[str, object]:
    rows = [
        _probe_row(ordinal)
        for ordinal in range(assay.EXPECTED_QUESTION_COUNT)
    ]
    return {
        "format": assay.PROBE_FORMAT,
        "status": "sealed_gold_free_locked_full100_probes",
        "population_identity": {
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
            "ordered_question_id_sha256s": [
                identity_sha256({"question_id": row["question_id"]})
                for row in rows
            ],
            "ordered_question_probe_sha256s": [
                row["locked_probe_identity_sha256"] for row in rows
            ],
        },
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "question_count": assay.EXPECTED_QUESTION_COUNT,
        "shard_count": 10,
        "source_bindings": [
            {"shard_offset": offset} for offset in assay.LOCKED_100Q_OFFSETS
        ],
        "questions": rows,
        "retrieval_query_form": assay.hot.RETRIEVAL_QUERY_FORM,
        "gold_fields_present": False,
        "provider_calls": 0,
    }


def _provider_arm(question: str) -> dict[str, object]:
    messages = [{"role": "user", "content": question}]
    payload = assay.hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    return {
        "raw_evidence_only": True,
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "prompt_workspace_token_proxy": 10,
    }


def _selection_artifact(implementation: object) -> dict[str, object]:
    probes = _probe_artifact()["questions"]
    compiled = [f"{ordinal + 1:064x}" for ordinal in range(10)]
    return {
        "format": assay.SELECTION_FORMAT,
        "status": "sealed_gold_blind_locked_full100_frozen_v6_candidate",
        "implementation": implementation,
        "bindings": {
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
            "probes_sha256": "a" * 64,
            "compiled_catalog_sha256": "b" * 64,
            "ordered_compiled_shard_sha256s": compiled,
        },
        "controls": {
            "policy_id": assay.POLICY_ID,
            "max_context_token_proxy": assay.MAX_CONTEXT_TOKENS,
            "max_prompt_workspace_token_proxy": assay.MAX_PROMPT_TOKENS,
        },
        "questions": [
            {
                **{
                    key: probes[ordinal][key]
                    for key in (
                        "ordinal",
                        "shard_offset",
                        "local_ordinal",
                        "question_id",
                        "probe_sha256",
                        "retrieval_query_sha256",
                        "prompt_question_sha256",
                    )
                },
                "arms": {
                    "a3_protected_union": _provider_arm(
                        f"question {ordinal}?"
                    )
                },
            }
            for ordinal in range(assay.EXPECTED_QUESTION_COUNT)
        ],
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }


def _answer_artifact(selection_sha256: str) -> dict[str, object]:
    rows = []
    for ordinal in range(assay.EXPECTED_QUESTION_COUNT):
        prediction = f"prediction {ordinal}"
        rows.append(
            {
                "ordinal": ordinal,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
            }
        )
    return {
        "format": evaluate.ANSWER_FORMAT,
        "status": "sealed_terra_predictions_without_gold",
        "selection_sha256": selection_sha256,
        "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        "questions": rows,
        "gold_fields_present": False,
        "retries": 0,
    }


def test_gold_free_row_guard_rejects_casefolded_label_fields() -> None:
    assay._assert_gold_free_rows(  # noqa: SLF001
        [{"question_id": "q", "retrieval_query": "where?"}]
    )

    for forbidden in ("answer", "Answer", "reference", "Evidence_Sources"):
        with pytest.raises(ValueError, match="gold fields"):
            assay._assert_gold_free_rows(  # noqa: SLF001
                [{"question_id": "q", forbidden: "leak"}]
            )


def test_probe_loader_accepts_only_the_global_locked_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _probe_artifact()
    monkeypatch.setattr(
        assay.hot,
        "_read_json_artifact",
        lambda _path: (artifact, "a" * 64),
    )
    monkeypatch.setattr(
        assay,
        "validate_locked_cumulative_population_identity",
        lambda value, **_kwargs: value,
    )

    loaded, digest = assay._load_probes(tmp_path)  # noqa: SLF001

    assert digest == "a" * 64
    assert [row["ordinal"] for row in loaded["questions"]] == list(range(100))

    artifact["questions"][0], artifact["questions"][1] = (  # type: ignore[index]
        artifact["questions"][1],  # type: ignore[index]
        artifact["questions"][0],  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="probe order changed"):
        assay._load_probes(tmp_path)  # noqa: SLF001


def test_selection_loader_checks_order_and_provider_packet_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = {"sha256": "implementation"}
    artifact = _selection_artifact(implementation)
    probes = _probe_artifact()
    catalog = {
        "shards": [
            {
                "shard_offset": offset,
                "compiled_sha256": f"{ordinal + 1:064x}",
            }
            for ordinal, offset in enumerate(assay.LOCKED_100Q_OFFSETS)
        ]
    }
    monkeypatch.setattr(assay, "_implementation_identity", lambda: implementation)
    monkeypatch.setattr(assay, "_load_probes", lambda _path: (probes, "a" * 64))
    monkeypatch.setattr(assay, "_load_catalog", lambda _path: (catalog, "b" * 64))
    monkeypatch.setattr(assay.hot, "_validate_arm_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        assay.hot,
        "_read_json_artifact",
        lambda _path: (artifact, "b" * 64),
    )

    loaded, digest = assay._load_selection(tmp_path)  # noqa: SLF001

    assert digest == "b" * 64
    assert len(loaded["questions"]) == assay.EXPECTED_QUESTION_COUNT

    broken_order = copy.deepcopy(artifact)
    broken_order["questions"][0]["ordinal"] = 1  # type: ignore[index]
    monkeypatch.setattr(
        assay.hot,
        "_read_json_artifact",
        lambda _path: (broken_order, "b" * 64),
    )
    with pytest.raises(ValueError, match="selection changed"):
        assay._load_selection(tmp_path)  # noqa: SLF001

    broken_packet = copy.deepcopy(artifact)
    broken_packet["questions"][0]["arms"]["a3_protected_union"][  # type: ignore[index]
        "provider_messages"
    ][0]["content"] = "substituted prompt"
    monkeypatch.setattr(
        assay.hot,
        "_read_json_artifact",
        lambda _path: (broken_packet, "b" * 64),
    )
    with pytest.raises(ValueError, match="provider packet binding changed"):
        assay._load_selection(tmp_path)  # noqa: SLF001


def test_temporal_event_scan_uses_the_dynamic_namespace_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}

    class LexicalStub:
        def search(self, query: str, limit: int = 100):
            observed["limit"] = limit
            return [("assistant", 3.0), ("future", 2.0), ("visited", 1.0)]

    plan = assay.hot.plan_temporal_enumeration(
        "What is the order of the museums I visited from earliest to latest?"
    )
    metadata = {
        "assistant": {
            "event_first_person": False,
            "event_fixed_completed": False,
            "event_ed_verbs": [],
        },
        "future": {
            "event_first_person": True,
            "event_fixed_completed": False,
            "event_ed_verbs": ["planned"],
        },
        "visited": {
            "event_first_person": True,
            "event_fixed_completed": True,
            "event_ed_verbs": ["visited"],
        },
    }
    monkeypatch.setattr(assay.hot, "EXPECTED_CHUNKS", 999_999)

    hits, elapsed = assay.hot._timed_temporal_events(  # noqa: SLF001
        LexicalStub(),  # type: ignore[arg-type]
        plan,
        metadata,
    )

    assert observed["limit"] == len(metadata)
    assert [hit.chunk_id for hit in hits] == ["visited"]
    assert elapsed >= 0


def test_source_binding_rejects_retrieval_receipt_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    shard_root = source_root / "shards" / "offset-000"
    store = shard_root / "source-current" / "stores" / "store-key" / "store"
    store.mkdir(parents=True)
    (store / "memory.db").write_bytes(b"db")
    (store / "hnsw_index.bin").write_bytes(b"index")
    selection = {
        "selected_store_entry": "stores/store-key",
        "embedding_identity": {
            "model_id": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
            "dimension": DEFAULT_MODEL_DIM,
        },
        "chunk_count": 1,
        "turn_count": 1,
        "receipt_sha256": "1" * 64,
        "database_sha256": "2" * 64,
        "index_sha256": "3" * 64,
    }
    retrieval = {
        "gold_fields_present": False,
        "provider_calls": 0,
        "shard_offset": 0,
        "source_store_receipt": selection,
        "source_store_receipt_sha256": selection["receipt_sha256"],
        "shard_identity": {"shard_identity_sha256": "9" * 64},
        "shard_identity_sha256": "9" * 64,
    }

    def read_artifact(path: Path):
        if path.name == "source-current-selection.json":
            return selection, "4" * 64
        if path.name == "retrieval.json":
            return retrieval, "5" * 64
        raise AssertionError(path)

    monkeypatch.setattr(assay.hot, "_read_json_artifact", read_artifact)
    monkeypatch.setattr(
        assay,
        "validate_locked_cumulative_shard_identity",
        lambda value: value,
    )
    binding = assay._load_source_binding(  # noqa: SLF001
        source_root,
        0,
        expected_shard_identity=None,
        verify_large_files=False,
    )
    assert binding.receipt_sha256 == selection["receipt_sha256"]

    retrieval["source_store_receipt_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="sealed retrieval store"):
        assay._load_source_binding(  # noqa: SLF001
            source_root,
            0,
            expected_shard_identity=None,
            verify_large_files=False,
        )


def test_answer_loader_rejects_prediction_substitution_before_journal_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_sha = "7" * 64
    artifact = _answer_artifact(selection_sha)
    monkeypatch.setattr(
        evaluate.assay.hot,
        "_read_json_artifact",
        lambda _path: (artifact, "8" * 64),
    )

    artifact["questions"][42]["prediction"] = "substituted"  # type: ignore[index]
    with pytest.raises(ValueError, match="prediction changed"):
        evaluate._load_answers(tmp_path, selection_sha)  # noqa: SLF001
