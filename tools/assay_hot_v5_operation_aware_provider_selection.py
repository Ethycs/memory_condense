#!/usr/bin/env python3
"""Publish the sealed locked100 packets with one universal answer policy.

This adapter authenticates only the v4 provider ``selection.json``.  It leaves
the dated question and retrieved evidence byte-exact, replaces only the system
message, and recomputes the prompt and provider-payload receipts.  It performs
no benchmark loading and no provider I/O.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import quote_sha256
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_v3_typed_operator_full100 as typed
from tools import assay_hot_v4_user_envelope_provider_selection as source_assay
from tools.matched_eval.contracts import (
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.hot_v5_user_spine_prompt import (
    OPERATION_AWARE_SYSTEM_PROMPT,
    replace_system_prompt_only,
)


FORMAT = "memory-condense-hot-v5-operation-aware-provider-selection-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
SELECTION_NAME = "selection.json"
EXPECTED_QUESTION_COUNT = 100
EXPECTED_POPULATION_SHA256 = (
    "9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246"
)
EXPECTED_SOURCE_SELECTION_SHA256 = (
    "7a900be230d6bebf4cf882988ef4f548efb3baedf0264fe66185325982e25150"
)

DEFAULT_SOURCE_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v4-user-envelope-provider-full100-20260907-r1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v5-operation-aware-provider-full100-20260907-r1"
)

_SELECTION_KEYS = frozenset(
    {
        "format",
        "gold_fields_present",
        "population_identity_sha256",
        "provider_calls",
        "question_count",
        "questions",
        "source_selection_sha256",
        "status",
    }
)
_ROW_KEYS = frozenset(
    {
        "arms",
        "format",
        "local_ordinal",
        "ordinal",
        "prompt_question_sha256",
        "question_id",
        "row_receipt_sha256",
        "shard_offset",
        "source_row_receipt_sha256",
    }
)
_ARM_KEYS = frozenset(
    {
        "context_token_proxy",
        "packed_chunk_ids",
        "packed_evidence_sha256",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_messages",
        "provider_payload_sha256",
        "provider_payload_utf8_bytes",
        "raw_evidence_only",
    }
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _load_source_selection(source_root: Path) -> tuple[dict[str, Any], str]:
    """Authenticate the single allowed source artifact through its owner."""

    selection, digest = source_assay._load_selection(source_root)  # noqa: SLF001
    _require(
        digest == EXPECTED_SOURCE_SELECTION_SHA256,
        "sealed v4 provider selection digest changed",
    )
    _require(
        selection.get("population_identity_sha256")
        == EXPECTED_POPULATION_SHA256
        and selection.get("question_count") == EXPECTED_QUESTION_COUNT
        and selection.get("provider_calls") == 0
        and selection.get("gold_fields_present") is False,
        "sealed v4 provider selection contract changed",
    )
    assert_gold_blind(selection, path="operation_aware_source_selection")
    return selection, digest


def _operation_aware_arm(source_value: object, *, ordinal: int) -> dict[str, Any]:
    _require(
        type(source_value) is dict and set(source_value) == _ARM_KEYS,
        f"source provider arm schema changed at {ordinal}",
    )
    source = source_value
    source_messages = source["provider_messages"]
    _require(
        type(source_messages) is list and len(source_messages) == 2,
        f"source provider messages changed at {ordinal}",
    )
    messages = replace_system_prompt_only(source_messages)
    _require(
        messages[1] == source_messages[1]
        and messages[1]["content"].encode("utf-8")
        == source_messages[1]["content"].encode("utf-8"),
        f"operation-aware transform changed the user message at {ordinal}",
    )
    prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
    workspace_tokens = prompt_tokens + hot.RESPONDER_OUTPUT_TOKEN_RESERVE
    _require(
        workspace_tokens <= 8_000,
        f"operation-aware prompt exceeds workspace at {ordinal}",
    )
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    arm = {
        "context_token_proxy": source["context_token_proxy"],
        "packed_chunk_ids": copy.deepcopy(source["packed_chunk_ids"]),
        "packed_evidence_sha256": source["packed_evidence_sha256"],
        "prompt_token_proxy": prompt_tokens,
        "prompt_workspace_token_proxy": workspace_tokens,
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "raw_evidence_only": source["raw_evidence_only"],
    }
    for field in (
        "context_token_proxy",
        "packed_chunk_ids",
        "packed_evidence_sha256",
        "raw_evidence_only",
    ):
        _require(
            arm[field] == source[field],
            f"operation-aware transform changed {field} at {ordinal}",
        )
    return arm


def _validate_arm(
    arm_value: object,
    *,
    ordinal: int,
    prompt_question_sha256: object,
) -> None:
    _require(
        type(arm_value) is dict and set(arm_value) == _ARM_KEYS,
        f"operation-aware arm schema changed at {ordinal}",
    )
    arm = arm_value
    require_sha256(
        prompt_question_sha256,  # type: ignore[arg-type]
        f"questions[{ordinal}].prompt_question_sha256",
    )
    require_sha256(
        arm["packed_evidence_sha256"],  # type: ignore[arg-type]
        f"questions[{ordinal}].packed_evidence_sha256",
    )
    require_sha256(
        arm["provider_payload_sha256"],  # type: ignore[arg-type]
        f"questions[{ordinal}].provider_payload_sha256",
    )
    chunk_ids = arm["packed_chunk_ids"]
    _require(
        type(chunk_ids) is list
        and all(
            type(chunk_id) is str
            and bool(chunk_id)
            and chunk_id.strip() == chunk_id
            for chunk_id in chunk_ids
        )
        and len(chunk_ids) == len(set(chunk_ids)),
        f"operation-aware packed chunk IDs changed at {ordinal}",
    )
    for field in (
        "context_token_proxy",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_payload_utf8_bytes",
    ):
        _require(
            type(arm[field]) is int and arm[field] >= 0,
            f"operation-aware {field} changed at {ordinal}",
        )
    messages = arm["provider_messages"]
    _require(
        type(messages) is list
        and len(messages) == 2
        and all(type(message) is dict for message in messages)
        and all(set(message) == {"role", "content"} for message in messages)
        and messages[0]
        == {"role": "system", "content": OPERATION_AWARE_SYSTEM_PROMPT},
        f"operation-aware prompt envelope changed at {ordinal}",
    )
    dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
    _require(
        quote_sha256(dated_question) == prompt_question_sha256,
        f"operation-aware question binding changed at {ordinal}",
    )
    prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
    _require(
        arm["prompt_token_proxy"] == prompt_tokens
        and arm["prompt_workspace_token_proxy"]
        == prompt_tokens + hot.RESPONDER_OUTPUT_TOKEN_RESERVE
        and arm["context_token_proxy"] <= 7_000
        and arm["prompt_workspace_token_proxy"] <= 8_000
        and arm["raw_evidence_only"] is True,
        f"operation-aware token boundary changed at {ordinal}",
    )
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    _require(
        arm["provider_payload_sha256"] == hashlib.sha256(payload).hexdigest()
        and arm["provider_payload_utf8_bytes"] == len(payload),
        f"operation-aware provider payload changed at {ordinal}",
    )


def _project_selection(
    source: Mapping[str, Any], *, source_selection_sha256: str
) -> dict[str, Any]:
    source_rows = source.get("questions")
    _require(
        type(source_rows) is list and len(source_rows) == EXPECTED_QUESTION_COUNT,
        "operation-aware source population changed",
    )
    rows: list[dict[str, Any]] = []
    for ordinal, source_row in enumerate(source_rows):
        _require(
            type(source_row) is dict
            and source_row.get("ordinal") == ordinal
            and source_row.get("local_ordinal") == ordinal % 10
            and source_row.get("shard_offset") == ordinal - (ordinal % 10)
            and type(source_row.get("question_id")) is str,
            f"operation-aware source row changed at {ordinal}",
        )
        source_unsigned = dict(source_row)
        source_receipt = source_unsigned.pop("row_receipt_sha256", None)
        _require(
            source_receipt == identity_sha256(source_unsigned),
            f"operation-aware source row receipt changed at {ordinal}",
        )
        source_arms = source_row.get("arms")
        _require(
            type(source_arms) is dict and set(source_arms) == {"a3_protected_union"},
            f"operation-aware source arms changed at {ordinal}",
        )
        arm = _operation_aware_arm(
            source_arms["a3_protected_union"], ordinal=ordinal
        )
        _validate_arm(
            arm,
            ordinal=ordinal,
            prompt_question_sha256=source_row.get("prompt_question_sha256"),
        )
        body = {
            "arms": {"a3_protected_union": arm},
            "format": ROW_FORMAT,
            "local_ordinal": ordinal % 10,
            "ordinal": ordinal,
            "prompt_question_sha256": source_row["prompt_question_sha256"],
            "question_id": source_row["question_id"],
            "shard_offset": ordinal - (ordinal % 10),
            "source_row_receipt_sha256": source_receipt,
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    selection = {
        "format": FORMAT,
        "gold_fields_present": False,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": len(rows),
        "questions": rows,
        "source_selection_sha256": source_selection_sha256,
        "status": "sealed_gold_free_operation_aware_provider_packets",
    }
    assert_gold_blind(selection, path="operation_aware_provider_selection")
    return selection


def materialize(*, source_root: Path, output_root: Path) -> str:
    _require(not output_root.exists(), "provider output root must be unique and absent")
    source, source_sha = _load_source_selection(source_root)
    selection = _project_selection(
        source, source_selection_sha256=source_sha
    )
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / SELECTION_NAME, selection
    )
    print(
        f"Operation-aware provider selection: {len(selection['questions'])} prompts; "
        f"selection={digest}",
        flush=True,
    )
    return digest


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    selection, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    rows = selection.get("questions")
    _require(
        set(selection) == _SELECTION_KEYS
        and selection.get("format") == FORMAT
        and selection.get("status")
        == "sealed_gold_free_operation_aware_provider_packets"
        and selection.get("gold_fields_present") is False
        and selection.get("provider_calls") == 0
        and selection.get("population_identity_sha256")
        == EXPECTED_POPULATION_SHA256
        and selection.get("source_selection_sha256")
        == EXPECTED_SOURCE_SELECTION_SHA256
        and selection.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "sealed operation-aware provider selection changed",
    )
    require_sha256(
        selection["population_identity_sha256"], "population_identity_sha256"
    )
    require_sha256(selection["source_selection_sha256"], "source_selection_sha256")
    for ordinal, row in enumerate(rows):
        _require(
            type(row) is dict and set(row) == _ROW_KEYS,
            f"operation-aware row schema changed at {ordinal}",
        )
        unsigned = dict(row)
        receipt = unsigned.pop("row_receipt_sha256", None)
        require_sha256(receipt, f"questions[{ordinal}].row_receipt_sha256")
        require_sha256(
            row["source_row_receipt_sha256"],
            f"questions[{ordinal}].source_row_receipt_sha256",
        )
        require_sha256(
            row["prompt_question_sha256"],
            f"questions[{ordinal}].prompt_question_sha256",
        )
        require_text(row["question_id"], f"questions[{ordinal}].question_id")
        arms = row["arms"]
        _require(
            receipt == identity_sha256(unsigned)
            and row.get("format") == ROW_FORMAT
            and row.get("ordinal") == ordinal
            and row.get("local_ordinal") == ordinal % 10
            and row.get("shard_offset") == ordinal - (ordinal % 10)
            and type(arms) is dict
            and set(arms) == {"a3_protected_union"},
            f"operation-aware row changed at {ordinal}",
        )
        _validate_arm(
            arms["a3_protected_union"],
            ordinal=ordinal,
            prompt_question_sha256=row["prompt_question_sha256"],
        )
    assert_gold_blind(selection, path="loaded_operation_aware_provider_selection")
    return selection, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_subparsers(dest="command", required=True).add_parser("materialize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        materialize(
            source_root=args.source_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT_ROOT", "FORMAT", "_load_selection", "materialize"]
