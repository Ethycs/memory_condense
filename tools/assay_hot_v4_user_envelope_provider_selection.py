#!/usr/bin/env python3
"""Publish a gold-free provider seam from the sealed user-envelope shadow.

The source root may also contain post-hoc evaluation output, so this adapter
opens exactly ``construction.json``, ``runtime.json``, and ``replay.json``.
It emits a compact, self-contained ``selection.json`` into a fresh root.  The
answer process can therefore consume the 100 sealed prompt packets without
filesystem access to benchmark references or the post-hoc score artifact.
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
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_v3_typed_operator_full100 as typed
from tools.matched_eval.contracts import (
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-hot-v4-user-envelope-provider-selection-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
SELECTION_NAME = "selection.json"
EXPECTED_QUESTION_COUNT = 100
EXPECTED_POPULATION_SHA256 = (
    "9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246"
)
EXPECTED_SOURCE_CONSTRUCTION_SHA256 = (
    "0ba317cefd6860623352078b58804285eaf02f46bc4524dd71e86585df81dbde"
)
EXPECTED_SOURCE_RUNTIME_SHA256 = (
    "6f7c76957bd4b5161d756db487e2734dfe07cb20731ba47c124ca095bcd351ce"
)
EXPECTED_SOURCE_REPLAY_SHA256 = (
    "11974e81cafbe4b5868d9d1eb79a9030c0cc969f0993bae575dae2c762e033bf"
)
SOURCE_CONSTRUCTION_FORMAT = (
    "memory-condense-hot-v4-user-envelope-shadow-full100-construction-v1"
)
SOURCE_RUNTIME_FORMAT = (
    "memory-condense-hot-v4-user-envelope-shadow-full100-runtime-v1"
)
SOURCE_REPLAY_FORMAT = (
    "memory-condense-hot-v4-user-envelope-shadow-full100-replay-v1"
)

_SELECTION_KEYS = frozenset(
    {
        "format",
        "gold_fields_present",
        "population_identity_sha256",
        "provider_calls",
        "question_count",
        "questions",
        "source_construction_sha256",
        "source_replay_sha256",
        "source_runtime_sha256",
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

DEFAULT_SOURCE_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v4-user-envelope-shadow-full100-20260907-r2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v4-user-envelope-provider-full100-20260907-r1"
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _load_source(source_root: Path) -> tuple[dict[str, Any], str, str, str]:
    """Authenticate the three gold-free source artifacts and nothing else."""

    construction, construction_sha = hot._read_json_artifact(  # noqa: SLF001
        source_root / "construction.json"
    )
    runtime, runtime_sha = hot._read_json_artifact(  # noqa: SLF001
        source_root / "runtime.json"
    )
    replay, replay_sha = hot._read_json_artifact(  # noqa: SLF001
        source_root / "replay.json"
    )
    _require(
        construction_sha == EXPECTED_SOURCE_CONSTRUCTION_SHA256
        and runtime_sha == EXPECTED_SOURCE_RUNTIME_SHA256
        and replay_sha == EXPECTED_SOURCE_REPLAY_SHA256,
        "sealed user-envelope source digest changed",
    )
    _require(
        construction.get("format") == SOURCE_CONSTRUCTION_FORMAT
        and runtime.get("format") == SOURCE_RUNTIME_FORMAT
        and replay.get("format") == SOURCE_REPLAY_FORMAT,
        "sealed user-envelope source format changed",
    )
    _require(
        runtime.get("construction_sha256") == construction_sha
        and replay.get("construction_sha256") == construction_sha
        and replay.get("runtime_sha256") == runtime_sha
        and replay.get("byte_identical") is True
        and construction.get("population_identity_sha256")
        == EXPECTED_POPULATION_SHA256
        and construction.get("question_count") == EXPECTED_QUESTION_COUNT
        and replay.get("question_count") == EXPECTED_QUESTION_COUNT,
        "sealed user-envelope lifecycle binding changed",
    )
    for artifact, label in (
        (construction, "construction"),
        (runtime, "runtime"),
        (replay, "replay"),
    ):
        _require(
            artifact.get("gold_loaded") is False
            and artifact.get("model_calls") == 0
            and artifact.get("new_provider_calls") == 0,
            f"{label} crossed the gold/provider firebreak",
        )
        assert_gold_blind(artifact, path=f"provider_source_{label}")
    return construction, construction_sha, runtime_sha, replay_sha


def _provider_arm(arm_value: object) -> dict[str, Any]:
    _require(type(arm_value) is dict, "effective provider arm changed type")
    arm = copy.deepcopy(arm_value)
    dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
    hot._validate_arm_payload(  # noqa: SLF001
        arm,
        prompt_question=dated_question,
        max_context_tokens=7_000,
        max_prompt_tokens=8_000,
    )
    messages = arm.get("provider_messages")
    _require(type(messages) is list and bool(messages), "provider messages are missing")
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    _require(
        arm.get("provider_payload_sha256") == hashlib.sha256(payload).hexdigest()
        and arm.get("provider_payload_utf8_bytes") == len(payload)
        and arm.get("raw_evidence_only") is True,
        "provider payload binding changed",
    )
    return {
        "context_token_proxy": int(arm["context_token_proxy"]),
        "packed_chunk_ids": list(arm["packed_chunk_ids"]),
        "packed_evidence_sha256": identity_sha256(arm["packed_evidence"]),
        "prompt_token_proxy": int(arm["prompt_token_proxy"]),
        "prompt_workspace_token_proxy": int(arm["prompt_workspace_token_proxy"]),
        "provider_messages": messages,
        "provider_payload_sha256": str(arm["provider_payload_sha256"]),
        "provider_payload_utf8_bytes": int(arm["provider_payload_utf8_bytes"]),
        "raw_evidence_only": True,
    }


def _validate_compact_provider_arm(
    arm_value: object,
    *,
    prompt_question_sha256: object,
    ordinal: int,
) -> None:
    """Validate every field retained at the compact provider boundary."""

    _require(
        type(arm_value) is dict and set(arm_value) == _ARM_KEYS,
        f"provider arm schema changed at {ordinal}",
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

    packed_chunk_ids = arm["packed_chunk_ids"]
    _require(
        type(packed_chunk_ids) is list
        and all(
            type(chunk_id) is str and bool(chunk_id) and chunk_id.strip() == chunk_id
            for chunk_id in packed_chunk_ids
        )
        and len(set(packed_chunk_ids)) == len(packed_chunk_ids),
        f"provider packed chunk IDs changed at {ordinal}",
    )
    for field in (
        "context_token_proxy",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_payload_utf8_bytes",
    ):
        _require(
            type(arm[field]) is int and arm[field] >= 0,
            f"provider {field} changed at {ordinal}",
        )

    messages = arm["provider_messages"]
    _require(
        type(messages) is list
        and len(messages) == 2
        and all(type(message) is dict for message in messages)
        and all(set(message) == {"role", "content"} for message in messages)
        and messages[0] == {"role": "system", "content": QA_SYSTEM_PROMPT},
        f"provider prompt envelope changed at {ordinal}",
    )
    dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
    _require(
        quote_sha256(dated_question) == prompt_question_sha256,
        f"provider question binding changed at {ordinal}",
    )
    exact_prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
    _require(
        arm["prompt_token_proxy"] == exact_prompt_tokens
        and arm["prompt_workspace_token_proxy"]
        == exact_prompt_tokens + hot.RESPONDER_OUTPUT_TOKEN_RESERVE
        and arm["context_token_proxy"] <= 7_000
        and arm["prompt_workspace_token_proxy"] <= 8_000
        and arm["raw_evidence_only"] is True,
        f"provider token boundary changed at {ordinal}",
    )
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    _require(
        arm["provider_payload_sha256"] == hashlib.sha256(payload).hexdigest()
        and arm["provider_payload_utf8_bytes"] == len(payload),
        f"provider payload changed at {ordinal}",
    )


def _project_selection(
    construction: Mapping[str, Any],
    *,
    construction_sha256: str,
    runtime_sha256: str,
    replay_sha256: str,
) -> dict[str, Any]:
    source_rows = construction.get("questions")
    _require(
        type(source_rows) is list and len(source_rows) == EXPECTED_QUESTION_COUNT,
        "provider source population changed",
    )
    rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(source_rows):
        _require(
            type(source) is dict
            and source.get("ordinal") == ordinal
            and type(source.get("question_id")) is str,
            "provider source question order changed",
        )
        unsigned_source = dict(source)
        source_receipt = unsigned_source.pop("row_receipt_sha256", None)
        _require(
            source_receipt == identity_sha256(unsigned_source),
            f"provider source row receipt changed at {ordinal}",
        )
        arm = _provider_arm(source.get("effective_arm"))
        dated_question = typed._extract_dated_question(  # noqa: SLF001
            source["effective_arm"]
        )
        _require(
            source.get("prompt_question_sha256") == quote_sha256(dated_question),
            f"provider source question binding changed at {ordinal}",
        )
        _validate_compact_provider_arm(
            arm,
            prompt_question_sha256=source["prompt_question_sha256"],
            ordinal=ordinal,
        )
        body = {
            "arms": {"a3_protected_union": arm},
            "format": ROW_FORMAT,
            "local_ordinal": ordinal % 10,
            "ordinal": ordinal,
            "prompt_question_sha256": source["prompt_question_sha256"],
            "question_id": source["question_id"],
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
        "source_construction_sha256": construction_sha256,
        "source_replay_sha256": replay_sha256,
        "source_runtime_sha256": runtime_sha256,
        "status": "sealed_gold_free_user_envelope_provider_packets",
    }
    assert_gold_blind(selection, path="user_envelope_provider_selection")
    return selection


def materialize(*, source_root: Path, output_root: Path) -> str:
    _require(not output_root.exists(), "provider output root must be unique and absent")
    construction, construction_sha, runtime_sha, replay_sha = _load_source(source_root)
    selection = _project_selection(
        construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
    )
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / SELECTION_NAME,
        selection,
    )
    print(
        f"User-envelope provider selection: {len(selection['questions'])} prompts; "
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
        == "sealed_gold_free_user_envelope_provider_packets"
        and selection.get("gold_fields_present") is False
        and type(selection.get("provider_calls")) is int
        and selection.get("provider_calls") == 0
        and selection.get("population_identity_sha256")
        == EXPECTED_POPULATION_SHA256
        and selection.get("source_construction_sha256")
        == EXPECTED_SOURCE_CONSTRUCTION_SHA256
        and selection.get("source_runtime_sha256")
        == EXPECTED_SOURCE_RUNTIME_SHA256
        and selection.get("source_replay_sha256") == EXPECTED_SOURCE_REPLAY_SHA256
        and type(selection.get("question_count")) is int
        and selection.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "sealed user-envelope provider selection changed",
    )
    require_sha256(
        selection["population_identity_sha256"],
        "population_identity_sha256",
    )
    for field in (
        "source_construction_sha256",
        "source_runtime_sha256",
        "source_replay_sha256",
    ):
        require_sha256(selection[field], field)
    for ordinal, row in enumerate(rows):
        _require(
            type(row) is dict and set(row) == _ROW_KEYS,
            f"provider selection row schema changed at {ordinal}",
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
            and type(row.get("ordinal")) is int
            and row.get("ordinal") == ordinal
            and type(row.get("local_ordinal")) is int
            and row.get("local_ordinal") == ordinal % 10
            and type(row.get("shard_offset")) is int
            and row.get("shard_offset") == ordinal - (ordinal % 10)
            and type(arms) is dict
            and set(arms) == {"a3_protected_union"},
            f"provider selection row changed at {ordinal}",
        )
        _validate_compact_provider_arm(
            arms["a3_protected_union"],
            prompt_question_sha256=row["prompt_question_sha256"],
            ordinal=ordinal,
        )
    assert_gold_blind(selection, path="loaded_user_envelope_provider_selection")
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


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "_load_selection",
    "materialize",
]
