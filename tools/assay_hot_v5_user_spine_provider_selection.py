#!/usr/bin/env python3
"""Publish sealed locked100 packets with user-spine evidence presentation.

Operation B starts from the authenticated v4 construction/runtime/replay
triple.  It does not rerun retrieval: each already-selected evidence population
is passed intact to the deterministic user-spine renderer.  Exact evidence-ID
deduplication happens inside that renderer, after selection, and no unique row
may be omitted.  This adapter performs no benchmark loading or provider I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
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
    HARD_WORKSPACE_TOKEN_CAP,
    OPERATION_AWARE_SYSTEM_PROMPT,
    OUTPUT_TOKEN_RESERVE,
    RENDERER_ID,
    render_user_spine_prompt,
)


FORMAT = "memory-condense-hot-v5-user-spine-provider-selection-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
SELECTION_NAME = "selection.json"
EXPECTED_QUESTION_COUNT = source_assay.EXPECTED_QUESTION_COUNT
EXPECTED_POPULATION_SHA256 = source_assay.EXPECTED_POPULATION_SHA256
EXPECTED_SOURCE_CONSTRUCTION_SHA256 = (
    source_assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256
)
EXPECTED_SOURCE_RUNTIME_SHA256 = source_assay.EXPECTED_SOURCE_RUNTIME_SHA256
EXPECTED_SOURCE_REPLAY_SHA256 = source_assay.EXPECTED_SOURCE_REPLAY_SHA256

DEFAULT_SOURCE_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v4-user-envelope-shadow-full100-20260907-r2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v5-user-spine-provider-full100-20260907-r1"
)

_AGGREGATE_KEYS = frozenset(
    {
        "dedup_excluded_evidence_count",
        "max_prompt_workspace_token_proxy",
        "retained_evidence_count",
        "selected_evidence_count",
        "selected_populations_sha256",
        "unique_selected_rows_omitted",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "aggregate",
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
        "source_effective_arm_sha256",
        "source_row_receipt_sha256",
    }
)
_ARM_KEYS = frozenset(
    {
        "context_sha256",
        "context_token_proxy",
        "dedup_excluded_evidence_count",
        "dedup_excluded_exact_id_bindings",
        "dedup_stage",
        "messages_sha256",
        "packed_chunk_ids",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_messages",
        "provider_payload_sha256",
        "provider_payload_utf8_bytes",
        "raw_evidence_only",
        "rendered_evidence_ids",
        "rendered_parent_ranks",
        "renderer_id",
        "retained_evidence_count",
        "retained_evidence_ids",
        "retained_row_sha256s",
        "selected_evidence_count",
        "selected_evidence_ids",
        "selected_population_sha256",
        "selected_row_sha256s",
        "source_block_count",
        "source_blocks_sha256",
        "unique_selected_rows_omitted",
        "user_spine_receipt_sha256",
    }
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _compact_arm(
    source_value: object,
    *,
    dated_question: str,
    ordinal: int,
) -> dict[str, Any]:
    """Render one selected population and retain its conservation receipts."""

    _require(type(source_value) is dict, f"source effective arm changed at {ordinal}")
    source = source_value
    packed_evidence = source.get("packed_evidence")
    _require(
        type(packed_evidence) is list,
        f"source packed evidence changed at {ordinal}",
    )
    rendered = render_user_spine_prompt(dated_question, packed_evidence)
    selected_ids = [row.get("evidence_id") for row in packed_evidence]
    _require(
        all(type(evidence_id) is str and bool(evidence_id) for evidence_id in selected_ids)
        and rendered["selected_evidence_ids"] == selected_ids
        and rendered["selected_evidence_count"] == len(packed_evidence)
        and rendered["selected_population_sha256"]
        == identity_sha256(packed_evidence),
        f"selected evidence population changed at {ordinal}",
    )
    source_chunk_ids = source.get("packed_chunk_ids")
    _require(
        type(source_chunk_ids) is list and source_chunk_ids == selected_ids,
        f"source packed chunk/evidence IDs changed at {ordinal}",
    )
    retained_ids = rendered["retained_evidence_ids"]
    rendered_ids = rendered["rendered_evidence_ids"]
    _require(
        len(retained_ids) == len(set(selected_ids))
        and len(retained_ids) == len(set(retained_ids))
        and Counter(retained_ids) == Counter(set(selected_ids))
        and Counter(rendered_ids) == Counter(retained_ids)
        and rendered["unique_selected_rows_omitted"] == 0,
        f"user-spine rendering omitted unique evidence at {ordinal}",
    )
    messages = rendered["provider_messages"]
    _require(
        messages[0]
        == {"role": "system", "content": OPERATION_AWARE_SYSTEM_PROMPT},
        f"user-spine system policy differs from Operation A at {ordinal}",
    )
    compact = {
        "context_sha256": rendered["context_sha256"],
        "context_token_proxy": rendered["context_token_proxy"],
        "dedup_excluded_evidence_count": rendered[
            "dedup_excluded_evidence_count"
        ],
        "dedup_excluded_exact_id_bindings": rendered[
            "dedup_excluded_exact_id_bindings"
        ],
        "dedup_stage": rendered["dedup_stage"],
        "messages_sha256": rendered["messages_sha256"],
        # This is the post-dedup provider population.  The complete selected
        # multiset remains independently bound below.
        "packed_chunk_ids": list(retained_ids),
        "prompt_token_proxy": rendered["prompt_token_proxy"],
        "prompt_workspace_token_proxy": rendered[
            "prompt_workspace_token_proxy"
        ],
        "provider_messages": messages,
        "provider_payload_sha256": rendered["provider_payload_sha256"],
        "provider_payload_utf8_bytes": rendered["provider_payload_utf8_bytes"],
        "raw_evidence_only": True,
        "rendered_evidence_ids": list(rendered_ids),
        "rendered_parent_ranks": list(rendered["rendered_parent_ranks"]),
        "renderer_id": rendered["renderer_id"],
        "retained_evidence_count": rendered["retained_evidence_count"],
        "retained_evidence_ids": list(retained_ids),
        "retained_row_sha256s": list(rendered["retained_row_sha256s"]),
        "selected_evidence_count": rendered["selected_evidence_count"],
        "selected_evidence_ids": list(rendered["selected_evidence_ids"]),
        "selected_population_sha256": rendered["selected_population_sha256"],
        "selected_row_sha256s": list(rendered["selected_row_sha256s"]),
        "source_block_count": rendered["source_block_count"],
        "source_blocks_sha256": identity_sha256(rendered["source_blocks"]),
        "unique_selected_rows_omitted": rendered[
            "unique_selected_rows_omitted"
        ],
        "user_spine_receipt_sha256": rendered["receipt_sha256"],
    }
    return compact


def _validate_arm(
    arm_value: object,
    *,
    ordinal: int,
    prompt_question_sha256: object,
) -> None:
    _require(
        type(arm_value) is dict and set(arm_value) == _ARM_KEYS,
        f"user-spine arm schema changed at {ordinal}",
    )
    arm = arm_value
    for name in (
        "context_sha256",
        "messages_sha256",
        "provider_payload_sha256",
        "selected_population_sha256",
        "source_blocks_sha256",
        "user_spine_receipt_sha256",
    ):
        require_sha256(arm[name], f"questions[{ordinal}].{name}")
    require_sha256(
        prompt_question_sha256,  # type: ignore[arg-type]
        f"questions[{ordinal}].prompt_question_sha256",
    )
    for field in (
        "context_token_proxy",
        "dedup_excluded_evidence_count",
        "prompt_token_proxy",
        "prompt_workspace_token_proxy",
        "provider_payload_utf8_bytes",
        "retained_evidence_count",
        "selected_evidence_count",
        "source_block_count",
        "unique_selected_rows_omitted",
    ):
        _require(
            type(arm[field]) is int and arm[field] >= 0,
            f"user-spine {field} changed at {ordinal}",
        )
    selected_ids = arm["selected_evidence_ids"]
    retained_ids = arm["retained_evidence_ids"]
    rendered_ids = arm["rendered_evidence_ids"]
    packed_ids = arm["packed_chunk_ids"]
    selected_hashes = arm["selected_row_sha256s"]
    retained_hashes = arm["retained_row_sha256s"]
    for name, values in (
        ("selected evidence IDs", selected_ids),
        ("retained evidence IDs", retained_ids),
        ("rendered evidence IDs", rendered_ids),
        ("packed chunk IDs", packed_ids),
    ):
        _require(
            type(values) is list
            and all(
                type(value) is str and bool(value) and value.strip() == value
                for value in values
            ),
            f"user-spine {name} changed at {ordinal}",
        )
    _require(
        type(selected_hashes) is list
        and type(retained_hashes) is list
        and all(type(value) is str and len(value) == 64 for value in selected_hashes)
        and all(type(value) is str and len(value) == 64 for value in retained_hashes),
        f"user-spine evidence row hashes changed at {ordinal}",
    )
    _require(
        arm["selected_evidence_count"] == len(selected_ids) == len(selected_hashes)
        and arm["retained_evidence_count"]
        == len(retained_ids)
        == len(retained_hashes)
        and len(retained_ids) == len(set(selected_ids))
        and len(retained_ids) == len(set(retained_ids))
        and Counter(retained_ids) == Counter(set(selected_ids))
        and packed_ids == retained_ids
        and len(rendered_ids) == len(set(rendered_ids))
        and Counter(rendered_ids) == Counter(retained_ids)
        and arm["dedup_excluded_evidence_count"]
        == len(selected_ids) - len(retained_ids)
        and arm["unique_selected_rows_omitted"] == 0,
        f"user-spine evidence conservation changed at {ordinal}",
    )
    bindings = arm["dedup_excluded_exact_id_bindings"]
    _require(
        type(bindings) is list
        and len(bindings) == arm["dedup_excluded_evidence_count"]
        and all(type(binding) is dict for binding in bindings),
        f"user-spine exact-ID dedup bindings changed at {ordinal}",
    )
    for binding in bindings:
        _require(
            set(binding)
            == {
                "evidence_id",
                "excluded_parent_rank",
                "retained_parent_rank",
                "row_sha256",
            }
            and binding["evidence_id"] in retained_ids
            and type(binding["excluded_parent_rank"]) is int
            and 1 <= binding["excluded_parent_rank"] <= len(selected_ids)
            and type(binding["retained_parent_rank"]) is int
            and 1 <= binding["retained_parent_rank"] < binding["excluded_parent_rank"]
            and selected_ids[binding["excluded_parent_rank"] - 1]
            == binding["evidence_id"]
            and selected_ids[binding["retained_parent_rank"] - 1]
            == binding["evidence_id"]
            and selected_hashes[binding["excluded_parent_rank"] - 1]
            == binding["row_sha256"]
            and selected_hashes[binding["retained_parent_rank"] - 1]
            == binding["row_sha256"],
            f"user-spine exact-ID dedup proof changed at {ordinal}",
        )
        require_sha256(binding["row_sha256"], "dedup row_sha256")
    _require(
        arm["dedup_stage"] == "post_selection_exact_evidence_id"
        and arm["renderer_id"] == RENDERER_ID
        and arm["raw_evidence_only"] is True,
        f"user-spine policy metadata changed at {ordinal}",
    )
    messages = arm["provider_messages"]
    _require(
        type(messages) is list
        and len(messages) == 2
        and all(type(message) is dict for message in messages)
        and all(set(message) == {"role", "content"} for message in messages)
        and messages[0]
        == {"role": "system", "content": OPERATION_AWARE_SYSTEM_PROMPT},
        f"user-spine prompt envelope changed at {ordinal}",
    )
    dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
    _require(
        quote_sha256(dated_question) == prompt_question_sha256,
        f"user-spine question binding changed at {ordinal}",
    )
    prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
    _require(
        arm["messages_sha256"] == identity_sha256(messages)
        and arm["prompt_token_proxy"] == prompt_tokens
        and arm["prompt_workspace_token_proxy"] == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and arm["prompt_workspace_token_proxy"] <= HARD_WORKSPACE_TOKEN_CAP,
        f"user-spine workspace boundary changed at {ordinal}",
    )
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    _require(
        arm["provider_payload_sha256"] == hashlib.sha256(payload).hexdigest()
        and arm["provider_payload_utf8_bytes"] == len(payload),
        f"user-spine provider payload changed at {ordinal}",
    )


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = [row["arms"]["a3_protected_union"] for row in rows]
    return {
        "dedup_excluded_evidence_count": sum(
            arm["dedup_excluded_evidence_count"] for arm in arms
        ),
        "max_prompt_workspace_token_proxy": max(
            arm["prompt_workspace_token_proxy"] for arm in arms
        ),
        "retained_evidence_count": sum(
            arm["retained_evidence_count"] for arm in arms
        ),
        "selected_evidence_count": sum(
            arm["selected_evidence_count"] for arm in arms
        ),
        "selected_populations_sha256": identity_sha256(
            [arm["selected_population_sha256"] for arm in arms]
        ),
        "unique_selected_rows_omitted": sum(
            arm["unique_selected_rows_omitted"] for arm in arms
        ),
    }


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
        "user-spine source population changed",
    )
    rows: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for ordinal, source_row in enumerate(source_rows):
        _require(
            type(source_row) is dict
            and source_row.get("ordinal") == ordinal
            and type(source_row.get("question_id")) is str,
            f"user-spine source question order changed at {ordinal}",
        )
        question_id = require_text(source_row["question_id"], "question ID")
        _require(
            question_id not in seen_question_ids,
            f"duplicate user-spine question ID at {ordinal}",
        )
        seen_question_ids.add(question_id)
        source_unsigned = dict(source_row)
        source_receipt = source_unsigned.pop("row_receipt_sha256", None)
        _require(
            source_receipt == identity_sha256(source_unsigned),
            f"user-spine source row receipt changed at {ordinal}",
        )
        source_arm = source_row.get("effective_arm")
        _require(type(source_arm) is dict, f"source arm changed at {ordinal}")
        dated_question = typed._extract_dated_question(source_arm)  # noqa: SLF001
        _require(
            source_row.get("prompt_question_sha256")
            == quote_sha256(dated_question),
            f"user-spine source question binding changed at {ordinal}",
        )
        arm = _compact_arm(
            source_arm,
            dated_question=dated_question,
            ordinal=ordinal,
        )
        _validate_arm(
            arm,
            ordinal=ordinal,
            prompt_question_sha256=source_row["prompt_question_sha256"],
        )
        body = {
            "arms": {"a3_protected_union": arm},
            "format": ROW_FORMAT,
            "local_ordinal": ordinal % 10,
            "ordinal": ordinal,
            "prompt_question_sha256": source_row["prompt_question_sha256"],
            "question_id": question_id,
            "shard_offset": ordinal - (ordinal % 10),
            "source_effective_arm_sha256": identity_sha256(source_arm),
            "source_row_receipt_sha256": source_receipt,
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    aggregate = _aggregate(rows)
    _require(
        aggregate["unique_selected_rows_omitted"] == 0,
        "user-spine projection omitted selected evidence",
    )
    selection = {
        "aggregate": aggregate,
        "format": FORMAT,
        "gold_fields_present": False,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": len(rows),
        "questions": rows,
        "source_construction_sha256": construction_sha256,
        "source_replay_sha256": replay_sha256,
        "source_runtime_sha256": runtime_sha256,
        "status": "sealed_gold_free_user_spine_provider_packets",
    }
    assert_gold_blind(selection, path="user_spine_provider_selection")
    return selection


def materialize(*, source_root: Path, output_root: Path) -> str:
    _require(not output_root.exists(), "provider output root must be unique and absent")
    construction, construction_sha, runtime_sha, replay_sha = (
        source_assay._load_source(source_root)  # noqa: SLF001
    )
    selection = _project_selection(
        construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
    )
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / SELECTION_NAME, selection
    )
    print(
        f"User-spine provider selection: {len(selection['questions'])} prompts; "
        f"max_workspace={selection['aggregate']['max_prompt_workspace_token_proxy']}; "
        f"selection={digest}",
        flush=True,
    )
    return digest


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    selection, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    rows = selection.get("questions")
    aggregate = selection.get("aggregate")
    _require(
        set(selection) == _SELECTION_KEYS
        and selection.get("format") == FORMAT
        and selection.get("status") == "sealed_gold_free_user_spine_provider_packets"
        and selection.get("gold_fields_present") is False
        and selection.get("provider_calls") == 0
        and selection.get("population_identity_sha256")
        == EXPECTED_POPULATION_SHA256
        and selection.get("source_construction_sha256")
        == EXPECTED_SOURCE_CONSTRUCTION_SHA256
        and selection.get("source_runtime_sha256")
        == EXPECTED_SOURCE_RUNTIME_SHA256
        and selection.get("source_replay_sha256")
        == EXPECTED_SOURCE_REPLAY_SHA256
        and selection.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT
        and type(aggregate) is dict
        and set(aggregate) == _AGGREGATE_KEYS,
        "sealed user-spine provider selection changed",
    )
    for field in (
        "population_identity_sha256",
        "source_construction_sha256",
        "source_runtime_sha256",
        "source_replay_sha256",
    ):
        require_sha256(selection[field], field)
    question_ids: list[str] = []
    for ordinal, row in enumerate(rows):
        _require(
            type(row) is dict and set(row) == _ROW_KEYS,
            f"user-spine row schema changed at {ordinal}",
        )
        unsigned = dict(row)
        receipt = unsigned.pop("row_receipt_sha256", None)
        require_sha256(receipt, f"questions[{ordinal}].row_receipt_sha256")
        for field in ("source_effective_arm_sha256", "source_row_receipt_sha256"):
            require_sha256(row[field], f"questions[{ordinal}].{field}")
        require_sha256(
            row["prompt_question_sha256"],
            f"questions[{ordinal}].prompt_question_sha256",
        )
        question_ids.append(require_text(row["question_id"], "question ID"))
        arms = row["arms"]
        _require(
            receipt == identity_sha256(unsigned)
            and row.get("format") == ROW_FORMAT
            and row.get("ordinal") == ordinal
            and row.get("local_ordinal") == ordinal % 10
            and row.get("shard_offset") == ordinal - (ordinal % 10)
            and type(arms) is dict
            and set(arms) == {"a3_protected_union"},
            f"user-spine row changed at {ordinal}",
        )
        _validate_arm(
            arms["a3_protected_union"],
            ordinal=ordinal,
            prompt_question_sha256=row["prompt_question_sha256"],
        )
    _require(
        len(question_ids) == len(set(question_ids)),
        "user-spine question IDs are not unique",
    )
    _require(
        aggregate == _aggregate(rows)
        and aggregate["unique_selected_rows_omitted"] == 0
        and aggregate["max_prompt_workspace_token_proxy"]
        <= HARD_WORKSPACE_TOKEN_CAP,
        "user-spine aggregate changed",
    )
    require_sha256(
        aggregate["selected_populations_sha256"],
        "aggregate.selected_populations_sha256",
    )
    assert_gold_blind(selection, path="loaded_user_spine_provider_selection")
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
