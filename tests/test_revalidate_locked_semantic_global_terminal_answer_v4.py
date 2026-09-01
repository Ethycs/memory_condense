from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import revalidate_locked_semantic_global_terminal_answer_v4 as v4
from tools.matched_eval.artifacts import SealedArtifact


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source() -> tuple[SealedArtifact, tuple[dict[str, object], ...]]:
    plans: list[dict[str, object]] = []
    completions: list[str] = []
    records: list[dict[str, object]] = []
    for index, ordinal in enumerate(v4.EXACT_ORDINALS):
        messages_sha = _sha(f"messages-{ordinal}")
        completion = (
            '{"decision":"keep_parent","prediction":"parent",'
            '"used_handle_ids":[]}'
        )
        plans.append(
            {
                "messages_sha256": messages_sha,
                "ordinal": ordinal,
            }
        )
        completions.append(completion)
        records.append(
            {
                "call_key_sha256": _sha(f"call-{index}"),
                "checkpoint_hit": True,
                "completion": completion,
                "completion_sha256": quote_sha256(completion),
                "messages_sha256": messages_sha,
                "physical_call": False,
                "request_journal_sha256": _sha(f"request-{index}"),
                "response_journal_sha256": _sha(f"response-{index}"),
            }
        )
    batch = {
        "logical_completions": completions,
        "unique_records": records,
        "usage": {
            "checkpoint_hits": v4.QUESTION_COUNT,
            "logical_calls": v4.QUESTION_COUNT,
            "physical_calls": 0,
            "unique_calls": v4.QUESTION_COUNT,
        },
    }
    return (
        SealedArtifact(
            Path("source-run.json"),
            _sha("source-run"),
            {"completion_batch": batch},
        ),
        tuple(plans),
    )


def test_completion_source_requires_checkpoint_only_exact11() -> None:
    run, plans = _source()
    rows = v4._validated_completion_records(run, plans)  # noqa: SLF001
    assert len(rows) == v4.QUESTION_COUNT
    assert all(record["physical_call"] is False for _, record in rows)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("physical", "completion record changed"),
        ("completion_hash", "completion record changed"),
        ("logical_completion", "authenticated prompt journal"),
        ("duplicate_messages", "completion record changed"),
    ),
)
def test_completion_source_tampering_fails_closed(
    mutation: str,
    match: str,
) -> None:
    run, plans = _source()
    payload = deepcopy(run.payload)
    if mutation == "physical":
        payload["completion_batch"]["unique_records"][0]["physical_call"] = True
    elif mutation == "completion_hash":
        payload["completion_batch"]["unique_records"][0][
            "completion_sha256"
        ] = _sha("wrong")
    elif mutation == "logical_completion":
        payload["completion_batch"]["logical_completions"][0] = "different"
    else:
        payload["completion_batch"]["unique_records"][1][
            "messages_sha256"
        ] = payload["completion_batch"]["unique_records"][0][
            "messages_sha256"
        ]
    mutated = SealedArtifact(run.path, _sha(f"mutated-{mutation}"), payload)
    with pytest.raises(
        v4.LockedSemanticGlobalTerminalValidatorV4Error,
        match=match,
    ):
        v4._validated_completion_records(mutated, plans)  # noqa: SLF001


def test_cli_exposes_no_provider_execution_command() -> None:
    parser = v4.build_parser()
    assert set(parser._subparsers._group_actions[0].choices) == {  # noqa: SLF001
        "materialize",
        "replay",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["provider-run"])
