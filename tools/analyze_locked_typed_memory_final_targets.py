#!/usr/bin/env python3
"""Post-hoc target coverage for the sealed typed-memory final composition.

The typed composition, its exact final prompt population, and an optional
materialized answer run are fully verified before this program opens either
the historical gold-bearing judge/score or the immutable target-owner plan.
The target plan is never consulted by retrieval, selection, deduplication,
packing, operator execution, or prompt construction.

This is a coverage assay, not an accuracy score.  In particular, an exhaustive
physical scan is never treated as proof of semantic absence.  Coverage-check
targets need their declared source witnesses *and* explicit typed slot support.
Relation targets need every declared source operand plus a story/operator link
that survived the final prompt boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from tools import run_locked_typed_memory_final_arm as typed_cli  # noqa: E402
from tools.build_locked_retrieval_target_registry import (  # noqa: E402
    _validate_plan,
)
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_connectivity_ledger import (  # noqa: E402
    FORMAT as CONNECTIVITY_FORMAT,
)
from tools.matched_eval.typed_lane_allocator import (  # noqa: E402
    SURPLUS_FORMAT,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    COMPOSITION_FORMAT,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    render_final_messages,
)


FORMAT = "memory-condense-locked-typed-final-target-coverage-v3"
OUTPUT_NAME = "typed-memory-final-target-coverage-v3.json"
STAGES = (
    "retrieved_selected",
    "post_dedup_retained",
    "lane_selected",
    "surplus_fill_retained",
    "fair_merge_retained",
    "final_fit_retained",
    "globally_bound",
    "operator_consumed",
)
ANSWERER_STAGE = "answerer_cited"

DEFAULT_TYPED_ROOT = typed_cli.DEFAULT_OUTPUT
DEFAULT_BASELINE_ROOT = Path(
    "eval_results/matched_eval_100/adaptive-solver-v3-dg-judge"
)
DEFAULT_TARGET_PLAN = Path(
    "docs/10 - Research Log/data/"
    "longmemeval-locked-100-target-owner-plan-v1.json"
)
DEFAULT_OUTPUT = DEFAULT_TYPED_ROOT / OUTPUT_NAME

BASELINE_JUDGE_NAME = "semantic-judge-sol.json"
BASELINE_JUDGE_REPLAY_NAME = "semantic-judge-sol-replay.json"
BASELINE_SCORE_NAME = "score-ledger.json"
BASELINE_SCORE_REPLAY_NAME = "score-ledger-replay.json"
EXPECTED_BASELINE_JUDGE_SHA256 = (
    "5ba3ab34ec099ebfa94d87d4247a0c12163e8b02a7e370c44febde3e903ae967"
)
EXPECTED_BASELINE_SCORE_SHA256 = (
    "6acc71cca864460d261fbd90e63580a395f4833b025ec5b5b9d6d474923a2c04"
)
PINNED_TARGET_PLAN_FILE_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)
PINNED_TARGET_PLAN_IDENTITY_SHA256 = (
    "2cabfbb103929c68dea47368502875444903ced282c708cba45ef26bee14d888"
)
BASELINE_MISS_ORDINALS = (
    6,
    7,
    14,
    16,
    27,
    28,
    31,
    36,
    37,
    42,
    43,
    52,
    53,
    54,
    61,
    65,
    67,
    69,
    72,
    75,
    77,
    79,
    81,
    82,
    86,
    93,
    94,
    97,
)
EXPECTED_QUESTION_COUNT = 100


class TypedFinalTargetCoverageError(ValueError):
    """Raised when a seal, post-hoc firewall, or lifecycle join changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedFinalTargetCoverageError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _ordered_text(value: object, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label)
    _require(
        all(type(row) is str and row and row.strip() == row for row in rows)
        and len(rows) == len(set(rows)),
        f"{label} must be ordered unique nonempty text",
    )
    return tuple(rows)


def _verify_identity_field(
    raw: Mapping[str, Any], field: str, label: str
) -> str:
    declared = require_sha256(raw.get(field), label)
    unsigned = dict(raw)
    unsigned.pop(field)
    _require(identity_sha256(unsigned) == declared, f"{label} changed")
    return declared


@dataclass(frozen=True, slots=True)
class VerifiedTypedRuntime:
    composition: SealedArtifact
    preflight: SealedArtifact
    answer_run: SealedArtifact | None
    composition_rows: tuple[dict[str, Any], ...]
    preflight_rows: tuple[dict[str, Any], ...]
    answer_used_handles: tuple[frozenset[str], ...] | None


def _verify_composition(path: Path, expected_sha256: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "typed composition SHA-256"),
        "typed composition file hash changed",
    )
    payload = artifact.payload
    assert_gold_blind(payload, path="posthoc_typed_composition_verification")
    rows = _exact_list(payload.get("questions"), "typed composition questions")
    _require(
        payload.get("format") == COMPOSITION_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "typed composition envelope changed",
    )
    question_ids: list[str] = []
    for ordinal, value in enumerate(rows):
        row = _exact_dict(value, f"typed composition row {ordinal}")
        _verify_identity_field(
            row, "composition_row_sha256", f"typed composition row {ordinal}"
        )
        provider = _exact_dict(
            row.get("provider_projection"), f"typed provider row {ordinal}"
        )
        provider_input = _exact_dict(
            provider.get("provider_input"), f"typed provider input {ordinal}"
        )
        messages = render_final_messages(provider_input)
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        allowed = _ordered_text(
            row.get("allowed_handle_ids"), f"allowed handles {ordinal}"
        )
        handle_groups = _exact_dict(
            row.get("handle_group_by_id"), f"handle groups {ordinal}"
        )
        _require(
            row.get("ordinal") == ordinal
            and row.get("format") == COMPOSITION_FORMAT
            and set(handle_groups) == set(allowed)
            and identity_sha256(list(messages))
            == provider.get("messages_sha256")
            and prompt_tokens == provider.get("prompt_token_proxy")
            and provider.get("full_chat_plus_output_tokens")
            == prompt_tokens + OUTPUT_TOKEN_RESERVE
            and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS,
            f"typed composition prompt/binding changed at {ordinal}",
        )
        question_ids.append(require_text(row.get("question_id"), "question ID"))
    _require(
        len(set(question_ids)) == EXPECTED_QUESTION_COUNT,
        "typed composition question IDs repeat",
    )
    return artifact


def _verify_preflight(
    path: Path,
    expected_sha256: str,
    composition: SealedArtifact,
) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "typed preflight SHA-256"),
        "typed preflight file hash changed",
    )
    payload = artifact.payload
    assert_gold_blind(payload, path="posthoc_typed_preflight_verification")
    rows = _exact_list(payload.get("physical_prompt_rows"), "typed prompt rows")
    composition_rows = composition.payload["questions"]
    _require(
        payload.get("format") == typed_cli.PREFLIGHT_FORMAT
        and payload.get("composition_artifact_sha256") == composition.sha256
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "typed preflight envelope changed",
    )
    for ordinal, (value, composition_value) in enumerate(
        zip(rows, composition_rows, strict=True)
    ):
        row = _exact_dict(value, f"typed prompt row {ordinal}")
        composition_row = _exact_dict(
            composition_value, f"typed composition row {ordinal}"
        )
        _verify_identity_field(
            row, "prompt_row_receipt_sha256", f"typed prompt row {ordinal}"
        )
        messages = _exact_list(row.get("messages"), f"typed messages {ordinal}")
        expected_messages = list(
            render_final_messages(
                composition_row["provider_projection"]["provider_input"]
            )
        )
        _require(
            row.get("ordinal") == ordinal
            and row.get("question_id") == composition_row.get("question_id")
            and row.get("question_sha256") == composition_row.get("question_sha256")
            and row.get("composition_row_sha256")
            == composition_row.get("composition_row_sha256")
            and row.get("typed_composition_receipt_sha256")
            == composition_row.get("typed_composition_receipt_sha256")
            and messages == expected_messages
            and row.get("messages_sha256") == identity_sha256(messages)
            and row.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(messages),
            f"typed preflight row escaped composition at {ordinal}",
        )
    return artifact


def _verify_answer_run(
    path: Path,
    expected_sha256: str,
    composition: SealedArtifact,
    preflight: SealedArtifact,
) -> tuple[SealedArtifact, tuple[frozenset[str], ...]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "typed run SHA-256"),
        "typed answer run file hash changed",
    )
    payload = artifact.payload
    assert_gold_blind(payload, path="posthoc_typed_answer_run_verification")
    rows = _exact_list(payload.get("questions"), "typed answer rows")
    prompt_rows = preflight.payload["physical_prompt_rows"]
    composition_rows = composition.payload["questions"]
    _require(
        payload.get("format") == typed_cli.RUN_FORMAT
        and payload.get("composition_artifact_sha256") == composition.sha256
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "typed answer run envelope changed",
    )
    used_by_row: list[frozenset[str]] = []
    for ordinal, (value, prompt, composition_row) in enumerate(
        zip(rows, prompt_rows, composition_rows, strict=True)
    ):
        row = _exact_dict(value, f"typed answer row {ordinal}")
        _verify_identity_field(
            row, "source_row_sha256", f"typed answer row {ordinal}"
        )
        used = frozenset(
            _ordered_text(row.get("used_handle_ids"), f"answer handles {ordinal}")
        )
        allowed = set(composition_row["allowed_handle_ids"])
        _require(
            row.get("ordinal") == ordinal
            and row.get("question_id") == composition_row.get("question_id")
            and row.get("question_sha256") == composition_row.get("question_sha256")
            and row.get("prompt_row_receipt_sha256")
            == prompt.get("prompt_row_receipt_sha256")
            and used <= allowed
            and (row.get("decision") == "replace" or not used),
            f"typed answer row binding changed at {ordinal}",
        )
        used_by_row.append(used)
    return artifact, tuple(used_by_row)


def verify_typed_runtime(
    *,
    typed_root: Path,
    expected_composition_sha256: str,
    expected_preflight_sha256: str,
    answer_run_path: Path | None = None,
    expected_answer_run_sha256: str | None = None,
) -> VerifiedTypedRuntime:
    """Verify every deployable artifact before opening post-hoc inputs."""

    composition = _verify_composition(
        typed_root / typed_cli.COMPOSITION_NAME, expected_composition_sha256
    )
    preflight = _verify_preflight(
        typed_root / typed_cli.PREFLIGHT_NAME,
        expected_preflight_sha256,
        composition,
    )
    answer_run: SealedArtifact | None = None
    used: tuple[frozenset[str], ...] | None = None
    if answer_run_path is not None or expected_answer_run_sha256 is not None:
        _require(
            answer_run_path is not None and expected_answer_run_sha256 is not None,
            "typed answer run path and expected hash must be supplied together",
        )
        answer_run, used = _verify_answer_run(
            answer_run_path,
            expected_answer_run_sha256,
            composition,
            preflight,
        )
    return VerifiedTypedRuntime(
        composition,
        preflight,
        answer_run,
        tuple(dict(row) for row in composition.payload["questions"]),
        tuple(dict(row) for row in preflight.payload["physical_prompt_rows"]),
        used,
    )


@dataclass(frozen=True, slots=True)
class VerifiedBaseline:
    judge: SealedArtifact
    score: SealedArtifact
    miss_ordinals: tuple[int, ...]


def _verify_baseline(
    root: Path,
    composition_rows: Sequence[Mapping[str, Any]],
) -> VerifiedBaseline:
    judge = read_sealed_json(root / BASELINE_JUDGE_NAME)
    judge_replay = read_sealed_json(root / BASELINE_JUDGE_REPLAY_NAME)
    score = read_sealed_json(root / BASELINE_SCORE_NAME)
    score_replay = read_sealed_json(root / BASELINE_SCORE_REPLAY_NAME)
    _require(
        judge.sha256 == judge_replay.sha256 == EXPECTED_BASELINE_JUDGE_SHA256
        and judge.payload == judge_replay.payload,
        "baseline judge/replay changed",
    )
    _require(
        score.sha256 == score_replay.sha256 == EXPECTED_BASELINE_SCORE_SHA256
        and score.payload == score_replay.payload,
        "baseline score/replay changed",
    )
    judge_rows = _exact_list(judge.payload.get("questions"), "baseline judge rows")
    score_rows = _exact_list(score.payload.get("rows"), "baseline score rows")
    _require(
        judge.payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and score.payload.get("row_count") == EXPECTED_QUESTION_COUNT
        and len(judge_rows) == len(score_rows) == EXPECTED_QUESTION_COUNT,
        "baseline population changed",
    )
    misses: list[int] = []
    for ordinal, (judge_value, score_value, composition_row) in enumerate(
        zip(judge_rows, score_rows, composition_rows, strict=True)
    ):
        judge_row = _exact_dict(judge_value, f"baseline judge row {ordinal}")
        score_row = _exact_dict(score_value, f"baseline score row {ordinal}")
        judge_sha = _verify_identity_field(
            judge_row, "judge_row_sha256", f"baseline judge row {ordinal}"
        )
        correct = judge_row.get("correct")
        _require(
            type(correct) is bool
            and judge_row.get("ordinal") == ordinal
            and judge_row.get("question_id") == composition_row.get("question_id")
            and judge_row.get("question_sha256")
            == composition_row.get("question_sha256")
            and score_row.get("judge_row_sha256") == judge_sha
            and score_row.get("correct") is correct,
            f"baseline judge/score/composition binding changed at {ordinal}",
        )
        if not correct:
            misses.append(ordinal)
    score_unsigned = dict(score.payload)
    score_identity = require_sha256(
        score_unsigned.pop("ledger_identity_sha256", None),
        "baseline score identity",
    )
    _require(
        identity_sha256(score_unsigned) == score_identity
        and judge.payload.get("aggregate", {}).get("correct") == 72
        and judge.payload.get("aggregate", {}).get("incorrect") == 28
        and score.payload.get("aggregate", {}).get("candidate_correct") == 72
        and tuple(misses) == BASELINE_MISS_ORDINALS,
        "baseline 72/100 miss checkpoint changed",
    )
    return VerifiedBaseline(judge, score, tuple(misses))


def _load_target_plan(
    path: Path,
    composition_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == PINNED_TARGET_PLAN_FILE_SHA256,
        "target-plan file differs from pinned checkpoint",
    )
    plan = _validate_plan(artifact.payload)
    _require(
        plan.get("plan_sha256") == PINNED_TARGET_PLAN_IDENTITY_SHA256
        and plan.get("runtime_use_forbidden") is True
        and plan.get("gold_target_tags_posthoc_only") is True
        and plan.get("provider_calls") == 0
        and isinstance(plan.get("policy"), Mapping)
        and plan["policy"].get("runtime_use_forbidden") is True
        and plan["policy"].get("gold_posthoc_only") is True,
        "target plan lost its post-hoc-only firewall",
    )
    expected_order = [
        {"ordinal": index, "question_id": row["question_id"]}
        for index, row in enumerate(composition_rows)
    ]
    _require(
        plan.get("ordered_question_keys") == expected_order,
        "target plan question order differs from typed composition",
    )
    return plan, artifact.sha256


def _validate_receipt_projection(raw: object, label: str) -> dict[str, Any]:
    row = _exact_dict(raw, label)
    _verify_identity_field(row, "receipt_sha256", label)
    return row


def _source_aliases(source_ids: Iterable[str], question_id: str) -> set[str]:
    prefix = f"{question_id}::"
    aliases: set[str] = set()
    for source_id in source_ids:
        aliases.add(source_id)
        if source_id.startswith(prefix):
            aliases.add(source_id[len(prefix) :])
    return aliases


def _local_source_map(local_audit: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Join prompt-external locator/binding receipts to raw source IDs."""

    source_by_locator: dict[str, set[str]] = defaultdict(set)

    map_audit = _exact_dict(local_audit.get("adaptive_parent_map"), "map audit")
    for value in _exact_list(map_audit.get("exact_item_bindings"), "map bindings"):
        row = _exact_dict(value, "map binding")
        binding = _validate_receipt_projection(row.get("binding"), "map local binding")
        alias = _exact_dict(row.get("payload_alias"), "map payload alias")
        source_by_locator[
            require_sha256(binding.get("local_source_locator_sha256"), "map locator")
        ].add(require_text(alias.get("source_id"), "map source ID"))

    for audit_key in ("adaptive_parent_source", "adaptive_tail_source"):
        raw_audit = local_audit.get(audit_key)
        if raw_audit is None:
            continue
        audit = _exact_dict(raw_audit, audit_key)
        direct = {
            require_text(row.get("evidence_id"), "direct evidence ID"): row
            for row in (
                _exact_dict(value, "direct evidence")
                for value in _exact_list(audit.get("direct_evidence"), "direct evidence")
            )
        }
        exclusions = {
            require_sha256(row.get("receipt_sha256"), "direct exclusion"): row
            for row in (
                _exact_dict(value, "direct exclusion")
                for value in _exact_list(
                    audit.get("direct_exclusions"), "direct exclusions"
                )
            )
        }
        fact_sources: dict[str, tuple[str, ...]] = {}
        for value in _exact_list(
            audit.get("source_fact_admission_bindings"), "source fact bindings"
        ):
            row = _validate_receipt_projection(value, "source fact audit binding")
            fact_sources[require_sha256(
                row.get("binding_receipt_sha256"), "source fact binding"
            )] = tuple(
                require_text(origin.get("source_id"), "fact origin source")
                for origin in (
                    _exact_dict(origin_value, "fact origin")
                    for origin_value in _exact_list(
                        row.get("exact_origins"), "fact exact origins"
                    )
                )
            )
        for contribution_value in _exact_list(
            audit.get("contributions"), "union contributions"
        ):
            contribution = _exact_dict(contribution_value, "union contribution")
            for binding_value in _exact_list(
                contribution.get("bindings"), "union bindings"
            ):
                binding = _validate_receipt_projection(
                    binding_value, "union local binding"
                )
                locator = require_sha256(
                    binding.get("local_source_locator_sha256"), "union locator"
                )
                binding_receipt = require_sha256(
                    binding.get("receipt_sha256"), "union binding receipt"
                )
                if binding.get("origin") == "source_fact":
                    sources = fact_sources.get(binding_receipt)
                elif binding.get("origin") == "direct_pointer":
                    exclusion = exclusions.get(binding.get("evidence_receipt_sha256"))
                    _require(exclusion is not None, "direct pointer lost its exclusion")
                    matching_ids = _ordered_text(
                        exclusion.get("matching_direct_evidence_ids"),
                        "direct matches",
                    )
                    _require(
                        all(evidence_id in direct for evidence_id in matching_ids),
                        "direct pointer cites missing protected evidence",
                    )
                    sources = tuple(
                        require_text(direct[evidence_id].get("source_id"), "direct source")
                        for evidence_id in matching_ids
                    )
                else:
                    raise TypedFinalTargetCoverageError("union binding origin changed")
                _require(bool(sources), "union binding lost every exact source")
                source_by_locator[locator].update(sources)

    full = _exact_dict(
        local_audit.get("full_store_slot_closure"), "full-store audit"
    )
    for value in _exact_list(
        full.get("local_citation_bindings"), "full-store local citations"
    ):
        row = _exact_dict(value, "full-store citation row")
        local = _validate_receipt_projection(
            row.get("local_citation_binding"), "full-store citation"
        )
        source_by_locator[identity_sha256(local)].add(
            require_text(local.get("source_id"), "full-store source")
        )

    active = _exact_dict(
        local_audit.get("active_reconstruction"), "active reconstruction audit"
    )
    active_result = _exact_dict(active.get("local_result"), "active local result")
    for value in _exact_list(
        active_result.get("local_bindings"), "active local citations"
    ):
        local = _validate_receipt_projection(value, "active local citation")
        source_by_locator[require_sha256(
            local.get("receipt_sha256"), "active locator"
        )].add(require_text(local.get("source_id"), "active source"))

    return {
        locator: frozenset(sources)
        for locator, sources in source_by_locator.items()
    }


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    handle_id: str
    binding_receipt_sha256: str
    local_source_locator_sha256: str
    mechanism_id: str
    lane_id: str
    source_group_handle: str
    source_ids: frozenset[str]
    supported_slot_ids: frozenset[str]
    story_link_ids: frozenset[str]
    post_dedup_retained: bool
    lane_selected: bool
    surplus_fill_retained: bool
    fair_merge_retained: bool
    final_fit_retained: bool
    globally_bound: bool
    operator_consumed: bool


@dataclass(frozen=True, slots=True)
class QuestionLifecycle:
    records: tuple[EvidenceRecord, ...]
    required_slot_ids: frozenset[str]
    answerer_used_handle_ids: frozenset[str] | None
    story_link_basis_by_id: Mapping[str, str]


def _story_links(
    row: Mapping[str, Any], local_audit: Mapping[str, Any]
) -> dict[str, str]:
    story = _exact_dict(row.get("story_coherence"), "story coherence")
    provider_story = _exact_dict(
        row["provider_projection"]["provider_input"].get("story_coherence"),
        "provider story coherence",
    )
    _require(story == provider_story, "local/provider story coherence changed")
    local_links: dict[str, tuple[str, ...]] = {}
    for value in _exact_list(
        local_audit.get("story_link_local_bindings"), "story local bindings"
    ):
        binding = _validate_receipt_projection(value, "story local binding")
        link_id = binding.get("link_id")
        if link_id is None:
            continue
        local_links[require_text(link_id, "story link ID")] = _ordered_text(
            binding.get("group_handles"), "story link groups"
        )
    result: dict[str, str] = {}
    for value in _exact_list(story.get("link_overlays"), "story overlays"):
        overlay = _exact_dict(value, "story overlay")
        link_id = require_text(overlay.get("link_id"), "story overlay ID")
        groups = _ordered_text(overlay.get("group_handles"), "story overlay groups")
        _require(local_links.get(link_id) == groups, "exact story overlay lost local proof")
        result[link_id] = "exact_story_overlay"
    for index, value in enumerate(
        _exact_list(story.get("group_links"), "story content links"), start=1
    ):
        link = _exact_dict(value, "story content link")
        require_text(link.get("left_group"), "story left group")
        require_text(link.get("right_group"), "story right group")
        result[f"group_links:{index}"] = "content_operator_link"
    return result


def _lane_bindings(
    local_audit: Mapping[str, Any],
) -> tuple[dict[str, str], frozenset[str], frozenset[str]]:
    allocation = _exact_dict(
        local_audit.get("non_borrowable_lane_allocation"), "lane allocation"
    )
    unsigned_allocation = dict(allocation)
    declared = require_sha256(
        unsigned_allocation.pop("receipt_sha256", None),
        "lane allocation receipt",
    )
    _require(
        declared == identity_sha256(unsigned_allocation),
        "lane allocation receipt changed",
    )
    mechanism_lane: dict[str, str] = {}
    selected: set[str] = set()
    omitted: set[str] = set()
    for value in _exact_list(allocation.get("lane_receipts"), "lane receipts"):
        lane = _validate_receipt_projection(value, "lane receipt")
        lane_id = require_text(lane.get("lane_id"), "lane ID")
        for mechanism in _ordered_text(lane.get("mechanism_ids"), "lane mechanisms"):
            _require(mechanism not in mechanism_lane, "mechanism belongs to two lanes")
            mechanism_lane[mechanism] = lane_id
        selected.update(
            _ordered_text(lane.get("selected_binding_receipt_sha256s"), "lane selected")
        )
        omitted.update(
            _ordered_text(lane.get("omitted_binding_receipt_sha256s"), "lane omitted")
        )
    _require(not (selected & omitted), "lane binding partitions overlap")
    return mechanism_lane, frozenset(selected), frozenset(omitted)


def _surplus_added_bindings(
    local_audit: Mapping[str, Any],
    *,
    lane_selected: frozenset[str],
    lane_omitted: frozenset[str],
) -> frozenset[str]:
    value = local_audit.get("shared_lane_surplus_fill")
    if value is None:
        return frozenset()
    audit = _exact_dict(value, "shared lane surplus fill")
    unsigned = dict(audit)
    declared = require_sha256(
        unsigned.pop("receipt_sha256", None), "shared lane surplus receipt"
    )
    added = frozenset(
        _ordered_text(
            audit.get("added_binding_receipt_sha256s"),
            "shared lane surplus added bindings",
        )
    )
    _require(
        identity_sha256(unsigned) == declared
        and audit.get("format") == SURPLUS_FORMAT
        and audit.get("gold_loaded") is False
        and audit.get("provider_prompt_count") == 0
        and audit.get("retained_transformer_token_state_bytes") == 0
        and type(audit.get("base_content_token_proxy")) is int
        and type(audit.get("final_content_token_proxy")) is int
        and type(audit.get("shared_final_content_token_cap")) is int
        and 0 <= int(audit["base_content_token_proxy"])
        <= int(audit["final_content_token_proxy"])
        <= int(audit["shared_final_content_token_cap"])
        and not (added & lane_selected)
        and added <= lane_omitted,
        "shared lane surplus receipt or binding partition changed",
    )
    return added


def _fair_merge_partitions(
    local_audit: Mapping[str, Any],
) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """Verify the fair-premerge item and allocated-binding partitions."""

    audit = _exact_dict(local_audit.get("fair_premerge"), "fair premerge audit")
    unsigned = dict(audit)
    declared = require_sha256(
        unsigned.pop("receipt_sha256", None), "fair premerge receipt"
    )
    fair_format = audit.get("format")
    _require(
        identity_sha256(unsigned) == declared
        and fair_format
        in {
            f"{typed_cli.FORMAT}-fair-premerge-audit-v1",
            f"{typed_cli.FORMAT}-fair-premerge-audit-v2",
            f"{typed_cli.FORMAT}-fair-premerge-audit-v3",
        },
        "fair premerge receipt or format changed",
    )
    protected_top: frozenset[str] | None = None
    if fair_format == f"{typed_cli.FORMAT}-fair-premerge-audit-v3":
        protected_top = frozenset(
            _ordered_text(
                audit.get("protected_minimum_item_receipt_sha256s"),
                "fair protected minimum items",
            )
        )
        lane_audit = _exact_dict(
            local_audit.get("non_borrowable_lane_allocation"),
            "non-borrowable lane allocation",
        )
        surplus_audit = _exact_dict(
            local_audit.get("shared_lane_surplus_fill"),
            "shared lane surplus fill",
        )
        surplus_partitions = _exact_dict(
            audit.get("surplus_partitions"),
            "fair surplus partitions",
        )
        _require(
            audit.get("gold_loaded") is False
            and audit.get("provider_prompt_count") == 0
            and audit.get("retained_transformer_token_state_bytes") == 0
            and audit.get("minimum_allocation_receipt_sha256")
            == require_sha256(
                lane_audit.get("allocation_receipt_sha256"),
                "underlying lane allocation receipt",
            )
            and audit.get("shared_lane_surplus_fill_receipt_sha256")
            == require_sha256(
                surplus_audit.get("receipt_sha256"),
                "shared lane surplus receipt",
            )
            and audit.get("input_contribution_receipt_sha256s")
            == surplus_audit.get("contribution_receipt_sha256s")
            and audit.get("protected_minimum_item_receipt_sha256s")
            == surplus_audit.get("minimum_item_receipt_sha256s")
            and audit.get("protected_minimum_binding_receipt_sha256s")
            == surplus_audit.get("minimum_binding_receipt_sha256s")
            and all(
                surplus_partitions.get(key) == surplus_audit.get(key)
                for key in (
                    "added_binding_receipt_sha256s",
                    "added_item_receipt_sha256s",
                    "budget_omitted_item_receipt_sha256s",
                    "ineligible_item_receipt_sha256s",
                )
            )
            and require_sha256(
                audit.get("packet_receipt_sha256"),
                "fair packet receipt",
            )
            == audit.get("packet_receipt_sha256"),
            "fair protected receipt chain changed",
        )
    admitted_by_mechanism: dict[str, frozenset[str]] = {}
    all_items: set[str] = set()
    for value in _exact_list(audit.get("mechanisms"), "fair mechanisms"):
        row = _exact_dict(value, "fair mechanism row")
        mechanism = require_text(row.get("mechanism_id"), "fair mechanism")
        admitted = frozenset(
            _ordered_text(
                row.get("admitted_item_receipt_sha256s"), "fair admitted items"
            )
        )
        dropped = frozenset(
            _ordered_text(
                row.get("dropped_item_receipt_sha256s"), "fair dropped items"
            )
        )
        if fair_format == f"{typed_cli.FORMAT}-fair-premerge-audit-v3":
            protected_values = frozenset(
                _ordered_text(
                    row.get("protected_minimum_item_receipt_sha256s"),
                    "fair mechanism protected items",
                )
            )
        else:
            protected = row.get("protected_minimum_item_receipt_sha256")
            protected_values = frozenset(
                ()
                if protected is None
                else (require_sha256(protected, "fair protected"),)
            )
        _require(
            mechanism not in admitted_by_mechanism
            and not (admitted & dropped)
            and len(admitted | dropped) == row.get("accepted_candidate_count")
            and type(row.get("usable_candidate_count")) is int
            and 0
            <= int(row["usable_candidate_count"])
            <= int(row["accepted_candidate_count"])
            and type(row.get("parser_rejected_count")) is int
            and int(row["parser_rejected_count"]) >= 0
            and protected_values <= admitted
            and not (all_items & (admitted | dropped)),
            "fair mechanism item partition changed",
        )
        admitted_by_mechanism[mechanism] = admitted
        all_items.update(admitted | dropped)
    if protected_top is not None:
        _require(
            protected_top
            == frozenset(
                receipt
                for value in _exact_list(
                    audit.get("mechanisms"), "fair mechanisms"
                )
                for receipt in _ordered_text(
                    _exact_dict(value, "fair mechanism row").get(
                        "protected_minimum_item_receipt_sha256s"
                    ),
                    "fair mechanism protected items",
                )
            )
            and protected_top <= all_items,
            "fair protected item partition changed",
        )

    dropped_bindings: set[str] = set()
    for value in _exact_list(
        local_audit.get("fair_premerge_dropped_allocated_bindings"),
        "fair dropped allocated bindings",
    ):
        binding = _validate_receipt_projection(value, "fair dropped binding")
        dropped_bindings.add(
            require_sha256(binding.get("receipt_sha256"), "fair dropped binding")
        )
    return admitted_by_mechanism, frozenset(dropped_bindings)


def _question_lifecycle(
    row: Mapping[str, Any],
    *,
    answerer_used_handles: frozenset[str] | None,
) -> QuestionLifecycle:
    local = _exact_dict(row.get("local_audit"), "typed local audit")
    source_by_locator = _local_source_map(local)
    mechanism_lane, lane_selected, lane_omitted = _lane_bindings(local)
    surplus_added = _surplus_added_bindings(
        local,
        lane_selected=lane_selected,
        lane_omitted=lane_omitted,
    )
    post_surplus_selected = set(lane_selected) | set(surplus_added)
    fair_items_by_mechanism, fair_dropped_bindings = _fair_merge_partitions(local)
    fair_audit = _exact_dict(local.get("fair_premerge"), "fair premerge audit")
    if fair_audit.get("format") == (
        f"{typed_cli.FORMAT}-fair-premerge-audit-v3"
    ):
        provider_projection = _exact_dict(
            row.get("provider_projection"), "typed provider projection"
        )
        protected_items = _ordered_text(
            fair_audit.get("protected_minimum_item_receipt_sha256s"),
            "fair protected minimum items",
        )
        protected_bindings = _ordered_text(
            fair_audit.get("protected_minimum_binding_receipt_sha256s"),
            "fair protected minimum bindings",
        )
        dropped_items = frozenset(
            _ordered_text(
                provider_projection.get("dropped_item_receipt_sha256s"),
                "fitted dropped item receipts",
            )
        )
        _require(
            provider_projection.get("protection_source_receipt_sha256")
            == fair_audit.get("receipt_sha256")
            and provider_projection.get("protected_item_receipt_sha256s")
            == list(protected_items)
            and provider_projection.get("protected_binding_receipt_sha256s")
            == list(protected_bindings)
            and not (set(protected_items) & dropped_items),
            "hard fit protection chain changed",
        )
    dedup_excluded: set[str] = set()
    for value in _exact_list(
        local.get("post_selection_dedup_exclusions"), "dedup exclusions"
    ):
        exclusion = _exact_dict(value, "dedup exclusion")
        _require(
            exclusion.get("operation_position")
            == "after_each_mechanism_selection",
            "dedup moved before mechanism selection",
        )
        dedup_excluded.update(
            _ordered_text(
                exclusion.get("duplicate_binding_receipt_sha256s"),
                "dedup binding receipts",
            )
        )
    retained_bindings = {
        require_sha256(binding.get("receipt_sha256"), "fitted binding receipt"):
        require_text(binding.get("handle_id"), "fitted binding handle")
        for binding in (
            _validate_receipt_projection(value, "fitted binding")
            for value in _exact_list(
                local.get("retained_fitted_bindings"), "fitted bindings"
            )
        )
    }
    provider = _exact_dict(row.get("provider_projection"), "provider projection")
    final_fit_dropped_bindings = frozenset(
        require_sha256(value, "final-fit dropped binding")
        for value in _ordered_text(
            provider.get("dropped_binding_receipt_sha256s"),
            "final-fit dropped bindings",
        )
    )
    link_basis = _story_links(row, local)
    connectivity = _exact_dict(
        local.get("local_to_global_connectivity"), "connectivity ledger"
    )
    _require(
        connectivity.get("format") == CONNECTIVITY_FORMAT
        and connectivity.get("gold_loaded") is False
        and connectivity.get("provider_prompt_count") == 0
        and connectivity.get("retained_transformer_token_state_bytes") == 0,
        "connectivity envelope changed",
    )
    connectivity_unsigned = dict(connectivity)
    connectivity_receipt = require_sha256(
        connectivity_unsigned.pop("receipt_sha256", None), "connectivity receipt"
    )
    _require(
        identity_sha256(connectivity_unsigned) == connectivity_receipt,
        "connectivity ledger receipt changed",
    )
    validation = _exact_dict(row.get("validation_contract"), "validation contract")
    used_handles: set[str] = set()
    for key in ("deterministic_execution_advisory", "scalar_validation_advisory"):
        advisory = validation.get(key)
        if advisory is not None:
            used_handles.update(
                _ordered_text(
                    _exact_dict(advisory, f"{key} value").get("used_handle_ids"),
                    f"{key} handles",
                )
            )
    records: list[EvidenceRecord] = []
    all_post_dedup = set(lane_selected) | set(lane_omitted)
    for value in _exact_list(connectivity.get("rows"), "connectivity rows"):
        item = _exact_dict(value, "connectivity row")
        _verify_identity_field(item, "row_receipt_sha256", "connectivity row")
        binding = require_sha256(
            item.get("binding_receipt_sha256"), "connectivity binding"
        )
        locator = require_sha256(
            item.get("local_source_locator_sha256"), "connectivity locator"
        )
        sources = source_by_locator.get(locator)
        _require(bool(sources), "connectivity row cannot resolve an exact local source")
        mechanism = require_text(item.get("mechanism_id"), "connectivity mechanism")
        lane = mechanism_lane.get(mechanism)
        admitted_items = fair_items_by_mechanism.get(mechanism)
        _require(
            lane is not None and admitted_items is not None,
            "connectivity mechanism lost its lane or fair partition",
        )
        post_dedup = binding not in dedup_excluded
        _require(
            (post_dedup and binding in all_post_dedup)
            or (not post_dedup and binding not in all_post_dedup),
            "dedup/lane binding lifecycle changed",
        )
        globally_bound = item.get("globally_bound") is True
        handle = require_text(item.get("handle_id"), "connectivity handle")
        item_receipt = require_sha256(
            item.get("item_receipt_sha256"), "connectivity item receipt"
        )
        selected_by_lane = binding in lane_selected
        retained_by_surplus_fill = binding in post_surplus_selected
        fair_retained = bool(
            post_dedup
            and retained_by_surplus_fill
            and item_receipt in admitted_items
            and binding not in fair_dropped_bindings
        )
        final_fit_retained = bool(
            fair_retained and retained_bindings.get(binding) == handle
        )
        _require(
            (not globally_bound) or final_fit_retained,
            "globally bound connectivity row lost fitted provenance",
        )
        declared_advisory = handle in used_handles
        _require(
            item.get("advisory_consumed") is declared_advisory,
            "connectivity advisory flag changed",
        )
        story_ids = frozenset(
            _ordered_text(item.get("story_link_ids"), "connectivity story links")
        )
        _require(
            item.get("retrieved_local") is True
            and story_ids <= set(link_basis),
            "connectivity row is not retrieved local or cites a missing story link",
        )
        records.append(
            EvidenceRecord(
                handle,
                binding,
                locator,
                mechanism,
                lane,
                require_text(item.get("source_group_handle"), "connectivity group"),
                sources or frozenset(),
                frozenset(
                    _ordered_text(item.get("discourse_slot_ids"), "discourse slots")
                ),
                story_ids,
                post_dedup,
                selected_by_lane,
                retained_by_surplus_fill,
                fair_retained,
                final_fit_retained,
                globally_bound,
                bool(globally_bound and declared_advisory),
            )
        )
    fair_binding_receipts = {
        record.binding_receipt_sha256
        for record in records
        if record.fair_merge_retained
    }
    _require(
        fair_dropped_bindings == post_surplus_selected - fair_binding_receipts,
        "fair merge post-surplus binding partition changed",
    )
    _require(
        final_fit_dropped_bindings
        == fair_binding_receipts - set(retained_bindings),
        "final hard-fit binding partition changed",
    )
    _require(
        connectivity.get("retrieved_local_count") == len(records)
        and connectivity.get("globally_bound_count")
        == sum(record.globally_bound for record in records),
        "connectivity aggregate changed",
    )
    _require(
        connectivity.get("operator_consumed_count")
        == sum(
            _exact_dict(value, "connectivity row").get("operator_consumed") is True
            for value in connectivity["rows"]
        ),
        "connectivity aggregate changed",
    )
    required_slots = frozenset(
        _ordered_text(validation.get("required_slot_ids"), "required slot IDs")
    )
    if answerer_used_handles is not None:
        _require(
            answerer_used_handles
            <= {record.handle_id for record in records if record.globally_bound},
            "answerer cited a handle that was not globally bound",
        )
    return QuestionLifecycle(
        tuple(records), required_slots, answerer_used_handles, link_basis
    )


def _records_at_stage(
    lifecycle: QuestionLifecycle, stage: str
) -> tuple[EvidenceRecord, ...]:
    if stage == "retrieved_selected":
        return lifecycle.records
    if stage == "post_dedup_retained":
        return tuple(row for row in lifecycle.records if row.post_dedup_retained)
    if stage == "lane_selected":
        return tuple(row for row in lifecycle.records if row.lane_selected)
    if stage == "surplus_fill_retained":
        return tuple(
            row for row in lifecycle.records if row.surplus_fill_retained
        )
    if stage == "fair_merge_retained":
        return tuple(row for row in lifecycle.records if row.fair_merge_retained)
    if stage == "final_fit_retained":
        return tuple(row for row in lifecycle.records if row.final_fit_retained)
    if stage == "globally_bound":
        return tuple(row for row in lifecycle.records if row.globally_bound)
    if stage == "operator_consumed":
        return tuple(row for row in lifecycle.records if row.operator_consumed)
    if stage == ANSWERER_STAGE:
        _require(
            lifecycle.answerer_used_handle_ids is not None,
            "answerer stage is not evaluated",
        )
        return tuple(
            row
            for row in lifecycle.records
            if row.globally_bound
            and row.handle_id in lifecycle.answerer_used_handle_ids
        )
    raise TypedFinalTargetCoverageError(f"unknown lifecycle stage: {stage}")


def _expected_sources(target: Mapping[str, Any]) -> tuple[str, ...]:
    if target.get("target_kind") == "source_id":
        return (require_text(target.get("target_id"), "source target ID"),)
    basis = _exact_dict(target.get("assignment_basis"), "target assignment basis")
    return _ordered_text(basis.get("expected_source_ids"), "expected source IDs")


def _target_stage(
    target: Mapping[str, Any],
    lifecycle: QuestionLifecycle,
    stage: str,
) -> dict[str, Any]:
    question_id = require_text(target.get("question_id"), "target question ID")
    expected = _expected_sources(target)
    records = _records_at_stage(lifecycle, stage)
    matching = tuple(
        row
        for row in records
        if set(expected) & _source_aliases(row.source_ids, question_id)
    )
    reached = set().union(
        *(_source_aliases(row.source_ids, question_id) for row in matching)
    ) if matching else set()
    operands_complete = set(expected) <= reached
    target_kind = target.get("target_kind")
    link_ids: list[str] = []
    link_basis: list[str] = []
    slots = set().union(*(row.supported_slot_ids for row in matching)) if matching else set()
    slots_complete = bool(lifecycle.required_slot_ids) and (
        lifecycle.required_slot_ids <= slots
    )

    if target_kind == "source_id":
        hit = operands_complete
    elif target_kind == "coverage_check":
        # Deliberately ignore frontier.mode/closed and scan row counts.  Only
        # explicit source witnesses with declared typed slots can satisfy this.
        hit = operands_complete and slots_complete
    elif target_kind == "relation":
        for link_id in sorted(lifecycle.story_link_basis_by_id):
            linked = tuple(row for row in matching if link_id in row.story_link_ids)
            linked_sources = set().union(
                *(_source_aliases(row.source_ids, question_id) for row in linked)
            ) if linked else set()
            if set(expected) <= linked_sources:
                link_ids.append(link_id)
                link_basis.append(lifecycle.story_link_basis_by_id[link_id])
        operator_link = tuple(row for row in matching if row.operator_consumed)
        operator_sources = set().union(
            *(_source_aliases(row.source_ids, question_id) for row in operator_link)
        ) if operator_link else set()
        if set(expected) <= operator_sources:
            link_ids.append("deterministic_operator_used_handles")
            link_basis.append("deterministic_or_scalar_operator_link")
        # Links are produced only at the global prompt boundary.  Earlier
        # source reach is exposed as operand completeness, not relation credit.
        link_available = bool(link_ids) and stage in {
            "globally_bound",
            "operator_consumed",
            ANSWERER_STAGE,
        }
        hit = operands_complete and link_available
    else:
        raise TypedFinalTargetCoverageError("unknown target kind")

    return {
        "hit": bool(hit),
        "operand_sources_complete": operands_complete,
        "expected_source_ids_reached": [
            source_id for source_id in expected if source_id in reached
        ],
        "declared_required_slot_ids": sorted(lifecycle.required_slot_ids),
        "declared_slot_ids_reached": sorted(slots & lifecycle.required_slot_ids),
        "declared_slots_complete": slots_complete,
        "exhaustive_physical_scan_absence_inference_used": False,
        "surviving_link_ids": list(dict.fromkeys(link_ids)),
        "surviving_link_bases": list(dict.fromkeys(link_basis)),
        "mechanism_ids": sorted({row.mechanism_id for row in matching}),
        "lane_ids": sorted({row.lane_id for row in matching}),
        "witness_handle_ids": sorted({row.handle_id for row in matching}),
    }


def _failure_transition(
    target: Mapping[str, Any], stages: Mapping[str, Mapping[str, Any]]
) -> str:
    if not stages["retrieved_selected"]["operand_sources_complete"]:
        return "missing_at_retrieval_selection"
    if not stages["post_dedup_retained"]["operand_sources_complete"]:
        return "lost_at_post_selection_dedup"
    recovered_by_surplus = bool(
        not stages["lane_selected"]["operand_sources_complete"]
        and stages["surplus_fill_retained"]["operand_sources_complete"]
    )
    if not stages["surplus_fill_retained"]["operand_sources_complete"]:
        return "lost_at_non_borrowable_lane_selection"
    if not stages["fair_merge_retained"]["operand_sources_complete"]:
        return "lost_at_fair_merge"
    if not stages["final_fit_retained"]["operand_sources_complete"]:
        return "lost_at_hard_prompt_fit"
    if not stages["globally_bound"]["operand_sources_complete"]:
        return "lost_at_provenance_or_semantic_global_binding"
    if target["target_kind"] == "coverage_check" and not stages[
        "globally_bound"
    ]["declared_slots_complete"]:
        return "declared_coverage_witness_or_slot_missing"
    if target["target_kind"] == "relation" and not stages["globally_bound"][
        "hit"
    ]:
        return "surviving_story_or_operator_link_missing"
    if recovered_by_surplus:
        return "recovered_by_global_surplus_fill"
    if not stages["operator_consumed"]["hit"]:
        return "not_consumed_by_deterministic_operator"
    return "consumed_by_deterministic_operator"


def build_analysis_payload(
    *,
    runtime: VerifiedTypedRuntime,
    baseline: VerifiedBaseline,
    plan: Mapping[str, Any],
    target_plan_file_sha256: str,
) -> dict[str, Any]:
    """Join verified runtime rows to post-hoc labels for the exact 28 misses."""

    _require(
        baseline.miss_ordinals == BASELINE_MISS_ORDINALS,
        "analysis baseline miss set changed",
    )
    lifecycle_by_ordinal = {
        ordinal: _question_lifecycle(
            row,
            answerer_used_handles=(
                None
                if runtime.answer_used_handles is None
                else runtime.answer_used_handles[ordinal]
            ),
        )
        for ordinal, row in enumerate(runtime.composition_rows)
        if ordinal in set(BASELINE_MISS_ORDINALS)
    }
    desired = _exact_list(plan.get("desired_targets"), "desired targets")
    targets = [
        _exact_dict(value, "desired target")
        for value in desired
        if _exact_dict(value, "desired target").get("ordinal")
        in set(BASELINE_MISS_ORDINALS)
    ]
    _require(
        {int(row["ordinal"]) for row in targets} == set(BASELINE_MISS_ORDINALS)
        and all(int(row["ordinal"]) in BASELINE_MISS_ORDINALS for row in targets),
        "post-hoc target projection escaped the exact 28 baseline misses",
    )

    target_rows: list[dict[str, Any]] = []
    for target in targets:
        ordinal = int(target["ordinal"])
        lifecycle = lifecycle_by_ordinal[ordinal]
        stage_rows = {
            stage: _target_stage(target, lifecycle, stage) for stage in STAGES
        }
        answerer = (
            None
            if runtime.answer_used_handles is None
            else _target_stage(target, lifecycle, ANSWERER_STAGE)
        )
        target_rows.append(
            {
                "ordinal": ordinal,
                "question_id": target["question_id"],
                "target_id": target["target_id"],
                "target_kind": target["target_kind"],
                "target_sha256": target["target_sha256"],
                "primary_owner": target["primary_owner"],
                "expected_source_ids": list(_expected_sources(target)),
                "stages": stage_rows,
                ANSWERER_STAGE: (
                    {"status": "not_evaluated", "result": None}
                    if answerer is None
                    else {"status": "evaluated", "result": answerer}
                ),
                "failure_transition": _failure_transition(target, stage_rows),
            }
        )

    by_stage = {
        stage: {
            "hit_count": sum(row["stages"][stage]["hit"] for row in target_rows),
            "target_count": len(target_rows),
            "by_target_kind": {
                kind: {
                    "hit_count": sum(
                        row["stages"][stage]["hit"]
                        for row in target_rows
                        if row["target_kind"] == kind
                    ),
                    "target_count": sum(
                        row["target_kind"] == kind for row in target_rows
                    ),
                }
                for kind in ("source_id", "relation", "coverage_check")
            },
        }
        for stage in STAGES
    }
    mechanisms = sorted(
        {
            mechanism
            for row in target_rows
            for stage in STAGES
            for mechanism in row["stages"][stage]["mechanism_ids"]
        }
    )
    lanes = sorted(
        {
            lane
            for row in target_rows
            for stage in STAGES
            for lane in row["stages"][stage]["lane_ids"]
        }
    )

    def nonexclusive_breakdown(key: str, values: Sequence[str]) -> dict[str, Any]:
        return {
            value: {
                stage: sum(
                    row["stages"][stage]["hit"]
                    and value in row["stages"][stage][key]
                    for row in target_rows
                )
                for stage in STAGES
            }
            for value in values
        }

    payload: dict[str, Any] = {
        "format": FORMAT,
        "runtime_use_forbidden": True,
        "gold_target_tags_posthoc_only": True,
        "provider_calls": 0,
        "runtime_artifacts_verified_before_posthoc_inputs_load": True,
        "posthoc_load_order": [
            "typed_composition",
            "typed_preflight",
            *(("typed_answer_run",) if runtime.answer_run is not None else ()),
            "baseline_judge_and_score",
            "target_owner_plan",
        ],
        "baseline_accuracy": 0.72,
        "baseline_miss_count": len(BASELINE_MISS_ORDINALS),
        "baseline_miss_ordinals": list(BASELINE_MISS_ORDINALS),
        "question_count": EXPECTED_QUESTION_COUNT,
        "analyzed_question_count": len(BASELINE_MISS_ORDINALS),
        "analyzed_target_count": len(target_rows),
        "stages": {
            "retrieved_selected": (
                "mechanism-selected exact local bindings before cross-method dedup"
            ),
            "post_dedup_retained": (
                "selected bindings retained after exact post-selection dedup; lane "
                "packing has not yet been credited"
            ),
            "lane_selected": (
                "post-dedup bindings admitted by their sealed non-borrowable "
                "per-method lane allowance"
            ),
            "surplus_fill_retained": (
                "all non-borrowable lane minima plus usable omitted bindings "
                "admitted within the sum of active lane allowances"
            ),
            "fair_merge_retained": (
                "post-surplus item and binding retained by the sealed fair-premerge "
                "item and binding partitions"
            ),
            "final_fit_retained": (
                "fair-merge binding retained after exact complete-chat hard-prompt "
                "fitting, before semantic/provenance connectivity credit"
            ),
            "globally_bound": (
                "exact provenance, group, semantic, temporal and discourse bindings "
                "surviving in the final LLM context"
            ),
            "operator_consumed": (
                "globally bound handles explicitly named by the sealed deterministic "
                "execution or bounded scalar-validation advisory"
            ),
            ANSWERER_STAGE: (
                "optional materialized-run handles cited by a validated replacement; "
                "not evaluated without a sealed typed answer run"
            ),
        },
        "coverage_check_policy": {
            "requires_expected_source_witnesses": True,
            "requires_declared_typed_slots": True,
            "exhaustive_physical_scan_implies_semantic_absence": False,
        },
        "relation_policy": {
            "requires_all_expected_source_operands": True,
            "requires_surviving_story_or_operator_link": True,
        },
        "bindings": {
            "typed_composition_sha256": runtime.composition.sha256,
            "typed_preflight_sha256": runtime.preflight.sha256,
            "typed_answer_run_sha256": (
                None if runtime.answer_run is None else runtime.answer_run.sha256
            ),
            "baseline_judge_sha256": baseline.judge.sha256,
            "baseline_score_sha256": baseline.score.sha256,
            "target_plan_file_sha256": target_plan_file_sha256,
            "target_plan_identity_sha256": plan["plan_sha256"],
        },
        "stage_summary": by_stage,
        "nonexclusive_target_hits_by_mechanism": nonexclusive_breakdown(
            "mechanism_ids", mechanisms
        ),
        "nonexclusive_target_hits_by_lane": nonexclusive_breakdown(
            "lane_ids", lanes
        ),
        "failure_transition_counts": dict(
            sorted(Counter(row["failure_transition"] for row in target_rows).items())
        ),
        "source_target_failure_transition_counts": dict(
            sorted(
                Counter(
                    row["failure_transition"]
                    for row in target_rows
                    if row["target_kind"] == "source_id"
                ).items()
            )
        ),
        "answerer_cited_status": (
            "not_evaluated"
            if runtime.answer_run is None
            else "evaluated_from_sealed_typed_answer_run"
        ),
        "targets": target_rows,
    }
    payload["analysis_sha256"] = identity_sha256(payload)
    return payload


def analyze_paths(
    *,
    typed_root: Path,
    expected_composition_sha256: str,
    expected_preflight_sha256: str,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    target_plan_path: Path = DEFAULT_TARGET_PLAN,
    answer_run_path: Path | None = None,
    expected_answer_run_sha256: str | None = None,
) -> dict[str, Any]:
    runtime = verify_typed_runtime(
        typed_root=typed_root,
        expected_composition_sha256=expected_composition_sha256,
        expected_preflight_sha256=expected_preflight_sha256,
        answer_run_path=answer_run_path,
        expected_answer_run_sha256=expected_answer_run_sha256,
    )
    # Gold-bearing data is intentionally not opened above this line.
    baseline = _verify_baseline(baseline_root, runtime.composition_rows)
    plan, plan_file_sha = _load_target_plan(
        target_plan_path, runtime.composition_rows
    )
    return build_analysis_payload(
        runtime=runtime,
        baseline=baseline,
        plan=plan,
        target_plan_file_sha256=plan_file_sha,
    )


def run_analysis(
    *,
    output_path: Path,
    **kwargs: Any,
) -> tuple[SealedArtifact, bool]:
    return publish_sealed_json(output_path, analyze_paths(**kwargs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-root", type=Path, default=DEFAULT_TYPED_ROOT)
    parser.add_argument("--expected-composition-sha256", required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument("--typed-answer-run", type=Path)
    parser.add_argument("--expected-typed-answer-run-sha256")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact, created = run_analysis(
        typed_root=args.typed_root,
        expected_composition_sha256=args.expected_composition_sha256,
        expected_preflight_sha256=args.expected_preflight_sha256,
        baseline_root=args.baseline_root,
        target_plan_path=args.target_plan,
        answer_run_path=args.typed_answer_run,
        expected_answer_run_sha256=args.expected_typed_answer_run_sha256,
        output_path=args.output,
    )
    summary = artifact.payload["stage_summary"]
    print(
        f"Typed-final target coverage {artifact.sha256} "
        f"({'created' if created else 'reused'}): "
        f"global={summary['globally_bound']['hit_count']}/"
        f"{summary['globally_bound']['target_count']}; "
        f"operator={summary['operator_consumed']['hit_count']}/"
        f"{summary['operator_consumed']['target_count']}; provider_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANSWERER_STAGE",
    "BASELINE_MISS_ORDINALS",
    "EXPECTED_BASELINE_JUDGE_SHA256",
    "EXPECTED_BASELINE_SCORE_SHA256",
    "FORMAT",
    "OUTPUT_NAME",
    "PINNED_TARGET_PLAN_FILE_SHA256",
    "PINNED_TARGET_PLAN_IDENTITY_SHA256",
    "STAGES",
    "TypedFinalTargetCoverageError",
    "VerifiedBaseline",
    "VerifiedTypedRuntime",
    "analyze_paths",
    "build_analysis_payload",
    "build_parser",
    "main",
    "run_analysis",
    "verify_typed_runtime",
]
