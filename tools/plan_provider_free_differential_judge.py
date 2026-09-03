#!/usr/bin/env python3
"""Plan and merge provider-free differential Sol judging.

The planner consumes a replay-verified policy-v5 answer and one or more
authenticated prior binary-Sol judge runs.  A prior judgment is reusable only
when question, reference, prediction, prompt contract, and every available
model identity agree.  Only genuinely novel predictions receive prompt rows.

This module never calls a provider.  Its ``merge`` command refuses to produce
a score until authenticated judgments cover every novel prompt.  Judge gold
is confined to the distinct judge-plan/merge artifacts and never flows back
into the answer-policy artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval._binary_judge_protocol import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    build_judge_prompt,
)
from tools import (  # noqa: E402
    revalidate_locked_semantic_global_terminal_full100_policy_v5 as policy_cli,
)
from tools.matched_eval import judging  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    judge_row_projection,
)


FORMAT = "memory-condense-provider-free-differential-sol-judge-v1"
PLAN_FORMAT = f"{FORMAT}-plan-v1"
TARGET_ROW_FORMAT = f"{FORMAT}-target-row-v1"
REUSED_ROW_FORMAT = f"{FORMAT}-reused-judgment-v1"
NOVEL_PROMPT_FORMAT = f"{FORMAT}-novel-prompt-v1"
MERGE_FORMAT = f"{FORMAT}-merge-v1"
MERGED_ROW_FORMAT = f"{FORMAT}-merged-row-v1"
PRIOR_BINDING_FORMAT = f"{FORMAT}-prior-binding-v1"

PLAN_NAME = "policy-v5-differential-sol-judge-plan-v1.json"
MERGE_NAME = "policy-v5-differential-sol-judge-merge-v1.json"
DEFAULT_OUTPUT_ROOT = policy_cli.DEFAULT_OUTPUT_ROOT / "differential-sol-judge-v1"
DEFAULT_JUDGE_MODEL = judging.DEFAULT_SOL_GATEWAY_MODEL

_LEGACY_MESSAGE_QUESTION_PREFLIGHT_FORMATS = frozenset(
    {
        "memory-condense-locked-specialist-final-reconciliation-sol-judge-preflight-v3",
    }
)
QUESTION_COUNT = policy_cli.QUESTION_COUNT
ALL_ORDINALS = tuple(range(QUESTION_COUNT))

JUDGE_CONTRACT_SHA256 = identity_sha256(
    {
        "binary_verdict_protocol": "leading-CORRECT-or-INCORRECT-v1",
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_template": JUDGE_USER_TEMPLATE,
    }
)


class DifferentialJudgePlannerError(MatchedEvalContractError):
    """A target, prior judgment, prompt plan, or merge binding changed."""


class DifferentialJudgeIncompleteError(DifferentialJudgePlannerError):
    """Novel predictions do not yet have authenticated judgments."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise DifferentialJudgePlannerError(message)


def _self_hash(
    raw: Mapping[str, Any], *, key: str, label: str, required: bool = True
) -> tuple[dict[str, Any], str]:
    _require(type(raw) is dict, f"{label} changed type")
    row = dict(raw)
    declared = row.pop(key, None)
    if declared is None and not required:
        digest = identity_sha256(row)
        row[key] = digest
        return row, digest
    digest = require_sha256(declared, label)
    _require(identity_sha256(row) == digest, f"{label} receipt changed")
    row[key] = digest
    return row, digest


def _messages(raw: object, label: str) -> tuple[dict[str, str], ...]:
    _require(type(raw) is list and bool(raw), f"{label} changed type")
    output: list[dict[str, str]] = []
    for value in raw:
        _require(
            type(value) is dict
            and set(value) == {"role", "content"}
            and value.get("role") in {"system", "user", "assistant"}
            and type(value.get("content")) is str,
            f"{label} changed message schema",
        )
        output.append({"role": value["role"], "content": value["content"]})
    return tuple(output)


def _artifact(
    path: str | Path, expected_sha256: str, label: str
) -> SealedArtifact:
    value = read_sealed_json(path)
    _require(
        value.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    return value


@dataclass(frozen=True, slots=True)
class AuthenticatedJudgeRun:
    """Normalized seam from a sealed prior Sol preflight/run/replay."""

    preflight: SealedArtifact
    judge: SealedArtifact
    replay: SealedArtifact
    model: str | None
    entries: tuple[dict[str, Any], ...]
    binding_sha256: str

    def projection(self) -> dict[str, Any]:
        return {
            "binding_sha256": self.binding_sha256,
            "format": PRIOR_BINDING_FORMAT,
            "judge_artifact_sha256": self.judge.sha256,
            "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
            "judge_model": self.model,
            "judge_replay_artifact_sha256": self.replay.sha256,
            "preflight_artifact_sha256": self.preflight.sha256,
            "question_count": len(self.entries),
        }


def _available_model(
    preflight: Mapping[str, Any], judge: Mapping[str, Any]
) -> str | None:
    value = preflight.get("model")
    if value is not None:
        return require_text(value, "prior judge model")
    value = judge.get("judge_model")
    if value is not None:
        return require_text(value, "prior judge model")
    batch = judge.get("completion_batch")
    if isinstance(batch, Mapping):
        provenance = batch.get("provenance")
        if isinstance(provenance, Mapping) and provenance.get("model") is not None:
            return require_text(provenance.get("model"), "prior judge model")
    return None


def _question_from_exact_legacy_message(
    *,
    preflight_format: object,
    messages: tuple[dict[str, Any], ...],
    reference: str,
    prediction: str,
) -> str:
    """Recover the omitted plaintext question from one frozen prompt schema.

    The historical V3 preflight stored the canonical messages and all hashes,
    but not a duplicate top-level ``question`` field.  Accept only that exact
    versioned format and exact current judge template; this is normalization,
    not a permissive parser for arbitrary legacy prompts.
    """

    _require(
        preflight_format in _LEGACY_MESSAGE_QUESTION_PREFLIGHT_FORMATS,
        "prior judge question is absent outside an approved legacy schema",
    )
    _require(
        len(messages) == 2
        and messages[0].get("role") == "system"
        and messages[1].get("role") == "user",
        "legacy prior judge messages changed shape",
    )
    prefix = "Question: "
    suffix = (
        f"\nGold answer: {reference}\nPredicted answer: {prediction}"
        "\n\nIs the predicted answer correct? Reply CORRECT or INCORRECT, "
        "then a one-sentence reason."
    )
    content = require_text(messages[1].get("content"), "legacy judge user message")
    _require(
        content.startswith(prefix)
        and content.endswith(suffix)
        and len(content) > len(prefix) + len(suffix),
        "legacy prior judge message changed canonical template",
    )
    return require_text(
        content[len(prefix) : -len(suffix)], "legacy prior judge question"
    )


def authenticate_prior_judge_run(
    preflight: SealedArtifact,
    judge: SealedArtifact,
    replay: SealedArtifact,
) -> AuthenticatedJudgeRun:
    """Authenticate and normalize a binary-Sol judge run.

    Callers may obtain the artifacts from a runner-specific verified loader.
    The generic boundary additionally requires a byte-identical judge replay,
    canonical binary-judge prompts, and exact prompt/verdict hash bindings.
    Subset runs are accepted so the same seam can consume later novel-only
    judgments.
    """

    _require(
        type(preflight) is SealedArtifact
        and type(judge) is SealedArtifact
        and type(replay) is SealedArtifact,
        "prior judge artifacts changed type",
    )
    preflight_payload = preflight.payload
    judge_payload = judge.payload
    _require(
        judge.sha256 == replay.sha256
        and judge.payload == replay.payload
        and preflight_payload.get("gold_loaded") is True
        and judge_payload.get("gold_loaded") is True
        and judge_payload.get("preflight_artifact_sha256") == preflight.sha256,
        "prior judge is not a replay-verified gold-bearing run",
    )
    for payload, label in (
        (preflight_payload, "prior preflight"),
        (judge_payload, "prior judge"),
    ):
        explicit_contract = payload.get("judge_contract_sha256")
        if explicit_contract is not None:
            _require(
                require_sha256(explicit_contract, f"{label} contract")
                == JUDGE_CONTRACT_SHA256,
                f"{label} uses another judge contract",
            )

    prompt_rows = preflight_payload.get("prompt_rows")
    judgment_rows = judge_payload.get("questions")
    _require(
        type(prompt_rows) is list
        and type(judgment_rows) is list
        and 0 < len(prompt_rows) == len(judgment_rows) <= QUESTION_COUNT,
        "prior judge population changed",
    )
    prompts_by_ordinal: dict[int, tuple[dict[str, Any], str]] = {}
    for raw in prompt_rows:
        _require(type(raw) is dict, "prior prompt row changed type")
        prompt, prompt_receipt = _self_hash(
            raw,
            key="prompt_row_receipt_sha256",
            label="prior prompt row",
        )
        ordinal = prompt.get("ordinal")
        reference = require_text(prompt.get("reference"), "prior judge reference")
        prediction = require_text(prompt.get("prediction"), "prior judge prediction")
        messages = _messages(prompt.get("messages"), "prior judge prompt messages")
        raw_question = prompt.get("question")
        question = (
            require_text(raw_question, "prior judge question")
            if raw_question is not None
            else _question_from_exact_legacy_message(
                preflight_format=preflight_payload.get("format"),
                messages=messages,
                reference=reference,
                prediction=prediction,
            )
        )
        expected = tuple(
            dict(row) for row in build_judge_prompt(question, reference, prediction)
        )
        _require(
            type(ordinal) is int
            and 0 <= ordinal < QUESTION_COUNT
            and ordinal not in prompts_by_ordinal
            and prompt.get("question_sha256") == quote_sha256(question)
            and prompt.get("reference_sha256") == quote_sha256(reference)
            and prompt.get("prediction_sha256") == quote_sha256(prediction)
            and messages == expected
            and prompt.get("messages_sha256") == identity_sha256(list(expected)),
            f"prior judge prompt {ordinal} changed contract or hashes",
        )
        prompts_by_ordinal[ordinal] = (
            {**prompt, "question": question},
            prompt_receipt,
        )

    normalized: list[dict[str, Any]] = []
    for raw in judgment_rows:
        _require(type(raw) is dict, "prior judgment row changed type")
        judgment, judgment_receipt = _self_hash(
            raw,
            key="judge_row_sha256",
            label="prior judgment row",
        )
        ordinal = judgment.get("ordinal")
        prompt_pair = prompts_by_ordinal.get(ordinal)
        _require(prompt_pair is not None, "prior judgment has no sealed prompt")
        assert prompt_pair is not None
        prompt, prompt_receipt = prompt_pair
        correct = judgment.get("correct")
        output = judgment.get("judge_output")
        _require(
            type(correct) is bool
            and judgment.get("question_id") == prompt.get("question_id")
            and judgment.get("question_sha256") == prompt.get("question_sha256")
            and judgment.get("reference_sha256") == prompt.get("reference_sha256")
            and judgment.get("prediction_sha256") == prompt.get("prediction_sha256")
            and judgment.get("messages_sha256") == prompt.get("messages_sha256"),
            f"prior judgment {ordinal} differs from its prompt",
        )
        if output is not None:
            try:
                parsed = parse_binary_judge_verdict(
                    require_text(output, "prior judge output")
                )
            except RuntimeError as exc:
                raise DifferentialJudgePlannerError(
                    f"prior judgment {ordinal} has malformed output"
                ) from exc
            _require(
                parsed is correct
                and judgment.get("judge_output_sha256") == quote_sha256(output),
                f"prior judgment {ordinal} output changed verdict",
            )
        normalized.append(
            {
                "correct": correct,
                "judge_row_sha256": judgment_receipt,
                "messages_sha256": prompt["messages_sha256"],
                "ordinal": ordinal,
                "prediction": prompt["prediction"],
                "prediction_sha256": prompt["prediction_sha256"],
                "prompt_row_receipt_sha256": prompt_receipt,
                "question": prompt["question"],
                "question_id": prompt["question_id"],
                "question_sha256": prompt["question_sha256"],
                "reference": prompt["reference"],
                "reference_sha256": prompt["reference_sha256"],
            }
        )
    normalized.sort(key=lambda row: int(row["ordinal"]))
    _require(
        tuple(row["ordinal"] for row in normalized)
        == tuple(sorted(prompts_by_ordinal)),
        "prior prompt/judgment population differs",
    )
    model = _available_model(preflight_payload, judge_payload)
    binding_body = {
        "judge_artifact_sha256": judge.sha256,
        "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
        "judge_model": model,
        "judge_replay_artifact_sha256": replay.sha256,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": len(normalized),
        "row_population_sha256": identity_sha256(
            [row["judge_row_sha256"] for row in normalized]
        ),
    }
    return AuthenticatedJudgeRun(
        preflight=preflight,
        judge=judge,
        replay=replay,
        model=model,
        entries=tuple(normalized),
        binding_sha256=identity_sha256(binding_body),
    )


def load_authenticated_prior_judge_run(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    judge_path: str | Path,
    expected_judge_sha256: str,
    replay_path: str | Path,
    expected_replay_sha256: str,
) -> AuthenticatedJudgeRun:
    preflight = _artifact(
        preflight_path, expected_preflight_sha256, "prior judge preflight"
    )
    judge = _artifact(judge_path, expected_judge_sha256, "prior judge run")
    replay = _artifact(replay_path, expected_replay_sha256, "prior judge replay")
    return authenticate_prior_judge_run(preflight, judge, replay)


def validate_verified_policy_v5_source(
    run: SealedArtifact, replay: SealedArtifact
) -> tuple[dict[str, Any], ...]:
    """Validate the gold-free 100-row policy seam and its sealed replay."""

    payload = run.payload
    questions = payload.get("questions")
    judge_rows = payload.get("judge_rows")
    _require(
        type(run) is SealedArtifact
        and type(replay) is SealedArtifact
        and payload.get("format") == policy_cli.RUN_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_revalidation") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and type(questions) is list
        and type(judge_rows) is list
        and len(questions) == len(judge_rows) == QUESTION_COUNT
        and replay.payload.get("format") == policy_cli.REPLAY_FORMAT
        and replay.payload.get("byte_identical") is True
        and replay.payload.get("gold_loaded") is False
        and replay.payload.get("physical_provider_calls") == 0
        and replay.payload.get("expected_run_sha256") == run.sha256
        and replay.payload.get("replayed_run_sha256") == run.sha256,
        "policy-v5 source is not replay-verified and gold-free",
    )
    result: list[dict[str, Any]] = []
    for ordinal, (raw, projected) in enumerate(zip(questions, judge_rows, strict=True)):
        _require(
            type(raw) is dict and type(projected) is dict,
            "policy-v5 target row changed type",
        )
        row, source_receipt = _self_hash(
            raw,
            key="source_row_sha256",
            label="policy-v5 target row",
        )
        prediction = require_text(row.get("prediction"), "policy-v5 prediction")
        _require(
            row.get("ordinal") == ordinal
            and row.get("prediction_sha256") == quote_sha256(prediction)
            and judge_row_projection(row) == projected,
            f"policy-v5 target row {ordinal} changed",
        )
        result.append(
            {
                "ordinal": ordinal,
                "prediction": prediction,
                "prediction_sha256": row["prediction_sha256"],
                "question_id": require_text(
                    row.get("question_id"), "policy-v5 question ID"
                ),
                "question_sha256": require_sha256(
                    row.get("question_sha256"), "policy-v5 question"
                ),
                "source_row_sha256": source_receipt,
            }
        )
    _require(
        len({row["question_id"] for row in result}) == QUESTION_COUNT,
        "policy-v5 question identities repeat",
    )
    return tuple(result)


def load_verified_policy_v5_source(
    output_root: str | Path,
    *,
    expected_run_sha256: str,
    expected_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    run = _artifact(
        root / policy_cli.RUN_NAME, expected_run_sha256, "policy-v5 run"
    )
    replay = _artifact(
        root / policy_cli.REPLAY_NAME, expected_replay_sha256, "policy-v5 replay"
    )
    return run, replay, validate_verified_policy_v5_source(run, replay)


def _reference_catalog(
    targets: Sequence[Mapping[str, Any]],
    priors: Sequence[AuthenticatedJudgeRun],
) -> dict[int, dict[str, Any]]:
    catalog: dict[int, dict[str, Any]] = {}
    for target in targets:
        ordinal = int(target["ordinal"])
        candidates = [
            row
            for prior in priors
            for row in prior.entries
            if row["ordinal"] == ordinal
            and row["question_id"] == target["question_id"]
            and row["question_sha256"] == target["question_sha256"]
        ]
        _require(candidates, f"no authenticated reference for target {ordinal}")
        question_hashes = {quote_sha256(str(row["question"])) for row in candidates}
        reference_hashes = {str(row["reference_sha256"]) for row in candidates}
        _require(
            question_hashes == {target["question_sha256"]}
            and len(reference_hashes) == 1
            and len({str(row["reference"]) for row in candidates}) == 1,
            f"authenticated references conflict for target {ordinal}",
        )
        chosen = min(candidates, key=lambda row: str(row["prompt_row_receipt_sha256"]))
        catalog[ordinal] = {
            "question": chosen["question"],
            "reference": chosen["reference"],
            "reference_sha256": chosen["reference_sha256"],
        }
    return catalog


def _reuse_key(
    *,
    question_sha256: str,
    reference_sha256: str,
    prediction_sha256: str,
    judge_model: str,
) -> str:
    return identity_sha256(
        {
            "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
            "judge_model": judge_model,
            "prediction_sha256": prediction_sha256,
            "question_sha256": question_sha256,
            "reference_sha256": reference_sha256,
        }
    )


def _judgment_index(
    priors: Sequence[AuthenticatedJudgeRun], *, judge_model: str
) -> dict[str, tuple[bool, tuple[dict[str, str], ...]]]:
    grouped: dict[str, list[tuple[bool, dict[str, str]]]] = {}
    for prior in priors:
        if prior.model is not None and prior.model != judge_model:
            continue
        for row in prior.entries:
            key = _reuse_key(
                question_sha256=str(row["question_sha256"]),
                reference_sha256=str(row["reference_sha256"]),
                prediction_sha256=str(row["prediction_sha256"]),
                judge_model=judge_model,
            )
            grouped.setdefault(key, []).append(
                (
                    bool(row["correct"]),
                    {
                        "judge_artifact_sha256": prior.judge.sha256,
                        "judge_row_sha256": str(row["judge_row_sha256"]),
                        "prior_binding_sha256": prior.binding_sha256,
                    },
                )
            )
    result: dict[str, tuple[bool, tuple[dict[str, str], ...]]] = {}
    for key, candidates in grouped.items():
        verdicts = {correct for correct, _source in candidates}
        _require(
            len(verdicts) == 1,
            f"conflicting authenticated prior judgments for reuse key {key}",
        )
        sources = tuple(
            sorted(
                {identity_sha256(source): source for _correct, source in candidates}.values(),
                key=identity_sha256,
            )
        )
        result[key] = (next(iter(verdicts)), sources)
    return result


def _reauthenticated_runs(
    runs: Sequence[AuthenticatedJudgeRun], *, label: str
) -> tuple[AuthenticatedJudgeRun, ...]:
    values = tuple(runs)
    _require(
        all(type(run) is AuthenticatedJudgeRun for run in values),
        f"{label} are not authenticated judge runs",
    )
    rebuilt = tuple(
        authenticate_prior_judge_run(run.preflight, run.judge, run.replay)
        for run in values
    )
    _require(rebuilt == values, f"{label} normalized bindings changed")
    return rebuilt


def build_differential_judge_plan(
    policy_run: SealedArtifact,
    policy_replay: SealedArtifact,
    prior_runs: Sequence[AuthenticatedJudgeRun],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, Any]:
    """Build a sealed-ready prompt plan containing only novel predictions."""

    targets = validate_verified_policy_v5_source(policy_run, policy_replay)
    priors = _reauthenticated_runs(prior_runs, label="prior judgment sources")
    _require(bool(priors), "differential planning requires prior judge runs")
    model = require_text(judge_model, "differential judge model")
    catalog = _reference_catalog(targets, priors)
    prior_index = _judgment_index(priors, judge_model=model)
    reused: list[dict[str, Any]] = []
    novel: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    reference_receipts: list[str] = []
    for target in targets:
        ordinal = int(target["ordinal"])
        reference = catalog[ordinal]
        reuse_key = _reuse_key(
            question_sha256=str(target["question_sha256"]),
            reference_sha256=str(reference["reference_sha256"]),
            prediction_sha256=str(target["prediction_sha256"]),
            judge_model=model,
        )
        reused_candidate = prior_index.get(reuse_key)
        resolution = "reused_prior_judgment" if reused_candidate else "novel_prompt"
        target_body = {
            "format": TARGET_ROW_FORMAT,
            "judge_reuse_key_sha256": reuse_key,
            "ordinal": ordinal,
            "prediction_sha256": target["prediction_sha256"],
            "question_id": target["question_id"],
            "question_sha256": target["question_sha256"],
            "reference_sha256": reference["reference_sha256"],
            "resolution": resolution,
            "source_policy_row_sha256": target["source_row_sha256"],
        }
        target_rows.append(
            {**target_body, "target_row_receipt_sha256": identity_sha256(target_body)}
        )
        reference_receipts.append(
            identity_sha256(
                {
                    "ordinal": ordinal,
                    "question_sha256": target["question_sha256"],
                    "reference_sha256": reference["reference_sha256"],
                }
            )
        )
        if reused_candidate is not None:
            correct, sources = reused_candidate
            body = {
                "correct": correct,
                "format": REUSED_ROW_FORMAT,
                "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
                "judge_model": model,
                "judge_reuse_key_sha256": reuse_key,
                "ordinal": ordinal,
                "prediction_sha256": target["prediction_sha256"],
                "question_id": target["question_id"],
                "question_sha256": target["question_sha256"],
                "reference_sha256": reference["reference_sha256"],
                "source_judgments": list(sources),
                "source_policy_row_sha256": target["source_row_sha256"],
            }
            reused.append(
                {**body, "reused_row_receipt_sha256": identity_sha256(body)}
            )
            continue

        messages = tuple(
            dict(row)
            for row in build_judge_prompt(
                str(reference["question"]),
                str(reference["reference"]),
                str(target["prediction"]),
            )
        )
        input_body = {
            "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
            "judge_model": model,
            "ordinal": ordinal,
            "prediction_sha256": target["prediction_sha256"],
            "question_sha256": target["question_sha256"],
            "reference_sha256": reference["reference_sha256"],
        }
        body = {
            "format": NOVEL_PROMPT_FORMAT,
            "judge_input_receipt_sha256": identity_sha256(input_body),
            "judge_reuse_key_sha256": reuse_key,
            "messages": list(messages),
            "messages_sha256": identity_sha256(list(messages)),
            "ordinal": ordinal,
            "prediction": target["prediction"],
            "prediction_sha256": target["prediction_sha256"],
            "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
            "question": reference["question"],
            "question_id": target["question_id"],
            "question_sha256": target["question_sha256"],
            "reference": reference["reference"],
            "reference_sha256": reference["reference_sha256"],
            "source_policy_row_sha256": target["source_row_sha256"],
        }
        novel.append(
            {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
        )

    prior_bindings = sorted(
        (prior.projection() for prior in priors),
        key=lambda value: str(value["binding_sha256"]),
    )
    payload = {
        "answer_policy_gold_loaded": False,
        "caller_ordinal_routing_available": False,
        "format": PLAN_FORMAT,
        "gold_loaded": True,
        "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
        "judge_input_population_sha256": identity_sha256(
            [row["judge_input_receipt_sha256"] for row in novel]
        ),
        "judge_model": model,
        "judge_model_identity_sha256": identity_sha256({"model": model}),
        "merge_ready": not novel,
        "novel_prompt_count": len(novel),
        "novel_prompt_rows": novel,
        "physical_provider_calls_during_planning": 0,
        "prior_judge_bindings": prior_bindings,
        "prior_judge_population_sha256": identity_sha256(prior_bindings),
        "provider_execution_command_available": False,
        "question_count": QUESTION_COUNT,
        "reference_population_sha256": identity_sha256(reference_receipts),
        "reused_judgment_count": len(reused),
        "reused_judgments": reused,
        "score_emitted": False,
        "source_policy_replay_artifact_sha256": policy_replay.sha256,
        "source_policy_run_artifact_sha256": policy_run.sha256,
        "target_population_sha256": identity_sha256(
            [row["target_row_receipt_sha256"] for row in target_rows]
        ),
        "target_rows": target_rows,
    }
    validate_differential_judge_plan(payload)
    return payload


def validate_differential_judge_plan(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate a plan without producing or exposing an aggregate score."""

    _require(type(payload) is dict, "differential judge plan changed type")
    targets = payload.get("target_rows")
    reused = payload.get("reused_judgments")
    novel = payload.get("novel_prompt_rows")
    _require(
        payload.get("format") == PLAN_FORMAT
        and payload.get("gold_loaded") is True
        and payload.get("answer_policy_gold_loaded") is False
        and payload.get("physical_provider_calls_during_planning") == 0
        and payload.get("provider_execution_command_available") is False
        and payload.get("caller_ordinal_routing_available") is False
        and payload.get("score_emitted") is False
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("judge_contract_sha256") == JUDGE_CONTRACT_SHA256
        and type(targets) is list
        and type(reused) is list
        and type(novel) is list
        and len(targets) == QUESTION_COUNT
        and len(reused) + len(novel) == QUESTION_COUNT
        and payload.get("reused_judgment_count") == len(reused)
        and payload.get("novel_prompt_count") == len(novel)
        and payload.get("merge_ready") is (not novel),
        "differential judge plan envelope changed",
    )
    model = require_text(payload.get("judge_model"), "differential judge model")
    prior_bindings = payload.get("prior_judge_bindings")
    _require(
        payload.get("judge_model_identity_sha256")
        == identity_sha256({"model": model})
        and type(prior_bindings) is list
        and bool(prior_bindings)
        and payload.get("prior_judge_population_sha256")
        == identity_sha256(prior_bindings),
        "differential judge model identity changed",
    )
    for binding in prior_bindings:
        _require(
            type(binding) is dict
            and binding.get("format") == PRIOR_BINDING_FORMAT
            and binding.get("judge_contract_sha256") == JUDGE_CONTRACT_SHA256
            and type(binding.get("question_count")) is int
            and 0 < int(binding["question_count"]) <= QUESTION_COUNT,
            "prior judge plan binding changed",
        )
        for key in (
            "binding_sha256",
            "judge_artifact_sha256",
            "judge_replay_artifact_sha256",
            "preflight_artifact_sha256",
        ):
            require_sha256(binding.get(key), f"prior judge binding {key}")
        if binding.get("judge_model") is not None:
            require_text(binding.get("judge_model"), "prior judge binding model")
    require_sha256(
        payload.get("source_policy_run_artifact_sha256"),
        "differential source policy run",
    )
    require_sha256(
        payload.get("source_policy_replay_artifact_sha256"),
        "differential source policy replay",
    )
    target_by_ordinal: dict[int, dict[str, Any]] = {}
    target_receipts: list[str] = []
    for raw in targets:
        row, receipt = _self_hash(
            raw, key="target_row_receipt_sha256", label="differential target row"
        )
        ordinal = row.get("ordinal")
        _require(
            row.get("format") == TARGET_ROW_FORMAT
            and type(ordinal) is int
            and 0 <= ordinal < QUESTION_COUNT
            and ordinal not in target_by_ordinal
            and row.get("resolution") in {"reused_prior_judgment", "novel_prompt"},
            "differential target row changed",
        )
        for key in (
            "judge_reuse_key_sha256",
            "prediction_sha256",
            "question_sha256",
            "reference_sha256",
            "source_policy_row_sha256",
        ):
            require_sha256(row.get(key), f"differential target {key}")
        require_text(row.get("question_id"), "differential target question ID")
        _require(
            row.get("judge_reuse_key_sha256")
            == _reuse_key(
                question_sha256=str(row["question_sha256"]),
                reference_sha256=str(row["reference_sha256"]),
                prediction_sha256=str(row["prediction_sha256"]),
                judge_model=model,
            ),
            "differential target reuse key changed",
        )
        target_by_ordinal[ordinal] = row
        target_receipts.append(receipt)
    _require(
        tuple(row.get("ordinal") for row in targets) == ALL_ORDINALS
        and tuple(sorted(target_by_ordinal)) == ALL_ORDINALS
        and payload.get("target_population_sha256")
        == identity_sha256(target_receipts)
        and payload.get("reference_population_sha256")
        == identity_sha256(
            [
                identity_sha256(
                    {
                        "ordinal": ordinal,
                        "question_sha256": target_by_ordinal[ordinal][
                            "question_sha256"
                        ],
                        "reference_sha256": target_by_ordinal[ordinal][
                            "reference_sha256"
                        ],
                    }
                )
                for ordinal in ALL_ORDINALS
            ]
        ),
        "differential target population changed",
    )

    resolved_ordinals: set[int] = set()
    for raw in reused:
        row, _receipt = _self_hash(
            raw, key="reused_row_receipt_sha256", label="reused judgment row"
        )
        target = target_by_ordinal.get(row.get("ordinal"))
        _require(
            row.get("format") == REUSED_ROW_FORMAT
            and type(row.get("correct")) is bool
            and target is not None
            and target.get("resolution") == "reused_prior_judgment"
            and row.get("judge_reuse_key_sha256")
            == target.get("judge_reuse_key_sha256")
            and row.get("question_sha256") == target.get("question_sha256")
            and row.get("reference_sha256") == target.get("reference_sha256")
            and row.get("prediction_sha256") == target.get("prediction_sha256")
            and row.get("judge_contract_sha256") == JUDGE_CONTRACT_SHA256
            and row.get("judge_model") == model
            and type(row.get("source_judgments")) is list
            and bool(row["source_judgments"]),
            "reused judgment row changed",
        )
        for source in row["source_judgments"]:
            _require(
                type(source) is dict
                and set(source)
                == {
                    "judge_artifact_sha256",
                    "judge_row_sha256",
                    "prior_binding_sha256",
                },
                "reused judgment provenance changed",
            )
            for key in source:
                require_sha256(source[key], f"reused judgment {key}")
        resolved_ordinals.add(int(row["ordinal"]))

    prompt_receipts: list[str] = []
    input_receipts: list[str] = []
    for raw in novel:
        row, prompt_receipt = _self_hash(
            raw, key="prompt_row_receipt_sha256", label="novel judge prompt"
        )
        ordinal = row.get("ordinal")
        target = target_by_ordinal.get(ordinal)
        question = require_text(row.get("question"), "novel judge question")
        reference = require_text(row.get("reference"), "novel judge reference")
        prediction = require_text(row.get("prediction"), "novel judge prediction")
        messages = _messages(row.get("messages"), "novel judge messages")
        expected = tuple(
            dict(value) for value in build_judge_prompt(question, reference, prediction)
        )
        input_body = {
            "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
            "judge_model": model,
            "ordinal": ordinal,
            "prediction_sha256": row.get("prediction_sha256"),
            "question_sha256": row.get("question_sha256"),
            "reference_sha256": row.get("reference_sha256"),
        }
        _require(
            row.get("format") == NOVEL_PROMPT_FORMAT
            and target is not None
            and target.get("resolution") == "novel_prompt"
            and row.get("judge_reuse_key_sha256")
            == target.get("judge_reuse_key_sha256")
            and row.get("question_sha256")
            == target.get("question_sha256")
            == quote_sha256(question)
            and row.get("reference_sha256")
            == target.get("reference_sha256")
            == quote_sha256(reference)
            and row.get("prediction_sha256")
            == target.get("prediction_sha256")
            == quote_sha256(prediction)
            and messages == expected
            and row.get("messages_sha256") == identity_sha256(list(expected))
            and row.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(expected)
            and row.get("judge_input_receipt_sha256")
            == identity_sha256(input_body),
            "novel judge prompt changed",
        )
        resolved_ordinals.add(int(ordinal))
        prompt_receipts.append(prompt_receipt)
        input_receipts.append(str(row["judge_input_receipt_sha256"]))
    _require(
        resolved_ordinals == set(ALL_ORDINALS)
        and payload.get("judge_input_population_sha256")
        == identity_sha256(input_receipts),
        "differential plan resolution population changed",
    )
    return tuple(target_by_ordinal[ordinal] for ordinal in ALL_ORDINALS)


def _merged_row(
    target: Mapping[str, Any],
    *,
    correct: bool,
    source: str,
    source_judgments: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    body = {
        "correct": correct,
        "format": MERGED_ROW_FORMAT,
        "judgment_source": source,
        "ordinal": target["ordinal"],
        "prediction_sha256": target["prediction_sha256"],
        "question_id": target["question_id"],
        "question_sha256": target["question_sha256"],
        "reference_sha256": target["reference_sha256"],
        "source_judgments": [dict(value) for value in source_judgments],
        "source_policy_row_sha256": target["source_policy_row_sha256"],
    }
    return {**body, "merged_row_receipt_sha256": identity_sha256(body)}


def merge_differential_judgments(
    plan: SealedArtifact,
    novel_judge_runs: Sequence[AuthenticatedJudgeRun] = (),
) -> dict[str, Any]:
    """Merge exact prior/new judgments, or fail before scoring if incomplete."""

    targets = validate_differential_judge_plan(plan.payload)
    novel_prompts = {
        int(row["ordinal"]): dict(row)
        for row in plan.payload["novel_prompt_rows"]
    }
    novel_runs = _reauthenticated_runs(
        novel_judge_runs, label="novel judgment sources"
    )
    model = str(plan.payload["judge_model"])
    novel_index = _judgment_index(novel_runs, judge_model=model)
    reused_by_ordinal = {
        int(row["ordinal"]): dict(row)
        for row in plan.payload["reused_judgments"]
    }
    merged: list[dict[str, Any]] = []
    missing: list[int] = []
    for target in targets:
        ordinal = int(target["ordinal"])
        reused = reused_by_ordinal.get(ordinal)
        if reused is not None:
            merged.append(
                _merged_row(
                    target,
                    correct=bool(reused["correct"]),
                    source="reused_prior_judgment",
                    source_judgments=tuple(reused["source_judgments"]),
                )
            )
            continue
        prompt = novel_prompts[ordinal]
        candidate = novel_index.get(str(prompt["judge_reuse_key_sha256"]))
        if candidate is None:
            missing.append(ordinal)
            continue
        correct, sources = candidate
        merged.append(
            _merged_row(
                target,
                correct=correct,
                source="authenticated_novel_judgment",
                source_judgments=sources,
            )
        )
    if missing:
        raise DifferentialJudgeIncompleteError(
            "novel judgments are required before scoring: "
            + ",".join(str(value) for value in missing)
        )
    merged.sort(key=lambda row: int(row["ordinal"]))
    _require(
        len(merged) == QUESTION_COUNT
        and tuple(row["ordinal"] for row in merged) == ALL_ORDINALS,
        "differential merge did not reconstruct 100 rows",
    )
    correct = sum(bool(row["correct"]) for row in merged)
    novel_bindings = sorted(
        (run.projection() for run in novel_runs),
        key=lambda value: str(value["binding_sha256"]),
    )
    payload = {
        "accuracy": correct / QUESTION_COUNT,
        "answer_policy_gold_loaded": False,
        "correct": correct,
        "differential_plan_artifact_sha256": plan.sha256,
        "format": MERGE_FORMAT,
        "gold_loaded": True,
        "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
        "judge_model": model,
        "merged_row_population_sha256": identity_sha256(
            [row["merged_row_receipt_sha256"] for row in merged]
        ),
        "novel_judge_bindings": novel_bindings,
        "novel_judge_population_sha256": identity_sha256(novel_bindings),
        "physical_provider_calls_during_merge": 0,
        "question_count": QUESTION_COUNT,
        "questions": merged,
        "reused_judgment_count": int(plan.payload["reused_judgment_count"]),
        "score_complete": True,
        "source_policy_replay_artifact_sha256": plan.payload[
            "source_policy_replay_artifact_sha256"
        ],
        "source_policy_run_artifact_sha256": plan.payload[
            "source_policy_run_artifact_sha256"
        ],
    }
    return payload


def load_verified_differential_judge_plan(
    path: str | Path, expected_sha256: str
) -> SealedArtifact:
    plan = _artifact(path, expected_sha256, "differential judge plan")
    validate_differential_judge_plan(plan.payload)
    return plan


def publish_differential_judge_plan(
    output_root: str | Path,
    policy_run: SealedArtifact,
    policy_replay: SealedArtifact,
    prior_runs: Sequence[AuthenticatedJudgeRun],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[SealedArtifact, bool]:
    payload = build_differential_judge_plan(
        policy_run, policy_replay, prior_runs, judge_model=judge_model
    )
    return publish_sealed_json(Path(output_root) / PLAN_NAME, payload)


def publish_differential_judge_merge(
    output_root: str | Path,
    plan: SealedArtifact,
    novel_judge_runs: Sequence[AuthenticatedJudgeRun] = (),
) -> tuple[SealedArtifact, bool]:
    payload = merge_differential_judgments(plan, novel_judge_runs)
    return publish_sealed_json(Path(output_root) / MERGE_NAME, payload)


def _load_prior_triplets(args: argparse.Namespace, prefix: str) -> tuple[AuthenticatedJudgeRun, ...]:
    preflights = tuple(getattr(args, f"{prefix}_preflight", ()) or ())
    preflight_hashes = tuple(
        getattr(args, f"expected_{prefix}_preflight_sha256", ()) or ()
    )
    judges = tuple(getattr(args, f"{prefix}_judge", ()) or ())
    judge_hashes = tuple(getattr(args, f"expected_{prefix}_judge_sha256", ()) or ())
    replays = tuple(getattr(args, f"{prefix}_replay", ()) or ())
    replay_hashes = tuple(
        getattr(args, f"expected_{prefix}_replay_sha256", ()) or ()
    )
    lengths = {
        len(preflights),
        len(preflight_hashes),
        len(judges),
        len(judge_hashes),
        len(replays),
        len(replay_hashes),
    }
    _require(len(lengths) == 1, f"{prefix} artifact arguments are not aligned")
    return tuple(
        load_authenticated_prior_judge_run(
            preflight_path=preflight,
            expected_preflight_sha256=preflight_sha,
            judge_path=judge,
            expected_judge_sha256=judge_sha,
            replay_path=replay,
            expected_replay_sha256=replay_sha,
        )
        for preflight, preflight_sha, judge, judge_sha, replay, replay_sha in zip(
            preflights,
            preflight_hashes,
            judges,
            judge_hashes,
            replays,
            replay_hashes,
            strict=True,
        )
    )


def _add_judge_triplets(
    parser: argparse.ArgumentParser, *, prefix: str, required: bool
) -> None:
    option = prefix.replace("_", "-")
    parser.add_argument(f"--{option}-preflight", type=Path, action="append", required=required)
    parser.add_argument(
        f"--expected-{option}-preflight-sha256", action="append", required=required
    )
    parser.add_argument(f"--{option}-judge", type=Path, action="append", required=required)
    parser.add_argument(
        f"--expected-{option}-judge-sha256", action="append", required=required
    )
    parser.add_argument(f"--{option}-replay", type=Path, action="append", required=required)
    parser.add_argument(
        f"--expected-{option}-replay-sha256", action="append", required=required
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--policy-root", type=Path, required=True)
    plan.add_argument("--expected-policy-run-sha256", required=True)
    plan.add_argument("--expected-policy-replay-sha256", required=True)
    plan.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_judge_triplets(plan, prefix="prior", required=True)

    merge = commands.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--expected-plan-sha256", required=True)
    merge.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_judge_triplets(merge, prefix="novel", required=False)
    return parser


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    run, replay, _rows = load_verified_policy_v5_source(
        args.policy_root,
        expected_run_sha256=str(args.expected_policy_run_sha256),
        expected_replay_sha256=str(args.expected_policy_replay_sha256),
    )
    priors = _load_prior_triplets(args, "prior")
    artifact, created = publish_differential_judge_plan(
        args.output_root,
        run,
        replay,
        priors,
        judge_model=str(args.judge_model),
    )
    return {
        "created": created,
        "novel_prompt_count": artifact.payload["novel_prompt_count"],
        "physical_provider_calls": 0,
        "plan_sha256": artifact.sha256,
        "reused_judgment_count": artifact.payload["reused_judgment_count"],
    }


def run_merge(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_verified_differential_judge_plan(
        args.plan, str(args.expected_plan_sha256)
    )
    novel = _load_prior_triplets(args, "novel")
    artifact, created = publish_differential_judge_merge(
        args.output_root, plan, novel
    )
    return {
        "accuracy": artifact.payload["accuracy"],
        "correct": artifact.payload["correct"],
        "created": created,
        "merge_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "question_count": QUESTION_COUNT,
    }


def _canonical_output(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_plan(args) if args.command == "plan" else run_merge(args)
    print(_canonical_output(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
