"""Journal-derived usage and final-cost closure for the Mem0 full100 arm.

The older campaign cost helper accepts a caller-supplied usage mapping.  That
mapping is useful for non-certifying development subsets, but it is not an
authority for the fair full100 comparison.  This module is the certifying
boundary: it first reopens the complete Terra and Sol lifecycles through their
strict readers, then derives calls, retries, token totals, latency, routes, and
budgets from the authenticated checkpoint batches.

No function in this module constructs a provider client or invokes an endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.live import DEFAULT_GATEWAY_URL
from tools.matched_eval.typed_memory_final_arm import (
    HARD_PROMPT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
)

from .typed_answer_lifecycle import (
    PREFLIGHT_NAME as ANSWER_PREFLIGHT_NAME,
    REPLAY_NAME as ANSWER_REPLAY_NAME,
    RUN_FORMAT as ANSWER_RUN_FORMAT,
    RUN_NAME as ANSWER_RUN_NAME,
)
from .typed_cost_ledger import (
    CommonProviderStageCost,
)
from .typed_epoch_campaign import (
    COMPARISON_SEMANTICS,
    FINAL_COST_FORMAT,
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKEN_RESERVE,
    MEM0_TYPED_EPOCH,
    RESPONDER_MODEL,
    _validate_common_input,
    _validate_cost_preflight,
)
from .typed_judge_lifecycle import (
    JUDGE_FORMAT,
    JUDGE_NAME,
    JUDGE_REPLAY_NAME,
    MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
    MAX_JUDGE_PROMPT_TOKENS,
    PREFLIGHT_NAME as JUDGE_PREFLIGHT_NAME,
    SCORE_NAME,
    SCORE_REPLAY_NAME,
    load_verified_judge_score,
)


QUESTION_COUNT = 100
USAGE_ATTESTATION_FORMAT = (
    "memory-condense-mem0-common-parent-journal-usage-attestation-v1"
)
USAGE_ATTESTATION_NAME = "mem0-common-parent-journal-usage-attestation-v1.json"
USAGE_ATTESTATION_REPLAY_NAME = (
    "mem0-common-parent-journal-usage-attestation-replay-v1.json"
)
FINAL_COST_ATTESTATION_FORMAT = f"{FINAL_COST_FORMAT}-journal-attested-v1"
COMMON_FINAL_ATTESTATION_FORMAT = (
    "memory-condense-common-final-cost-journal-attested-v1"
)
EPOCH_COST_ATTESTATION_FORMAT = (
    "memory-condense-mem0-typed-epoch-cost-journal-attested-v1"
)
FINAL_COST_ATTESTATION_NAME = "mem0-typed-final-cost-journal-attested-v1.json"
FINAL_COST_ATTESTATION_REPLAY_NAME = (
    "mem0-typed-final-cost-journal-attested-replay-v1.json"
)


class Mem0UsageAttestationError(MatchedEvalContractError):
    """A journal, lifecycle, usage projection, or final cost escaped closure."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Mem0UsageAttestationError(message)


def _dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _read(path: str | Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, f"expected {label}"),
        f"{label} SHA-256 changed",
    )
    return artifact


def _exact_int(value: object, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be non-negative")
    return value  # type: ignore[return-value]


def _exact_float(value: object, label: str) -> float:
    _require(
        type(value) in {int, float}
        and math.isfinite(float(value))
        and float(value) >= 0,
        f"{label} must be finite and non-negative",
    )
    return float(value)


def _known_sum(records: list[dict[str, Any]], key: str) -> tuple[int, bool]:
    values = [record.get(key) for record in records]
    _require(
        all(value is None or (type(value) is int and value >= 0) for value in values),
        f"journal {key} changed",
    )
    return sum(int(value) for value in values if value is not None), all(
        value is not None for value in values
    )


def _receipt(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": identity_sha256(body)}


def _derive_stage(
    *,
    role: Literal["responder", "judge"],
    batch_payload: object,
    preflight: SealedArtifact,
) -> dict[str, Any]:
    """Derive one stage solely from a strict-reader-authenticated batch."""

    batch = _dict(batch_payload, f"{role} completion batch")
    usage = _dict(batch.get("usage"), f"{role} completion usage")
    provenance = _dict(batch.get("provenance"), f"{role} runtime provenance")
    benchmark = _dict(
        provenance.get("benchmark_provenance"), f"{role} benchmark provenance"
    )
    population = _dict(batch.get("prompt_population"), f"{role} prompt population")
    population_rows = [
        _dict(row, f"{role} population row")
        for row in _list(population.get("ordered_rows"), f"{role} population rows")
    ]
    records = [
        _dict(row, f"{role} completion record")
        for row in _list(batch.get("unique_records"), f"{role} completion records")
    ]
    prompt_rows = [
        _dict(row, f"{role} preflight prompt row")
        for row in _list(preflight.payload.get("prompt_rows"), f"{role} prompt rows")
    ]

    model = RESPONDER_MODEL if role == "responder" else JUDGE_MODEL
    output_reserve = (
        OUTPUT_TOKEN_RESERVE if role == "responder" else JUDGE_OUTPUT_TOKEN_RESERVE
    )
    prompt_cap = (
        MAX_CHAT_PROMPT_TOKENS if role == "responder" else MAX_JUDGE_PROMPT_TOKENS
    )
    complete_cap = (
        HARD_PROMPT_TOKEN_CAP
        if role == "responder"
        else MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS
    )
    experiment_format = ANSWER_RUN_FORMAT if role == "responder" else JUDGE_FORMAT

    _require(
        len(records) == len(prompt_rows) == len(population_rows) == QUESTION_COUNT
        and usage.get("logical_calls") == QUESTION_COUNT
        and usage.get("unique_calls") == QUESTION_COUNT
        and usage.get("deduplicated_logical_calls") == 0
        and usage.get("checkpoint_hits") == QUESTION_COUNT
        and usage.get("physical_calls") == 0
        and provenance.get("model") == model
        and provenance.get("max_new_tokens") == output_reserve
        and provenance.get("max_prompt_token_proxy") == prompt_cap
        and provenance.get("retries") == 0
        and provenance.get("request_options") == {}
        and provenance.get("persisted_transformer_token_state") is False
        and provenance.get("retained_transformer_token_state_bytes") == 0
        and provenance.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and batch.get("runtime_identity_sha256") == identity_sha256(provenance)
        and population == preflight.payload.get("prompt_population")
        and benchmark.get("authorized_unique_calls") == QUESTION_COUNT
        and benchmark.get("comparison_semantics") == COMPARISON_SEMANTICS
        and benchmark.get("gateway_url") == DEFAULT_GATEWAY_URL
        and benchmark.get("preflight_artifact_sha256") == preflight.sha256
        and preflight.payload.get("model") == model
        and preflight.payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and preflight.payload.get("sdk_retries") == 0
        and preflight.payload.get("required_authorized_provider_calls")
        == QUESTION_COUNT
        and preflight.payload.get("output_token_reserve") == output_reserve
        and preflight.payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP,
        f"{role} batch is not the exact full100 zero-retry route",
    )
    if role == "responder":
        _require(
            benchmark.get("experiment_format") == experiment_format
            and preflight.payload.get("max_chat_prompt_tokens") == prompt_cap,
            "responder route or prompt budget changed",
        )
    else:
        _require(
            benchmark.get("experiment_format") == experiment_format
            and preflight.payload.get("max_judge_prompt_tokens") == prompt_cap
            and preflight.payload.get("max_judge_complete_envelope_tokens")
            == complete_cap,
            "judge route or prompt budget changed",
        )

    request_rows: list[dict[str, str]] = []
    response_rows: list[dict[str, str]] = []
    pair_rows: list[dict[str, str]] = []
    prompt_proxy = completion_proxy = 0
    elapsed_values: list[float] = []
    call_keys: set[str] = set()
    request_receipts: set[str] = set()
    response_receipts: set[str] = set()
    response_ids: set[str] = set()
    returned_models: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records):
        prompt_row = prompt_rows[ordinal]
        population_row = population_rows[ordinal]
        call_key = require_sha256(record.get("call_key_sha256"), "call key")
        request_sha = require_sha256(
            record.get("request_journal_sha256"), "request journal"
        )
        response_sha = require_sha256(
            record.get("response_journal_sha256"), "response journal"
        )
        response_id = record.get("response_id")
        response_model = record.get("response_model")
        record_prompt_proxy = record.get("prompt_token_proxy")
        record_completion_proxy = record.get("completion_token_proxy")
        reported_prompt = record.get("reported_prompt_tokens")
        reported_completion = record.get("reported_completion_tokens")
        reported_total = record.get("reported_total_tokens")
        accounting_prompt = (
            reported_prompt if reported_prompt is not None else record_prompt_proxy
        )
        _require(
            record.get("checkpoint_hit") is True
            and record.get("physical_call") is False
            and record.get("requested_model") == model
            and population_row.get("ordinal") == ordinal
            and record.get("messages_sha256")
            == population_row.get("messages_sha256")
            == prompt_row.get("messages_sha256")
            and record.get("prompt_token_proxy")
            == population_row.get("prompt_token_proxy")
            == prompt_row.get("prompt_token_proxy")
            and record.get("finish_reason") == "stop"
            and type(response_id) is str
            and bool(response_id)
            and response_id not in response_ids
            and type(response_model) is str
            and bool(response_model)
            and type(record_completion_proxy) is int
            and 0 <= record_completion_proxy <= output_reserve
            and (
                reported_prompt is None
                or (
                    type(reported_prompt) is int
                    and 1 <= reported_prompt <= prompt_cap
                )
            )
            and (
                reported_completion is None
                or (
                    type(reported_completion) is int
                    and 0 <= reported_completion <= output_reserve
                )
            )
            and (
                reported_total is None
                or (
                    type(reported_total) is int
                    and 0 <= reported_total <= accounting_prompt + output_reserve
                )
            )
            and call_key not in call_keys
            and request_sha not in request_receipts
            and response_sha not in response_receipts,
            f"{role} journal pair {ordinal} changed or repeats",
        )
        call_keys.add(call_key)
        request_receipts.add(request_sha)
        response_receipts.add(response_sha)
        response_ids.add(response_id)
        returned_models.append(
            {"ordinal": ordinal, "response_model": response_model}
        )
        prompt_proxy += _exact_int(
            record.get("prompt_token_proxy"), f"{role} prompt token proxy"
        )
        completion_proxy += _exact_int(
            record.get("completion_token_proxy"),
            f"{role} completion token proxy",
        )
        elapsed_values.append(
            _exact_float(
                record.get("provider_elapsed_s"), f"{role} provider latency"
            )
        )
        request_rows.append(
            {"call_key_sha256": call_key, "request_journal_sha256": request_sha}
        )
        response_rows.append(
            {"call_key_sha256": call_key, "response_journal_sha256": response_sha}
        )
        pair_rows.append(
            {
                "call_key_sha256": call_key,
                "request_journal_sha256": request_sha,
                "response_journal_sha256": response_sha,
            }
        )

    reported_prompt_known, reported_prompt_complete = _known_sum(
        records, "reported_prompt_tokens"
    )
    reported_completion_known, reported_completion_complete = _known_sum(
        records, "reported_completion_tokens"
    )
    reported_total_known, reported_total_complete = _known_sum(
        records, "reported_total_tokens"
    )
    _require(
        all(
            record.get("reported_total_tokens")
            == record.get("reported_prompt_tokens")
            + record.get("reported_completion_tokens")
            for record in records
            if record.get("reported_prompt_tokens") is not None
            and record.get("reported_completion_tokens") is not None
            and record.get("reported_total_tokens") is not None
        ),
        f"{role} provider-reported total token accounting does not close",
    )
    elapsed = math.fsum(elapsed_values)
    _require(
        usage.get("prompt_token_proxy") == prompt_proxy
        and usage.get("completion_token_proxy") == completion_proxy
        and usage.get("recorded_reported_prompt_tokens")
        == reported_prompt_known
        and usage.get("recorded_reported_completion_tokens")
        == reported_completion_known
        and usage.get("recorded_reported_total_tokens") == reported_total_known
        and usage.get("reported_prompt_tokens_complete")
        is reported_prompt_complete
        and usage.get("reported_completion_tokens_complete")
        is reported_completion_complete
        and usage.get("reported_total_tokens_complete") is reported_total_complete
        and math.isclose(
            float(usage.get("recorded_provider_elapsed_s")),
            elapsed,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{role} aggregate usage differs from its journal rows",
    )

    observed_prompt_max = max(
        _exact_int(row.get("prompt_token_proxy"), f"{role} row prompt tokens")
        for row in prompt_rows
    )
    observed_complete_max = max(
        _exact_int(row.get("prompt_token_proxy"), f"{role} row prompt tokens")
        + output_reserve
        for row in prompt_rows
    )
    _require(
        observed_prompt_max <= prompt_cap
        and observed_complete_max
        == preflight.payload.get("observed_max_complete_envelope_tokens")
        and observed_complete_max <= complete_cap,
        f"{role} observed request budget changed",
    )

    routes = [
        {
            "ordinal": ordinal,
            "route_id": row.get("route_id"),
            **(
                {"demand_class": row.get("demand_class")}
                if role == "judge"
                else {}
            ),
        }
        for ordinal, row in enumerate(prompt_rows)
    ]
    _require(
        all(
            type(row["route_id"]) is str
            and bool(row["route_id"])
            and (
                role == "responder"
                or (
                    type(row.get("demand_class")) is str
                    and bool(row["demand_class"])
                )
            )
            for row in routes
        ),
        f"{role} route population changed",
    )

    input_basis = (
        "provider_reported" if reported_prompt_complete else "deterministic_proxy"
    )
    output_basis = (
        "provider_reported"
        if reported_completion_complete
        else "deterministic_proxy"
    )
    body = {
        "budget": {
            "complete_envelope_token_cap": complete_cap,
            "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
            "observed_max_complete_envelope_tokens": observed_complete_max,
            "observed_max_prompt_token_proxy": observed_prompt_max,
            "output_token_reserve": output_reserve,
            "prompt_token_proxy_cap": prompt_cap,
        },
        "calls": {
            "attempted": len(request_rows),
            "completed": len(response_rows),
            "failed": len(request_rows) - len(response_rows),
            "retry_attempts": int(provenance["retries"]),
            "scope": "journaled_request_response_pairs",
        },
        "claim_boundary": (
            "content_authenticated_checkpoint_pairs_not_provider_signed_billing_"
            "or_gateway_internal_retry_evidence"
        ),
        "derivation": (
            "strict_lifecycle_reader_authenticated_request_response_journal_pairs_v1"
        ),
        "journal_pairs_sha256": identity_sha256(pair_rows),
        "latency_s": elapsed,
        "model_id": model,
        "prompt_population_sha256": require_sha256(
            provenance.get("prompt_population_sha256"), "prompt population"
        ),
        "request_journal_population_sha256": identity_sha256(request_rows),
        "response_journal_population_sha256": identity_sha256(response_rows),
        "retained_transformer_token_state_bytes": 0,
        "returned_model_accounting": {
            "claim_scope": (
                "provider_response_model_string_not_backend_route_attestation"
            ),
            "ordered_population_sha256": identity_sha256(returned_models),
            "returned_models": [
                {
                    "completed": sum(
                        row["response_model"] == model_name
                        for row in returned_models
                    ),
                    "response_model": model_name,
                }
                for model_name in sorted(
                    {row["response_model"] for row in returned_models}
                )
            ],
        },
        "role": role,
        "route": {
            "gateway_url_claim": DEFAULT_GATEWAY_URL,
            "model_id": model,
            "route_population_sha256": identity_sha256(routes),
            "route_scope": "requested_model_and_benchmark_gateway_claim",
            "runtime_identity_sha256": require_sha256(
                batch.get("runtime_identity_sha256"), "runtime identity"
            ),
        },
        "tokens": {
            "accounted_input_tokens": (
                reported_prompt_known if reported_prompt_complete else prompt_proxy
            ),
            "accounted_output_tokens": (
                reported_completion_known
                if reported_completion_complete
                else completion_proxy
            ),
            "completion_token_proxy": completion_proxy,
            "input_accounting_basis": input_basis,
            "output_accounting_basis": output_basis,
            "prompt_token_proxy": prompt_proxy,
            "reported_completion_tokens_complete": reported_completion_complete,
            "reported_completion_tokens_known_sum": reported_completion_known,
            "reported_prompt_tokens_complete": reported_prompt_complete,
            "reported_prompt_tokens_known_sum": reported_prompt_known,
            "reported_total_tokens_complete": reported_total_complete,
            "reported_total_tokens_known_sum": reported_total_known,
        },
    }
    _require(
        body["calls"]
        == {
            "attempted": 100,
            "completed": 100,
            "failed": 0,
            "retry_attempts": 0,
            "scope": "journaled_request_response_pairs",
        },
        f"{role} journal population is not exactly 100 successful zero-retry calls",
    )
    return _receipt(body)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedMem0UsageAttestation:
    """Capability issued only by a strict journal/lifecycle rebuild."""

    artifact: SealedArtifact
    replay: SealedArtifact
    responder: dict[str, Any]
    judge: dict[str, Any]

    def __init__(
        self,
        artifact: SealedArtifact,
        replay: SealedArtifact,
        responder: dict[str, Any],
        judge: dict[str, Any],
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_USAGE_TOKEN:
            raise Mem0UsageAttestationError(
                "usage capability requires the strict journal reader"
            )
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "replay", replay)
        object.__setattr__(self, "responder", dict(responder))
        object.__setattr__(self, "judge", dict(judge))


_VERIFIED_USAGE_TOKEN = object()
_VERIFIED_FINAL_COST_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedMem0FinalCost:
    """Capability issued only after lifecycle and final-cost replay rebuilds."""

    artifact: SealedArtifact
    replay: SealedArtifact
    usage: VerifiedMem0UsageAttestation

    def __init__(
        self,
        artifact: SealedArtifact,
        replay: SealedArtifact,
        usage: VerifiedMem0UsageAttestation,
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_FINAL_COST_TOKEN:
            raise Mem0UsageAttestationError(
                "final-cost capability requires the strict lifecycle reader"
            )
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "replay", replay)
        object.__setattr__(self, "usage", usage)


def _derive_usage_payload(
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    answer_output_root: str | Path,
    expected_answer_preflight_sha256: str,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    judge_output_root: str | Path,
    dataset_path: str | Path,
    split_path: str | Path,
    expected_judge_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
) -> dict[str, Any]:
    """Run both strict readers and derive an exact full100 usage projection."""

    judge, judge_replay, score, score_replay, _rows = load_verified_judge_score(
        judge_output_root,
        common_input_path=common_input_path,
        expected_common_input_sha256=expected_common_input_sha256,
        answer_output_root=answer_output_root,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_answer_run_sha256=expected_answer_run_sha256,
        expected_answer_replay_sha256=expected_answer_replay_sha256,
        dataset_path=dataset_path,
        split_path=split_path,
        expected_preflight_sha256=expected_judge_preflight_sha256,
        expected_judge_sha256=expected_judge_sha256,
        expected_judge_replay_sha256=expected_judge_replay_sha256,
        expected_score_sha256=expected_score_sha256,
        expected_score_replay_sha256=expected_score_replay_sha256,
        expected_question_count=QUESTION_COUNT,
    )
    answer_root = Path(answer_output_root)
    judge_root = Path(judge_output_root)
    answer_preflight = _read(
        answer_root / ANSWER_PREFLIGHT_NAME,
        expected_answer_preflight_sha256,
        "answer preflight",
    )
    answer_run = _read(
        answer_root / ANSWER_RUN_NAME, expected_answer_run_sha256, "answer run"
    )
    answer_replay = _read(
        answer_root / ANSWER_REPLAY_NAME,
        expected_answer_replay_sha256,
        "answer replay",
    )
    judge_preflight = _read(
        judge_root / JUDGE_PREFLIGHT_NAME,
        expected_judge_preflight_sha256,
        "judge preflight",
    )
    _require(
        answer_run.sha256 == answer_replay.sha256
        and judge.sha256 == judge_replay.sha256
        and score.sha256 == score_replay.sha256
        and answer_run.payload.get("question_count") == QUESTION_COUNT
        and judge.payload.get("question_count") == QUESTION_COUNT,
        "usage sources are not byte-identical full100 lifecycle replays",
    )
    responder = _derive_stage(
        role="responder",
        batch_payload=answer_run.payload.get("completion_batch"),
        preflight=answer_preflight,
    )
    sol = _derive_stage(
        role="judge",
        batch_payload=judge.payload.get("completion_batch"),
        preflight=judge_preflight,
    )
    answer_routes = [
        row.get("route_id")
        for row in _list(answer_preflight.payload.get("prompt_rows"), "answer routes")
    ]
    judge_routes = [
        row.get("route_id")
        for row in _list(judge_preflight.payload.get("prompt_rows"), "judge routes")
    ]
    common_sha = require_sha256(expected_common_input_sha256, "common input")
    _require(
        answer_routes == judge_routes
        and answer_preflight.payload.get("common_input_sha256") == common_sha
        and judge_preflight.payload.get("common_input_sha256") == common_sha
        and judge.payload.get("common_input_sha256") == common_sha
        and judge.payload.get("answer_run_sha256") == answer_run.sha256
        and judge.payload.get("answer_replay_sha256") == answer_replay.sha256,
        "Terra/Sol route or common-input binding changed",
    )
    body = {
        "common_input_sha256": common_sha,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": USAGE_ATTESTATION_FORMAT,
        "judge": sol,
        "parent_origin_receipt_sha256": require_sha256(
            judge.payload.get("parent_origin_receipt_sha256"), "parent origin"
        ),
        "question_count": QUESTION_COUNT,
        "responder": responder,
        "retained_transformer_token_state_bytes": 0,
        "strict_sources": {
            "answer_preflight_sha256": answer_preflight.sha256,
            "answer_replay_sha256": answer_replay.sha256,
            "answer_run_sha256": answer_run.sha256,
            "judge_preflight_sha256": judge_preflight.sha256,
            "judge_replay_sha256": judge_replay.sha256,
            "judge_run_sha256": judge.sha256,
            "score_replay_sha256": score_replay.sha256,
            "score_run_sha256": score.sha256,
        },
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    return _receipt(body)


def publish_usage_attestation(
    output_root: str | Path,
    **lifecycle_authority: Any,
) -> tuple[SealedArtifact, SealedArtifact]:
    """Publish a provider-free usage run/replay from strict lifecycle sources."""

    payload = _derive_usage_payload(**lifecycle_authority)
    root = Path(output_root)
    artifact, _ = publish_sealed_json(root / USAGE_ATTESTATION_NAME, payload)
    replay, _ = publish_sealed_json(root / USAGE_ATTESTATION_REPLAY_NAME, payload)
    _require(
        artifact.sha256 == replay.sha256 and artifact.payload == replay.payload,
        "usage attestation replay changed",
    )
    return artifact, replay


def load_verified_usage_attestation(
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    replay_path: str | Path,
    expected_replay_sha256: str,
    **lifecycle_authority: Any,
) -> VerifiedMem0UsageAttestation:
    """Rebuild journal usage and return a non-forgeable local capability."""

    artifact = _read(
        attestation_path, expected_attestation_sha256, "usage attestation"
    )
    replay = _read(replay_path, expected_replay_sha256, "usage attestation replay")
    rebuilt = _derive_usage_payload(**lifecycle_authority)
    _require(
        artifact.sha256 == replay.sha256
        and artifact.payload == replay.payload == rebuilt,
        "usage attestation is not an exact strict-reader replay",
    )
    return VerifiedMem0UsageAttestation(
        artifact,
        replay,
        _dict(artifact.payload.get("responder"), "responder usage"),
        _dict(artifact.payload.get("judge"), "judge usage"),
        _token=_VERIFIED_USAGE_TOKEN,
    )


def reopen_verified_usage_capability(
    usage: VerifiedMem0UsageAttestation,
) -> tuple[SealedArtifact, SealedArtifact, dict[str, Any], dict[str, Any]]:
    """Reopen a capability's files so mutable in-memory dicts have no authority."""

    _require(
        type(usage) is VerifiedMem0UsageAttestation,
        "usage authority is not a strict attestation capability",
    )
    artifact = _read(
        usage.artifact.path, usage.artifact.sha256, "usage capability artifact"
    )
    replay = _read(
        usage.replay.path, usage.replay.sha256, "usage capability replay"
    )
    responder = _dict(artifact.payload.get("responder"), "capability responder")
    judge = _dict(artifact.payload.get("judge"), "capability judge")
    _require(
        artifact.sha256 == replay.sha256
        and artifact.payload == replay.payload
        and artifact.payload == usage.artifact.payload
        and replay.payload == usage.replay.payload
        and responder == usage.responder
        and judge == usage.judge,
        "usage capability changed after strict verification",
    )
    return artifact, replay, responder, judge


def _cost_stage(
    stage: Mapping[str, Any], *, role: Literal["responder", "judge"]
) -> CommonProviderStageCost:
    calls = _dict(stage.get("calls"), f"{role} attested calls")
    tokens = _dict(stage.get("tokens"), f"{role} attested tokens")
    _require(
        calls
        == {
            "attempted": 100,
            "completed": 100,
            "failed": 0,
            "retry_attempts": 0,
            "scope": "journaled_request_response_pairs",
        },
        f"{role} final-cost calls are not closed",
    )
    return CommonProviderStageCost(
        role=role,
        model_id=RESPONDER_MODEL if role == "responder" else JUDGE_MODEL,
        logical_calls_attempted=calls["attempted"],
        logical_calls_completed=calls["completed"],
        logical_calls_failed=calls["failed"],
        sdk_retry_attempts=calls["retry_attempts"],
        provider_input_tokens=_exact_int(
            tokens.get("accounted_input_tokens"), f"{role} accounted input tokens"
        ),
        provider_output_tokens=_exact_int(
            tokens.get("accounted_output_tokens"),
            f"{role} accounted output tokens",
        ),
        latency_s=_exact_float(stage.get("latency_s"), f"{role} latency"),
    )


def build_verified_final_cost_payload(
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    cost_preflight_path: str | Path,
    expected_cost_preflight_sha256: str,
    usage: VerifiedMem0UsageAttestation,
) -> dict[str, Any]:
    """Build final costs only from a strict usage capability, never self-report."""

    _require(
        type(usage) is VerifiedMem0UsageAttestation,
        "final cost requires a strict usage-attestation capability",
    )
    usage_artifact, usage_replay, responder_usage, judge_usage = (
        reopen_verified_usage_capability(usage)
    )
    common_artifact = _read(
        common_input_path, expected_common_input_sha256, "common input"
    )
    cost_artifact = _read(
        cost_preflight_path, expected_cost_preflight_sha256, "cost preflight"
    )
    common = _validate_common_input(
        common_artifact.payload, expected_question_count=QUESTION_COUNT
    )
    cost_value, write, read = _validate_cost_preflight(
        cost_artifact.payload,
        common_input_sha256=common_artifact.sha256,
        contribution_bundle_sha256=common["contribution_bundle_sha256"],
        expected_question_count=QUESTION_COUNT,
    )
    _require(
        usage_artifact.payload.get("common_input_sha256") == common_artifact.sha256
        and usage_artifact.payload.get("parent_origin_receipt_sha256")
        == common.get("parent_origin_receipt_sha256")
        and read.retrieval_artifact_sha256 == common["retrieval_bundle_sha256"],
        "journal usage escaped the cost/common-input authority",
    )
    responder = _cost_stage(responder_usage, role="responder")
    judge = _cost_stage(judge_usage, role="judge")
    responder_budget = _dict(responder_usage.get("budget"), "responder budget")
    judge_budget = _dict(judge_usage.get("budget"), "judge budget")
    responder_prompt_max = _exact_int(
        responder_budget.get("observed_max_prompt_token_proxy"),
        "responder maximum prompt",
    )
    judge_prompt_max = _exact_int(
        judge_budget.get("observed_max_prompt_token_proxy"),
        "judge maximum prompt",
    )
    _require(
        responder_budget
        == {
            "complete_envelope_token_cap": HARD_PROMPT_TOKEN_CAP,
            "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
            "observed_max_complete_envelope_tokens": (
                responder_prompt_max + OUTPUT_TOKEN_RESERVE
            ),
            "observed_max_prompt_token_proxy": responder_prompt_max,
            "output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "prompt_token_proxy_cap": MAX_CHAT_PROMPT_TOKENS,
        }
        and judge_budget
        == {
            "complete_envelope_token_cap": MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
            "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
            "observed_max_complete_envelope_tokens": (
                judge_prompt_max + JUDGE_OUTPUT_TOKEN_RESERVE
            ),
            "observed_max_prompt_token_proxy": judge_prompt_max,
            "output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
            "prompt_token_proxy_cap": MAX_JUDGE_PROMPT_TOKENS,
        },
        "journal-attested final budgets differ from the Terra/Sol protocols",
    )
    final_body = {
        "format": COMMON_FINAL_ATTESTATION_FORMAT,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "judge": judge.projection(),
        "judge_complete_envelope_token_cap": MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
        "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
        "max_full_judge_prompt_token_proxy": judge_prompt_max,
        "max_full_responder_prompt_token_proxy": responder_prompt_max,
        "prompt_budget_compliant": True,
        "question_count": QUESTION_COUNT,
        "responder": responder.projection(),
        "responder_complete_envelope_token_cap": HARD_PROMPT_TOKEN_CAP,
        "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    final = _receipt(final_body)
    epoch_body = {
        "common_final_receipt_sha256": final["receipt_sha256"],
        "format": EPOCH_COST_ATTESTATION_FORMAT,
        "population_identity_sha256": cost_value["population_identity_sha256"],
        "read_receipt_sha256": read.receipt_sha256,
        "retrieval_artifact_sha256": read.retrieval_artifact_sha256,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "write_receipt_sha256": write.receipt_sha256,
    }
    epoch = _receipt(epoch_body)
    body = {
        "common_final_cost": final,
        "common_input_sha256": common_artifact.sha256,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "contribution_bundle_sha256": common["contribution_bundle_sha256"],
        "cost_preflight_sha256": cost_artifact.sha256,
        "epoch_cost": epoch,
        "format": FINAL_COST_ATTESTATION_FORMAT,
        "journal_usage_attestation_replay_sha256": usage_replay.sha256,
        "journal_usage_attestation_sha256": usage_artifact.sha256,
        "parent_origin_receipt_sha256": common["parent_origin_receipt_sha256"],
        "question_count": QUESTION_COUNT,
        "read_cost": read.projection(),
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "token_accounting": {
            "judge_input_basis": judge_usage["tokens"]["input_accounting_basis"],
            "judge_output_basis": judge_usage["tokens"]["output_accounting_basis"],
            "responder_input_basis": responder_usage["tokens"][
                "input_accounting_basis"
            ],
            "responder_output_basis": responder_usage["tokens"][
                "output_accounting_basis"
            ],
        },
        "usage_receipt_sha256": require_sha256(
            usage_artifact.payload.get("receipt_sha256"), "usage receipt"
        ),
        "write_cost": write.projection(),
    }
    return _receipt(body)


def publish_verified_final_cost(
    output_root: str | Path,
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    cost_preflight_path: str | Path,
    expected_cost_preflight_sha256: str,
    usage: VerifiedMem0UsageAttestation,
) -> tuple[SealedArtifact, SealedArtifact]:
    payload = build_verified_final_cost_payload(
        common_input_path=common_input_path,
        expected_common_input_sha256=expected_common_input_sha256,
        cost_preflight_path=cost_preflight_path,
        expected_cost_preflight_sha256=expected_cost_preflight_sha256,
        usage=usage,
    )
    root = Path(output_root)
    artifact, _ = publish_sealed_json(root / FINAL_COST_ATTESTATION_NAME, payload)
    replay, _ = publish_sealed_json(
        root / FINAL_COST_ATTESTATION_REPLAY_NAME, payload
    )
    _require(artifact.sha256 == replay.sha256, "final cost replay changed")
    return artifact, replay


def load_verified_final_cost(
    final_path: str | Path,
    expected_final_sha256: str,
    replay_path: str | Path,
    expected_replay_sha256: str,
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    cost_preflight_path: str | Path,
    expected_cost_preflight_sha256: str,
    usage_attestation_path: str | Path,
    expected_usage_attestation_sha256: str,
    usage_attestation_replay_path: str | Path,
    expected_usage_attestation_replay_sha256: str,
    lifecycle_authority: Mapping[str, Any],
) -> VerifiedMem0FinalCost:
    """Strict final reader: rerun lifecycle derivation, then rebuild costs."""

    _require(
        type(lifecycle_authority) is dict,
        "final cost lifecycle authority must be an exact argument object",
    )
    try:
        usage = load_verified_usage_attestation(
            usage_attestation_path,
            expected_usage_attestation_sha256,
            usage_attestation_replay_path,
            expected_usage_attestation_replay_sha256,
            **dict(lifecycle_authority),
        )
    except TypeError as exc:
        raise Mem0UsageAttestationError(
            "final cost lifecycle authority arguments changed"
        ) from exc
    artifact = _read(final_path, expected_final_sha256, "attested final cost")
    replay = _read(replay_path, expected_replay_sha256, "attested final cost replay")
    rebuilt = build_verified_final_cost_payload(
        common_input_path=common_input_path,
        expected_common_input_sha256=expected_common_input_sha256,
        cost_preflight_path=cost_preflight_path,
        expected_cost_preflight_sha256=expected_cost_preflight_sha256,
        usage=usage,
    )
    _require(
        artifact.sha256 == replay.sha256
        and artifact.payload == replay.payload == rebuilt,
        "attested final cost is not an exact journal-derived replay",
    )
    return VerifiedMem0FinalCost(
        artifact,
        replay,
        usage,
        _token=_VERIFIED_FINAL_COST_TOKEN,
    )


def reopen_verified_final_cost_capability(
    capability: VerifiedMem0FinalCost,
) -> tuple[SealedArtifact, SealedArtifact]:
    _require(
        type(capability) is VerifiedMem0FinalCost,
        "final-cost authority is not a strict reader capability",
    )
    artifact = _read(
        capability.artifact.path,
        capability.artifact.sha256,
        "final-cost capability",
    )
    replay = _read(
        capability.replay.path,
        capability.replay.sha256,
        "final-cost capability replay",
    )
    reopen_verified_usage_capability(capability.usage)
    _require(
        artifact.sha256 == replay.sha256
        and artifact.payload == replay.payload
        and artifact.payload == capability.artifact.payload
        and replay.payload == capability.replay.payload,
        "final-cost capability changed after verification",
    )
    return artifact, replay


__all__ = [
    "COMMON_FINAL_ATTESTATION_FORMAT",
    "EPOCH_COST_ATTESTATION_FORMAT",
    "FINAL_COST_ATTESTATION_FORMAT",
    "FINAL_COST_ATTESTATION_NAME",
    "FINAL_COST_ATTESTATION_REPLAY_NAME",
    "Mem0UsageAttestationError",
    "QUESTION_COUNT",
    "USAGE_ATTESTATION_FORMAT",
    "USAGE_ATTESTATION_NAME",
    "USAGE_ATTESTATION_REPLAY_NAME",
    "VerifiedMem0UsageAttestation",
    "VerifiedMem0FinalCost",
    "build_verified_final_cost_payload",
    "load_verified_final_cost",
    "load_verified_usage_attestation",
    "publish_usage_attestation",
    "publish_verified_final_cost",
    "reopen_verified_final_cost_capability",
    "reopen_verified_usage_capability",
]
