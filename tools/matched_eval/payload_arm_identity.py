"""Semantic identity sidecars for immutable shared payload answer planes.

Query expansion, partition scan, and query-guided scan deliberately reuse the
same payload responder and changed-only judge execution contract.  Their
historical answer and judge artifacts therefore carry the same execution arm
label.  This module adds a separate, sealed semantic identity without changing
any answer, prompt, runtime, judge, score, or provider-journal byte.

The semantic profile is recovered from the exact non-S0 alias tier already
sealed into every answer row.  The sidecar then binds that profile to the
adapter population, construction sources, answer run/replay, runtime
run/replay, prompt population, parent answer, and a prediction-hash projection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import _validated_runtime_ledger
from .query_payload_live import (
    ALIAS_RECEIPT_FORMAT,
    ANSWER_PLAN_ID,
    ANSWER_PREFLIGHT_FORMAT,
    ANSWER_PREFLIGHT_NAME,
    ANSWER_REPLAY_NAME,
    ANSWER_RUN_FORMAT,
    ANSWER_RUN_NAME,
    ARM_LABEL,
    ARM_PLAN_ID,
    PARENT_ARM_LABEL,
    RENDERER_ID,
    RUNTIME_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME,
)


BINDING_FORMAT = "memory-condense-payload-semantic-arm-binding-v1"
BINDING_NAME = "semantic-arm-binding.json"


class PayloadArmIdentityError(MatchedEvalContractError):
    """Raised when a semantic profile does not match immutable answer bytes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise PayloadArmIdentityError(message)


@dataclass(frozen=True, slots=True)
class PayloadSemanticArmProfile:
    kind: str
    semantic_arm_label: str
    semantic_arm_plan_id: str
    construction_kind: str
    delta_tier: str

    def projection(self) -> dict[str, str]:
        return {
            "construction_kind": self.construction_kind,
            "delta_tier": self.delta_tier,
            "kind": self.kind,
            "semantic_arm_label": self.semantic_arm_label,
            "semantic_arm_plan_id": self.semantic_arm_plan_id,
        }

    @property
    def profile_sha256(self) -> str:
        return identity_sha256(self.projection())


QUERY_PAYLOAD_PROFILE = PayloadSemanticArmProfile(
    kind="query_payload",
    semantic_arm_label="S0_PLUS_QUERY_PAYLOAD_V1",
    semantic_arm_plan_id="matched_s0_plus_query_payload_v1",
    construction_kind="query_expansion_v1",
    delta_tier="query_expansion_delta",
)
PARTITION_PAYLOAD_PROFILE = PayloadSemanticArmProfile(
    kind="partition_payload",
    semantic_arm_label="S0_PLUS_PARTITION_SCAN_V2_PAYLOAD_V1",
    semantic_arm_plan_id="matched_s0_plus_partition_scan_v2_payload_v1",
    construction_kind="partition_scan_v2",
    delta_tier="partition_scan_v2_delta",
)
QUERY_GUIDED_PAYLOAD_PROFILE = PayloadSemanticArmProfile(
    kind="query_guided_payload",
    semantic_arm_label="S0_PLUS_QUERY_GUIDED_PAYLOAD_V1",
    semantic_arm_plan_id="matched_s0_plus_query_guided_payload_v1",
    construction_kind="query_guided_scan_v1",
    delta_tier="query_guided_scan_delta",
)

PAYLOAD_SEMANTIC_PROFILES = (
    QUERY_PAYLOAD_PROFILE,
    PARTITION_PAYLOAD_PROFILE,
    QUERY_GUIDED_PAYLOAD_PROFILE,
)
_PROFILE_BY_KIND = {profile.kind: profile for profile in PAYLOAD_SEMANTIC_PROFILES}
_PROFILE_BY_TIER = {
    profile.delta_tier: profile for profile in PAYLOAD_SEMANTIC_PROFILES
}

CLI_ARM_TO_PROFILE_KIND = {
    "query-payload": QUERY_PAYLOAD_PROFILE.kind,
    "partition-payload": PARTITION_PAYLOAD_PROFILE.kind,
    "query-guided-payload": QUERY_GUIDED_PAYLOAD_PROFILE.kind,
}


def profile_for_kind(kind: str) -> PayloadSemanticArmProfile:
    try:
        return _PROFILE_BY_KIND[kind]
    except KeyError as exc:
        raise PayloadArmIdentityError(
            f"unknown payload semantic arm kind: {kind!r}"
        ) from exc


def profile_for_cli_arm(arm: str) -> PayloadSemanticArmProfile:
    try:
        return profile_for_kind(CLI_ARM_TO_PROFILE_KIND[arm])
    except KeyError as exc:
        raise PayloadArmIdentityError(
            f"CLI arm has no payload semantic profile: {arm!r}"
        ) from exc


def profile_for_delta_tier(delta_tier: str) -> PayloadSemanticArmProfile:
    try:
        return _PROFILE_BY_TIER[delta_tier]
    except KeyError as exc:
        raise PayloadArmIdentityError(
            f"alias tier has no payload semantic profile: {delta_tier!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class _VerifiedPayloadArtifacts:
    preflight: SealedArtifact
    run: SealedArtifact
    replay: SealedArtifact
    runtime: SealedArtifact
    runtime_replay: SealedArtifact
    runtime_identity_sha256: str
    source_bindings: Mapping[str, str]
    observed_delta_tier: str
    prediction_projection_sha256: str


def _read_pair(
    root: Path, name: str, replay_name: str, label: str
) -> tuple[SealedArtifact, SealedArtifact]:
    source = read_sealed_json(root / name)
    replay = read_sealed_json(root / replay_name)
    _require(
        source.sha256 == replay.sha256 and source.payload == replay.payload,
        f"{label} run/replay differ",
    )
    return source, replay


def _source_bindings(
    runtime: Mapping[str, Any], *, expected_prefix: str
) -> dict[str, str]:
    raw = runtime.get("source_artifacts")
    _require(type(raw) is list, "payload runtime source artifacts changed")
    expected_roles = (
        "sealed_retrieval",
        "query_preflight",
        "query_run",
        "query_adapter",
        "parent_answer_run",
        "parent_runtime_ledger",
        "answer_preflight",
        "answer_run",
    )
    roles: list[str] = []
    result: dict[str, str] = {}
    for index, item in enumerate(raw):
        _require(
            type(item) is dict and set(item) == {"role", "sha256"},
            f"payload runtime source artifact {index} changed",
        )
        role = require_text(item["role"], "payload runtime source role")
        _require(
            role.startswith(expected_prefix),
            "payload runtime source execution arm changed",
        )
        short = role[len(expected_prefix) :]
        roles.append(short)
        result[short] = require_sha256(
            item["sha256"], f"payload runtime source {short}"
        )
    _require(
        tuple(roles) == expected_roles and len(result) == len(roles),
        "payload runtime source binding order changed",
    )
    return result


def _verify_answer_rows(
    rows: object,
    *,
    expected_question_count: int,
) -> tuple[str, str]:
    _require(type(rows) is list, "payload answer questions must be an exact array")
    _require(
        len(rows) == expected_question_count,
        "payload answer question count changed",
    )
    delta_tiers: set[str] = set()
    projection: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        _require(type(raw) is dict, f"payload answer row {index} changed")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        _require(
            type(source_row_sha) is str
            and source_row_sha == identity_sha256(unsigned),
            f"payload answer row {index} identity seal changed",
        )
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        _require(
            type(prediction) is str
            and bool(prediction)
            and prediction_sha == quote_sha256(prediction),
            f"payload answer prediction {index} changed",
        )
        aliases = raw.get("alias_receipt")
        _require(
            type(aliases) is list and all(type(alias) is dict for alias in aliases),
            f"payload answer aliases {index} changed",
        )
        alias_projection = {
            "aliases": aliases,
            "format": ALIAS_RECEIPT_FORMAT,
            "query_row_receipt_sha256": raw.get("query_row_receipt_sha256"),
        }
        _require(
            raw.get("alias_receipt_format") == ALIAS_RECEIPT_FORMAT
            and raw.get("alias_receipt_sha256")
            == identity_sha256(alias_projection),
            f"payload alias receipt {index} changed",
        )
        for alias in aliases:
            tier = require_text(alias.get("tier"), "payload alias tier")
            _require(
                tier == "protected_s0" or tier in _PROFILE_BY_TIER,
                f"unknown payload semantic alias tier: {tier!r}",
            )
            if tier != "protected_s0":
                delta_tiers.add(tier)
        projection.append(
            {
                "changed_from_parent": raw.get("changed_from_parent"),
                "ordinal": raw.get("ordinal"),
                "prediction_sha256": prediction_sha,
                "question_id": raw.get("question_id"),
                "question_sha256": raw.get("question_sha256"),
                "source_row_sha256": source_row_sha,
            }
        )
    _require(
        len(delta_tiers) == 1,
        "payload semantic arm requires one exact non-S0 alias tier",
    )
    return next(iter(delta_tiers)), identity_sha256(projection)


def _verify_payload_artifacts(
    output_root: str | Path,
    *,
    expected_question_count: int,
) -> _VerifiedPayloadArtifacts:
    root = Path(output_root)
    preflight = read_sealed_json(root / ANSWER_PREFLIGHT_NAME)
    run, replay = _read_pair(root, ANSWER_RUN_NAME, ANSWER_REPLAY_NAME, "answer")
    runtime, runtime_replay = _read_pair(
        root,
        RUNTIME_LEDGER_NAME,
        RUNTIME_LEDGER_REPLAY_NAME,
        "runtime ledger",
    )
    _require(
        preflight.payload.get("format") == ANSWER_PREFLIGHT_FORMAT
        and run.payload.get("format") == ANSWER_RUN_FORMAT,
        "payload answer execution format changed",
    )
    for payload, label in (
        (preflight.payload, "preflight"),
        (run.payload, "answer run"),
    ):
        _require(
            payload.get("arm_label") == ARM_LABEL
            and payload.get("arm_plan_id") == ARM_PLAN_ID
            and payload.get("answer_plan_id") == ANSWER_PLAN_ID
            and payload.get("parent_arm_label") == PARENT_ARM_LABEL
            and payload.get("renderer_id") == RENDERER_ID,
            f"payload {label} shared execution identity changed",
        )
        _require(payload.get("gold_loaded") is False, f"payload {label} loaded gold")
        _require(
            payload.get("retained_request_token_state_bytes") == 0,
            f"payload {label} retained transformer request state",
        )
    _require(
        preflight.payload.get("provider_calls") == 0,
        "payload answer preflight performed provider calls",
    )
    _require(
        preflight.payload.get("adapter_population_id")
        == run.payload.get("adapter_population_id")
        and preflight.payload.get("parent_answer_run_sha256")
        == run.payload.get("parent_answer_run_sha256")
        and preflight.payload.get("prompt_population_sha256")
        == run.payload.get("prompt_population_sha256")
        and preflight.payload.get("retrieval_sha256")
        == run.payload.get("retrieval_sha256")
        and preflight.payload.get("snapshot_id") == run.payload.get("snapshot_id"),
        "payload preflight/run binding changed",
    )
    runtime_identity, answer_row_ids = _validated_runtime_ledger(runtime.payload)
    _require(
        runtime.payload.get("plan_id") == ANSWER_PLAN_ID
        and runtime.payload.get("snapshot_id") == run.payload.get("snapshot_id")
        and len(answer_row_ids) == expected_question_count,
        "payload runtime execution identity changed",
    )
    for index, row in enumerate(runtime.payload["rows"]):
        _require(
            row.get("arm_label") == ARM_LABEL
            and row.get("parent_arm_label") == PARENT_ARM_LABEL
            and row.get("renderer_id") == RENDERER_ID,
            f"payload runtime row {index} execution identity changed",
        )
    sources = _source_bindings(
        runtime.payload,
        expected_prefix=f"{ARM_LABEL}:",
    )
    _require(
        sources["sealed_retrieval"] == run.payload.get("retrieval_sha256")
        and sources["query_adapter"]
        == run.payload.get("adapter_population_id")
        and sources["parent_answer_run"]
        == run.payload.get("parent_answer_run_sha256")
        and sources["parent_runtime_ledger"]
        == run.payload.get("parent_answer_runtime_ledger_sha256")
        and sources["answer_preflight"] == preflight.sha256
        and sources["answer_run"] == run.sha256
        and sources["query_preflight"]
        == preflight.payload.get("query_preflight_sha256")
        and sources["query_run"] == preflight.payload.get("query_run_sha256"),
        "payload runtime source bindings changed",
    )
    observed_tier, prediction_projection = _verify_answer_rows(
        run.payload.get("questions"),
        expected_question_count=expected_question_count,
    )
    _require(
        run.payload.get("question_count") == expected_question_count
        and run.payload.get("logical_prediction_count") == expected_question_count,
        "payload answer population changed",
    )
    return _VerifiedPayloadArtifacts(
        preflight=preflight,
        run=run,
        replay=replay,
        runtime=runtime,
        runtime_replay=runtime_replay,
        runtime_identity_sha256=runtime_identity,
        source_bindings=sources,
        observed_delta_tier=observed_tier,
        prediction_projection_sha256=prediction_projection,
    )


def _binding_payload(
    output_root: str | Path,
    *,
    profile: PayloadSemanticArmProfile,
    expected_question_count: int,
) -> dict[str, Any]:
    if type(profile) is not PayloadSemanticArmProfile:
        raise TypeError("profile must be an exact PayloadSemanticArmProfile")
    verified = _verify_payload_artifacts(
        output_root,
        expected_question_count=expected_question_count,
    )
    _require(
        verified.observed_delta_tier == profile.delta_tier,
        "payload semantic profile does not match the sealed alias tier",
    )
    run = verified.run.payload
    source_bindings = dict(verified.source_bindings)
    body: dict[str, Any] = {
        "adapter_population_id": run["adapter_population_id"],
        "answer_preflight_sha256": verified.preflight.sha256,
        "answer_replay_sha256": verified.replay.sha256,
        "answer_run_sha256": verified.run.sha256,
        "answer_runtime_ledger_replay_sha256": verified.runtime_replay.sha256,
        "answer_runtime_ledger_sha256": verified.runtime.sha256,
        "answer_runtime_identity_sha256": verified.runtime_identity_sha256,
        "binding_kind": "sidecar_over_immutable_shared_payload_execution",
        "construction_source_bindings": {
            "adapter_population_sha256": source_bindings["query_adapter"],
            "construction_preflight_sha256": source_bindings["query_preflight"],
            "construction_run_sha256": source_bindings["query_run"],
        },
        "execution_identity": {
            "answer_plan_id": ANSWER_PLAN_ID,
            "arm_label": ARM_LABEL,
            "arm_plan_id": ARM_PLAN_ID,
            "renderer_id": RENDERER_ID,
        },
        "format": BINDING_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "parent_answer_run_sha256": run["parent_answer_run_sha256"],
        "prediction_hash_projection_sha256": (
            verified.prediction_projection_sha256
        ),
        "prompt_population_sha256": run["prompt_population_sha256"],
        "provider_prompt_content_mutated": False,
        "question_count": expected_question_count,
        "retrieval_sha256": run["retrieval_sha256"],
        "sealed_prediction_bytes_mutated": False,
        "semantic_profile": profile.projection(),
        "semantic_profile_sha256": profile.profile_sha256,
        "snapshot_id": run["snapshot_id"],
    }
    assert_gold_blind(body, path="payload_semantic_arm_binding")
    body["semantic_arm_binding_sha256"] = identity_sha256(body)
    return body


def ensure_payload_semantic_arm_binding(
    output_root: str | Path,
    *,
    profile: PayloadSemanticArmProfile,
    expected_question_count: int = 100,
) -> tuple[SealedArtifact, bool]:
    """Publish once, or verify, a sidecar over immutable payload artifacts."""

    payload = _binding_payload(
        output_root,
        profile=profile,
        expected_question_count=expected_question_count,
    )
    return publish_sealed_json(Path(output_root) / BINDING_NAME, payload)


def load_verified_payload_semantic_arm_binding(
    output_root: str | Path,
    *,
    expected_profile: PayloadSemanticArmProfile | None = None,
    expected_binding_sha256: str | None = None,
    expected_question_count: int = 100,
) -> SealedArtifact:
    """Verify the sidecar and rederive it from the current sealed artifacts."""

    root = Path(output_root)
    artifact = read_sealed_json(root / BINDING_NAME)
    raw_profile = artifact.payload.get("semantic_profile")
    _require(type(raw_profile) is dict, "payload semantic profile changed")
    kind = require_text(raw_profile.get("kind"), "payload semantic profile kind")
    profile = profile_for_kind(kind)
    if expected_profile is not None:
        _require(
            type(expected_profile) is PayloadSemanticArmProfile
            and profile == expected_profile,
            "payload semantic profile does not match the requested arm",
        )
    if expected_binding_sha256 is not None:
        _require(
            artifact.sha256
            == require_sha256(
                expected_binding_sha256,
                "expected payload semantic-arm binding",
            ),
            "payload semantic-arm binding SHA-256 changed",
        )
    expected = _binding_payload(
        root,
        profile=profile,
        expected_question_count=expected_question_count,
    )
    _require(
        artifact.payload == expected,
        "payload semantic-arm binding no longer matches immutable artifacts",
    )
    unsigned = dict(artifact.payload)
    declared = unsigned.pop("semantic_arm_binding_sha256", None)
    _require(
        type(declared) is str and declared == identity_sha256(unsigned),
        "payload semantic-arm binding self-identity changed",
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "verify"))
    parser.add_argument(
        "--profile-kind",
        choices=tuple(_PROFILE_BY_KIND),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256")
    parser.add_argument("--expected-question-count", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = profile_for_kind(args.profile_kind)
    if args.command == "seal":
        artifact, created = ensure_payload_semantic_arm_binding(
            args.output_root,
            profile=profile,
            expected_question_count=args.expected_question_count,
        )
    else:
        artifact = load_verified_payload_semantic_arm_binding(
            args.output_root,
            expected_profile=profile,
            expected_binding_sha256=args.expected_binding_sha256,
            expected_question_count=args.expected_question_count,
        )
        created = False
    print(
        json.dumps(
            {
                "artifact": artifact.path.as_posix(),
                "created": created,
                "profile_kind": profile.kind,
                "provider_calls": 0,
                "semantic_arm_binding_sha256": artifact.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BINDING_FORMAT",
    "BINDING_NAME",
    "CLI_ARM_TO_PROFILE_KIND",
    "PARTITION_PAYLOAD_PROFILE",
    "PAYLOAD_SEMANTIC_PROFILES",
    "PayloadArmIdentityError",
    "PayloadSemanticArmProfile",
    "QUERY_GUIDED_PAYLOAD_PROFILE",
    "QUERY_PAYLOAD_PROFILE",
    "ensure_payload_semantic_arm_binding",
    "load_verified_payload_semantic_arm_binding",
    "profile_for_cli_arm",
    "profile_for_delta_tier",
    "profile_for_kind",
]
