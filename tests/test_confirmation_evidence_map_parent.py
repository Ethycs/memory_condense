from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_confirmation_query_payload_parent import _confirmation_plan
from tests.test_matched_eval_query_expansion import _StructuredClient
from tools import confirmation_evidence_map_parent as confirmation_map
from tools import confirmation_query_payload_parent as confirmation_query
from tools.matched_eval import live
from tools.matched_eval import query_evidence_map_solver_v2_live as map_live
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
)


def _direct_plane(tmp_path: Path, *, count: int):
    fixture, plan = _confirmation_plan(tmp_path, count=count)
    output = fixture.root / "query-payload"
    preflight = confirmation_query.publish_confirmation_query_payload_preflight(
        plan, output_root=output, max_concurrency=1
    )
    release = confirmation_query.approve_confirmation_query_payload_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        approve_provider_release=True,
        authorized_provider_calls=plan.required_calls,
    )
    confirmation_query.run_confirmation_query_payload_provider(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        expected_release_sha256=release.sha256,
        enable_provider=bool(plan.required_calls),
        authorized_provider_calls=plan.required_calls,
        client_factory=lambda _gateway, _env: _StructuredClient(
            "The exact globally retrieved answer."
        ),
    )
    materialized = confirmation_query.materialize_confirmation_query_payload_answers(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        expected_release_sha256=release.sha256,
    )
    direct = confirmation_query.replay_confirmation_query_payload_answers(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        expected_release_sha256=release.sha256,
        expected_run_sha256=materialized.answer_artifact.sha256,
    )
    return fixture, plan, direct


def _map_completion(plan: confirmation_map.ConfirmationEvidenceMapPlan) -> str:
    row = plan.map_plan.submitted_rows[0]
    alias = row.aliases[-1]
    evidence = row.retained_query_delta[-1]
    assert row.route.style.value == "direct_extract"
    return json.dumps(
        {
            "items": [
                {
                    "alias": alias.alias,
                    "candidate": "the exact globally retrieved fact",
                    "citation": evidence.text,
                    "kind": "extract_span",
                }
            ]
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _map_preflight(tmp_path: Path, *, count: int):
    fixture, query_plan, direct = _direct_plane(tmp_path, count=count)
    plan = confirmation_map.build_confirmation_evidence_map_plan(
        query_plan, direct
    )
    output = fixture.root / "evidence-map"
    preflight = confirmation_map.publish_confirmation_evidence_map_preflight(
        plan, output_root=output, max_concurrency=1
    )
    return fixture, plan, output, preflight


def test_full_native_map_lifecycle_returns_exact_replayed_plane(
    tmp_path: Path,
) -> None:
    _fixture, plan, output, preflight = _map_preflight(tmp_path, count=3)

    assert plan.required_calls == 3
    assert preflight.prompt_artifact.payload["format"] == confirmation_map.PROMPT_FORMAT
    assert preflight.prompt_artifact.payload["runtime"]["retries"] == 0
    assert preflight.prompt_artifact.payload["physical_provider_calls"] == 0
    assert [
        row["provider_input"]["messages"]
        for row in preflight.prompt_artifact.payload["ordered_rows"]
    ] == preflight.map_preflight_artifact.payload["provider_prompts"]

    with pytest.raises(
        confirmation_map.ConfirmationEvidenceMapError,
        match="exact remaining",
    ):
        confirmation_map.approve_confirmation_evidence_map_release(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=2,
        )
    release = confirmation_map.approve_confirmation_evidence_map_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=3,
    )
    factory_calls: list[tuple[str, str]] = []

    def factory(gateway: str, environment: str):
        factory_calls.append((gateway, environment))
        return _StructuredClient(_map_completion(plan))

    provider = confirmation_map.run_confirmation_evidence_map_provider(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=3,
        client_factory=factory,
    )
    assert provider.physical_provider_calls == 3
    assert provider.checkpoint_hits == 0
    assert len(factory_calls) == 1

    materialized = confirmation_map.materialize_confirmation_evidence_map(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        expected_release_sha256=release.sha256,
    )
    assert materialized.physical_provider_calls == 0
    verified = confirmation_map.replay_confirmation_evidence_map(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        expected_release_sha256=release.sha256,
        expected_run_sha256=materialized.map_artifact.sha256,
    )
    assert type(verified) is map_live.VerifiedEvidenceMapPlane
    assert verified.run_sha256 == verified.replay_sha256
    assert verified.parent_plane is plan.direct_plane
    assert len(verified.rows) == 3
    assert all(row.accepted_items for row in verified.rows)


def test_partial_native_checkpoint_releases_only_remaining_calls(
    tmp_path: Path,
) -> None:
    _fixture, plan, output, preflight = _map_preflight(tmp_path, count=2)
    population = map_live.load_map_provider_population(
        output_root=output,
        expected_preflight_sha256=preflight.map_preflight_artifact.sha256,
    )
    runtime = map_live._provider_runtime(
        population,
        client=_StructuredClient(_map_completion(plan)),
        max_concurrency=1,
        gateway_url=live.DEFAULT_GATEWAY_URL,
    )
    assert population.prompt_population is not None
    first_messages_sha = population.prompt_population.ordered_rows[0].messages_sha256
    try:
        first = runtime._provider_call(first_messages_sha)
    finally:
        runtime.close()
    assert first.physical_call is True

    release = confirmation_map.approve_confirmation_evidence_map_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=1,
    )
    assert release.payload["checkpoint_snapshot"][
        "authenticated_complete_count"
    ] == 1
    assert release.payload["required_authorized_provider_calls"] == 1

    provider = confirmation_map.run_confirmation_evidence_map_provider(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client_factory=lambda _gateway, _env: _StructuredClient(
            _map_completion(plan)
        ),
    )
    assert provider.physical_provider_calls == 1
    assert provider.checkpoint_hits == 1


def test_request_only_map_checkpoint_refuses_release_without_a_client(
    tmp_path: Path,
) -> None:
    _fixture, _plan, output, preflight = _map_preflight(tmp_path, count=1)
    checkpoint = output / map_live.MAP_CHECKPOINT_DIR_NAME
    checkpoint.mkdir()
    (checkpoint / ("a" * 64 + ".request.json")).write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(
        confirmation_map.ConfirmationEvidenceMapError,
        match="unsafe retry forbidden",
    ):
        confirmation_map.approve_confirmation_evidence_map_release(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=1,
        )


def test_resealed_release_extension_fails_before_client_construction(
    tmp_path: Path,
) -> None:
    _fixture, _plan, output, preflight = _map_preflight(tmp_path, count=1)
    release = confirmation_map.approve_confirmation_evidence_map_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=1,
    )
    changed = copy.deepcopy(release.payload)
    changed["benign_extension"] = "not allowed"
    changed.pop("release_identity_sha256")
    changed["release_identity_sha256"] = identity_sha256(changed)
    raw = canonical_json_bytes(changed)
    digest = hashlib.sha256(raw).hexdigest()
    release.path.write_bytes(raw)
    release.path.with_name(release.path.name + ".sha256").write_bytes(
        f"{digest}  {release.path.name}\n".encode("ascii")
    )
    factory_calls: list[bool] = []

    with pytest.raises(
        confirmation_map.ConfirmationEvidenceMapError,
        match="release schema changed",
    ):
        confirmation_map.run_confirmation_evidence_map_provider(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_map_preflight_sha256=preflight.map_preflight_artifact.sha256,
            expected_release_sha256=digest,
            enable_provider=True,
            authorized_provider_calls=1,
            client_factory=lambda *_args: factory_calls.append(True),
        )
    assert factory_calls == []


def test_direct_parent_binding_tamper_is_rejected(tmp_path: Path) -> None:
    _fixture, query_plan, direct = _direct_plane(tmp_path, count=1)
    changed = replace(direct, run_sha256="a" * 64)

    with pytest.raises(MatchedEvalContractError, match="binding changed"):
        confirmation_map.build_confirmation_evidence_map_plan(
            query_plan, changed
        )
