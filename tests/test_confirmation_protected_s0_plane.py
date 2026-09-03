from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import confirmation_protected_s0_plane as parent
from tools import confirmation_s0_prompt_preflight as s0
from tools import confirmation_terra_completion_lifecycle as lifecycle
from tools.confirmation_contracts import SealedJson, publish_sealed_json
from tools.matched_eval.population import MatchedS0Population
from tools.matched_eval.query_expansion import (
    FrozenSourceMembership,
    FrozenSourceNamespace,
    build_query_expansion_population,
)
from tools.v4_population_firebreak.canonical import canonical_sha256
from tests.test_confirmation_s0_prompt_preflight import Fixture, _build_fixture


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sealed(value: dict[str, object], key: str) -> dict[str, object]:
    return {**value, key: canonical_sha256(value)}


def _completion_fixture(
    fixture: Fixture,
) -> tuple[SealedJson, SealedJson, str, str, tuple[str, ...]]:
    prompt, _ = s0.publish_confirmation_s0_preflight(
        fixture.root / "s0-prompt.json", **fixture.kwargs()
    )
    verified = lifecycle.verify_prompt_artifact(
        prompt.path, expected_sha256=prompt.sha256
    )
    lifecycle_sha = _digest(f"lifecycle:{fixture.semantics}")
    release_sha = _digest(f"release:{fixture.semantics}")
    predictions = tuple(f"value {semantic}" for semantic in fixture.semantics)
    rows = []
    records = []
    for index, (prediction, question_id, source_receipt, prompt_row) in enumerate(
        zip(
            predictions,
            verified.question_ids,
            verified.row_receipts,
            verified.prompt_population.ordered_rows,
            strict=True,
        )
    ):
        call_key = _digest(f"call:{question_id}")
        request = _digest(f"request:{question_id}")
        response = _digest(f"response:{question_id}")
        prediction_sha = quote_sha256(prediction)
        row_body = {
            "format": lifecycle.COMPLETION_ROW_FORMAT,
            "row_index": index,
            "source_prompt_row_index": verified.source_row_indexes[index],
            "question_id": question_id,
            "source_prompt_row_receipt_sha256": source_receipt,
            "messages_sha256": prompt_row.messages_sha256,
            "completion": prediction,
            "completion_sha256": prediction_sha,
            "call_key_sha256": call_key,
            "request_journal_sha256": request,
            "response_journal_sha256": response,
        }
        rows.append(_sealed(row_body, "completion_row_receipt_sha256"))
        records.append(
            {
                "messages_sha256": prompt_row.messages_sha256,
                "completion": prediction,
                "completion_sha256": prediction_sha,
                "call_key_sha256": call_key,
                "request_journal_sha256": request,
                "response_journal_sha256": response,
                "checkpoint_hit": True,
                "physical_call": False,
            }
        )
    population = {
        **lifecycle.compile_lifecycle_preflight(verified)["population"],
        "question_count": len(rows),
    }
    body = {
        "format": lifecycle.COMPLETION_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "source_prompt_artifact_sha256": prompt.sha256,
        "lifecycle_preflight_sha256": lifecycle_sha,
        "provider_release_sha256": release_sha,
        "runtime": verified.runtime,
        "population": population,
        "ordered_rows": rows,
        "completion_batch": {
            "logical_completions": list(predictions),
            "unique_records": records,
            "usage": {
                "logical_calls": len(rows),
                "unique_calls": len(rows),
                "physical_calls": 0,
                "checkpoint_hits": len(rows),
            },
            "provenance": {"synthetic": True},
            "runtime_identity_sha256": _digest("synthetic-runtime"),
            "prompt_population": verified.prompt_population.model_dump(),
        },
        "physical_provider_calls_during_materialization": 0,
    }
    completion, _ = publish_sealed_json(
        fixture.root / "s0-completion.json",
        _sealed(body, "completion_artifact_identity_sha256"),
    )
    return prompt, completion, lifecycle_sha, release_sha, predictions


def _kwargs(
    fixture: Fixture,
    prompt: SealedJson,
    completion: SealedJson,
    lifecycle_sha: str,
    release_sha: str,
) -> dict[str, object]:
    return {
        **fixture.kwargs(),
        "s0_prompt_path": prompt.path,
        "expected_s0_prompt_sha256": prompt.sha256,
        "s0_completion_path": completion.path,
        "expected_s0_completion_sha256": completion.sha256,
        "expected_s0_lifecycle_preflight_sha256": lifecycle_sha,
        "expected_s0_provider_release_sha256": release_sha,
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {key for child in value for key in _all_keys(child)}
    return set()


def _query_namespaces(
    plane: parent.ProtectedS0AnswerPlane,
) -> dict[str, FrozenSourceNamespace]:
    namespaces: dict[str, FrozenSourceNamespace] = {}
    result: dict[str, FrozenSourceNamespace] = {}
    for index, row in enumerate(plane.payload["ordered_rows"]):
        namespace_id = row["namespace_id"]
        namespace = namespaces.get(namespace_id)
        if namespace is None:
            namespace = FrozenSourceNamespace(
                snapshot_id=plane.source_population.snapshot.snapshot_id,
                combined_store_receipt_sha256=_digest(
                    f"combined-store:{namespace_id}"
                ),
                sources=(
                    FrozenSourceMembership(
                        source_id=f"partition-{len(namespaces)}::history",
                        content_chunk_ids=(f"content-{len(namespaces)}",),
                        metadata_chunk_ids=(),
                        stream_sha256=_digest(f"stream:{namespace_id}"),
                    ),
                ),
            )
            namespaces[namespace_id] = namespace
        result[row["question_id"]] = namespace
    return result


def test_materializes_replays_and_feeds_exact_query_expansion_type(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "complete",
        semantics=(3, 1, 4, 2),
        id_prefix="arbitrary",
        namespace_sizes=(1, 3),
    )
    prompt, completion, lifecycle_sha, release_sha, predictions = (
        _completion_fixture(fixture)
    )
    kwargs = _kwargs(fixture, prompt, completion, lifecycle_sha, release_sha)
    artifact, created, plane = parent.publish_protected_s0_answer_plane(
        fixture.root / "protected-s0.json", **kwargs
    )

    assert created is True
    assert type(plane.query_expansion_source) is MatchedS0Population
    assert plane.predictions == predictions
    assert plane.payload["cumulative_stage_ids"] == [
        "causal_graph_coverage_predecessor",
        "direct_episode_additions",
        "representative_episode_additions",
        "artifact_global_closure_additions",
    ]
    assert plane.payload["physical_provider_calls_during_materialization"] == 0
    assert "ordinal" not in _all_keys(plane.payload)

    query_population = build_query_expansion_population(
        plane.query_expansion_source,
        namespaces_by_question=_query_namespaces(plane),
        include_s0_evidence=True,
    )
    assert len(query_population.rows) == 4
    assert query_population.source_population is plane.query_expansion_source

    same, created_again, _ = parent.publish_protected_s0_answer_plane(
        fixture.root / "protected-s0.json", **kwargs
    )
    assert created_again is False
    assert same.sha256 == artifact.sha256

    replay, replay_created, replayed_plane = (
        parent.replay_protected_s0_answer_plane(
            source_plane_path=artifact.path,
            expected_source_plane_sha256=artifact.sha256,
            replay_output_path=fixture.root / "protected-s0-replay.json",
            **kwargs,
        )
    )
    assert replay_created is True
    assert replay.sha256 == artifact.sha256
    assert replay.payload == artifact.payload == replayed_plane.payload


@pytest.mark.parametrize("mutation", ["missing", "reordered", "tampered"])
def test_incomplete_reordered_or_tampered_completion_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_fixture(
        tmp_path / mutation,
        semantics=(0, 1, 2),
        id_prefix=mutation,
        namespace_sizes=(2, 1),
    )
    prompt, completion, lifecycle_sha, release_sha, _ = _completion_fixture(fixture)
    changed = copy.deepcopy(completion.payload)
    if mutation == "missing":
        changed["ordered_rows"] = changed["ordered_rows"][:-1]
    elif mutation == "reordered":
        changed["ordered_rows"] = list(reversed(changed["ordered_rows"]))
    else:
        row = changed["ordered_rows"][1]
        row["completion"] = "tampered answer"
        row["completion_sha256"] = quote_sha256(row["completion"])
        row_body = dict(row)
        row_body.pop("completion_row_receipt_sha256")
        row["completion_row_receipt_sha256"] = canonical_sha256(row_body)
        changed["completion_batch"]["logical_completions"][1] = row["completion"]
    body = dict(changed)
    body.pop("completion_artifact_identity_sha256")
    changed["completion_artifact_identity_sha256"] = canonical_sha256(body)
    mutated, _ = publish_sealed_json(
        fixture.root / f"s0-completion-{mutation}.json", changed
    )
    kwargs = _kwargs(fixture, prompt, mutated, lifecycle_sha, release_sha)
    with pytest.raises(parent.ConfirmationProtectedS0Error, match="incomplete|row|record"):
        parent.build_protected_s0_answer_plane(**kwargs)


def test_wrong_lifecycle_binding_and_label_bearing_completion_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "bindings",
        semantics=(8, 9),
        id_prefix="bindings",
        namespace_sizes=(2,),
    )
    prompt, completion, lifecycle_sha, release_sha, _ = _completion_fixture(fixture)
    kwargs = _kwargs(fixture, prompt, completion, lifecycle_sha, release_sha)
    kwargs["expected_s0_lifecycle_preflight_sha256"] = _digest("wrong")
    with pytest.raises(parent.ConfirmationProtectedS0Error, match="envelope"):
        parent.build_protected_s0_answer_plane(**kwargs)

    changed = copy.deepcopy(completion.payload)
    changed["reference_answer"] = "forbidden"
    body = dict(changed)
    body.pop("completion_artifact_identity_sha256")
    changed["completion_artifact_identity_sha256"] = canonical_sha256(body)
    label_bearing, _ = publish_sealed_json(
        fixture.root / "s0-completion-label.json", changed
    )
    kwargs = _kwargs(fixture, prompt, label_bearing, lifecycle_sha, release_sha)
    with pytest.raises((ValueError, parent.ConfirmationProtectedS0Error)):
        parent.build_protected_s0_answer_plane(**kwargs)


def _actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    result = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                result.extend(_actions(child))
    return result


def test_cli_has_no_provider_execution_surface() -> None:
    destinations = {action.dest for action in _actions(parent.build_parser())}
    assert not destinations & {
        "api_key",
        "authorized_provider_calls",
        "enable_provider",
        "execute",
        "provider",
        "retry",
        "token",
    }
    source = Path(parent.__file__).read_text(encoding="utf-8").casefold()
    assert "import litellm" not in source
    assert "import openai" not in source
