#!/usr/bin/env python3
"""Build the gold-blind specialist successor to the locked 73/100 run.

The command verifies the compact typed-memory composition, its sealed full-
store cache receipts, and the replayed terminal answer population before doing
any specialist work.  It then streams exactly one namespace index at a time.
Question text alone decides whether a specialist is eligible.  Questions with
no routed specialist, or with no terminal specialist advisory after fitting,
are explicit byte-bound passthroughs to the replay-verified parent prediction.
No model provider is imported or called by this module.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_typed_memory_final_arm as typed_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_specialist_retrieval_assay as specialist  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    build_full_store_window_index,
)
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    PROMPT_FORMAT as SPECIALIST_PROMPT_FORMAT,
    render_specialist_scoped_prompt,
)
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    load_verified_typed_final_judge_source,
)


FORMAT = "memory-condense-locked-specialist-final-v1"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction"
CONSTRUCTION_NAME = "locked-specialist-final-construction-v1.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/typed-memory-final-v2-compact-budget"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-v1"
)

EXPECTED_PARENT_COMPOSITION_SHA256 = (
    "21be1ebfe628eae55dd543312e59c315f08de298b9d1895fc757b6517f869933"
)
EXPECTED_PARENT_CLOSURE_SHA256 = (
    "044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4"
)
EXPECTED_PARENT_RUN_SHA256 = (
    "ce81033e0658fcf2706e95214cfe29323f4c84adb5ce3deb96f8da79ceb34907"
)
EXPECTED_PARENT_REPLAY_SHA256 = (
    "117ff8ea1d7f1745263ec90ae2d13ba13f2a9814defaac6bfb435c7421a82a61"
)
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
EXPECTED_NAMESPACE_COUNT = 10
ORDINALS = tuple(range(EXPECTED_QUESTION_COUNT))


class LockedSpecialistFinalConstructionError(MatchedEvalContractError):
    """A sealed parent, store, specialist, or construction invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalConstructionError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value


def _sealed_rows(
    artifact: SealedArtifact,
    *,
    expected_sha256: str,
    receipt_field: str,
    label: str,
) -> tuple[dict[str, Any], ...]:
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    rows = tuple(
        _exact_dict(value, f"{label} row")
        for value in _exact_list(artifact.payload.get("questions"), f"{label} questions")
    )
    _require(len(rows) == EXPECTED_QUESTION_COUNT, f"{label} population changed")
    for ordinal, row in enumerate(rows):
        body = dict(row)
        declared = require_sha256(body.pop(receipt_field, None), f"{label} row")
        _require(
            row.get("ordinal") == ordinal and identity_sha256(body) == declared,
            f"{label} row seal/order changed at ordinal {ordinal}",
        )
    return rows


def _load_parent_inputs(
    parent_root: Path,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    composition = read_sealed_json(parent_root / typed_cli.COMPOSITION_NAME)
    composition_rows = _sealed_rows(
        composition,
        expected_sha256=EXPECTED_PARENT_COMPOSITION_SHA256,
        receipt_field="composition_row_sha256",
        label="parent composition",
    )
    closure = read_sealed_json(parent_root / typed_cli.CLOSURE_INPUT_NAME)
    closure_rows = _sealed_rows(
        closure,
        expected_sha256=EXPECTED_PARENT_CLOSURE_SHA256,
        receipt_field="row_receipt_sha256",
        label="parent full-store input",
    )
    run, replay, judge_rows = load_verified_typed_final_judge_source(
        parent_root,
        expected_run_sha256=EXPECTED_PARENT_RUN_SHA256,
        expected_replay_sha256=EXPECTED_PARENT_REPLAY_SHA256,
    )
    run_rows = tuple(
        _exact_dict(value, "parent terminal source row")
        for value in _exact_list(run.payload.get("questions"), "parent terminal rows")
    )
    _require(
        len(run_rows) == len(judge_rows) == EXPECTED_QUESTION_COUNT
        and closure.payload.get("database_read_passes_per_unique_namespace") == 1
        and closure.payload.get("new_provider_calls") == 0
        and closure.payload.get("retained_transformer_token_state_bytes") == 0,
        "parent replay/full-store population changed",
    )
    for ordinal, (composition_row, closure_row, source_row, judge_row) in enumerate(
        zip(composition_rows, closure_rows, run_rows, judge_rows, strict=True)
    ):
        _require(
            composition_row.get("question_id")
            == closure_row.get("question_id")
            == source_row.get("question_id")
            == judge_row.get("question_id")
            and composition_row.get("question_sha256")
            == source_row.get("question_sha256")
            == judge_row.get("question_sha256")
            and composition_row.get("dated_question_sha256")
            == source_row.get("dated_question_sha256")
            == judge_row.get("dated_question_sha256")
            and source_row.get("source_row_sha256")
            == judge_row.get("source_row_sha256")
            and source_row.get("prediction") == judge_row.get("prediction")
            and source_row.get("prediction_sha256")
            == judge_row.get("prediction_sha256"),
            f"parent composition/closure/terminal binding changed at {ordinal}",
        )
    assert_gold_blind(composition.payload, path="locked_specialist_parent_composition")
    assert_gold_blind(closure.payload, path="locked_specialist_parent_closure")
    assert_gold_blind(run.payload, path="locked_specialist_parent_run")
    return (
        composition,
        closure,
        run,
        replay,
        composition_rows,
        closure_rows,
        run_rows,
        judge_rows,
    )


def _parent_source_projection(
    *,
    ordinal: int,
    run: SealedArtifact,
    replay: SealedArtifact,
    source_row: Mapping[str, Any],
    judge_row: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = require_text(judge_row.get("prediction"), "parent prediction")
    prediction_sha = require_sha256(
        judge_row.get("prediction_sha256"), "parent prediction"
    )
    _require(
        judge_row.get("ordinal") == ordinal
        and source_row.get("ordinal") == ordinal
        and prediction_sha == quote_sha256(prediction)
        and source_row.get("source_row_sha256") == judge_row.get("source_row_sha256"),
        f"parent terminal row changed at ordinal {ordinal}",
    )
    parent_judge_row = dict(judge_row)
    body = {
        "parent_judge_row": parent_judge_row,
        "parent_judge_row_sha256": identity_sha256(parent_judge_row),
        "prediction": prediction,
        "prediction_sha256": prediction_sha,
        "replay_artifact_sha256": replay.sha256,
        "run_artifact_sha256": run.sha256,
        "source_row_sha256": require_sha256(
            source_row.get("source_row_sha256"), "parent source row"
        ),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _passthrough_question(
    *,
    ordinal: int,
    namespace_id: str,
    composition_row: Mapping[str, Any],
    parent_source: Mapping[str, Any],
) -> dict[str, Any]:
    dated_question, _old_prediction, question_id = specialist._question_inputs(  # noqa: SLF001
        composition_row
    )
    body = {
        "applicable_specialist_ids": list(
            specialist.applicable_specialist_ids(dated_question)
        ),
        "dated_question_sha256": composition_row.get("dated_question_sha256"),
        "methods": [],
        "mode": "parent_passthrough",
        "namespace_id": namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_source": dict(parent_source),
        "question_id": question_id,
        "question_sha256": composition_row.get("question_sha256"),
        "retained_transformer_token_state_bytes": 0,
        "route": specialist.route_question(dated_question).identity_payload(),
        "terminal_prompt": None,
    }
    assert_gold_blind(body, path="locked_specialist_parent_passthrough")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _specialist_or_passthrough_question(
    *,
    ordinal: int,
    namespace_id: str,
    index: Any,
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    parent_source: Mapping[str, Any],
) -> dict[str, Any]:
    dated_question, _old_prediction, question_id = specialist._question_inputs(  # noqa: SLF001
        composition_row
    )
    if not specialist.applicable_specialist_ids(dated_question):
        return _passthrough_question(
            ordinal=ordinal,
            namespace_id=namespace_id,
            composition_row=composition_row,
            parent_source=parent_source,
        )
    generic = specialist._composed_question(  # noqa: SLF001
        ordinal=ordinal,
        index=index,
        composition_row=composition_row,
        parent_composition_artifact_sha256=composition_sha256,
        frozen_row={
            "namespace_id": namespace_id,
            "ordinal": ordinal,
            "question_id": question_id,
        },
        parent_prediction_override=require_text(
            parent_source.get("prediction"), "verified parent prediction"
        ),
    )
    terminal = _exact_dict(generic.get("terminal_prompt"), "generic terminal prompt")
    provider_input = _exact_dict(
        terminal.get("provider_input"), "generic terminal provider input"
    )
    advisories = _exact_list(
        provider_input.get("specialist_advisories"), "specialist advisories"
    )
    if not advisories:
        return _passthrough_question(
            ordinal=ordinal,
            namespace_id=namespace_id,
            composition_row=composition_row,
            parent_source=parent_source,
        )
    fitted_input = dict(provider_input)
    fitted_input.pop("specialist_advisories")
    scoped_terminal = specialist._terminal_projection(  # noqa: SLF001
        provider_input=fitted_input,
        specialist_advisories=advisories,
        fitted_prompt_receipt_sha256=require_sha256(
            terminal.get("fitted_prompt_receipt_sha256"), "fitted specialist prompt"
        ),
        message_renderer_format=SPECIALIST_PROMPT_FORMAT,
        prompt_envelope_renderer=render_specialist_scoped_prompt,
    )
    body = dict(generic)
    body.pop("question_receipt_sha256")
    body.update(
        {
            "mode": "specialist",
            "parent_source": dict(parent_source),
            "terminal_prompt": scoped_terminal,
        }
    )
    _require(
        scoped_terminal["full_chat_plus_output_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "specialist terminal prompt escaped 8k",
    )
    assert_gold_blind(body, path="locked_specialist_question")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _cache_receipts_by_namespace(closure: SealedArtifact) -> dict[str, dict[str, Any]]:
    rows = tuple(
        _exact_dict(value, "sealed cache receipt")
        for value in _exact_list(
            closure.payload.get("cache_receipts"), "sealed cache receipts"
        )
    )
    result = {
        require_sha256(row.get("namespace_id"), "sealed cache namespace"): row
        for row in rows
    }
    _require(
        len(rows) == len(result) == EXPECTED_NAMESPACE_COUNT,
        "sealed namespace cache population changed",
    )
    return result


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    (
        composition,
        closure,
        run,
        replay,
        composition_rows,
        _closure_rows,
        run_rows,
        judge_rows,
    ) = _load_parent_inputs(Path(args.parent_root))
    context = typed_cli._guided_context(reduced_cli._guided_args(args))  # noqa: SLF001
    context_by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    _require(
        len(context.population.namespaces) == EXPECTED_NAMESPACE_COUNT
        and len(context_by_question) == EXPECTED_QUESTION_COUNT,
        "locked query/store population changed",
    )
    ordinals_by_namespace: dict[str, list[int]] = defaultdict(list)
    for ordinal, composition_row in enumerate(composition_rows):
        question_id = require_text(
            composition_row.get("question_id"), "composition question ID"
        )
        population_row = context_by_question.get(question_id)
        _require(population_row is not None, "composition question left locked store")
        packet = population_row.source.packet
        _require(
            packet.question_sha256 == composition_row.get("question_sha256")
            and packet.dated_question_sha256
            == composition_row.get("dated_question_sha256"),
            f"locked question text changed at ordinal {ordinal}",
        )
        ordinals_by_namespace[population_row.namespace.namespace_id].append(ordinal)

    sealed_cache = _cache_receipts_by_namespace(closure)
    _require(
        set(ordinals_by_namespace) == set(sealed_cache),
        "locked questions changed namespace ownership",
    )
    questions_by_ordinal: dict[int, dict[str, Any]] = {}
    lifecycle: list[dict[str, Any]] = []
    for namespace in sorted(
        context.population.namespaces, key=lambda value: value.namespace_id
    ):
        namespace_id = namespace.namespace_id
        database_path = context.store_dirs_by_namespace[namespace_id] / "memory.db"
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=context.database_sha256_by_namespace[namespace_id],
                source_store_receipt_sha256=namespace.combined_store_receipt_sha256,
            )
        index = build_full_store_window_index(cache)
        sealed = sealed_cache[namespace_id]
        _require(
            sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
            and sealed.get("window_index_receipt_sha256") == index.receipt_sha256
            and sealed.get("content_row_count") == cache.content_row_count
            and sealed.get("physical_store_row_count") == cache.physical_store_row_count,
            f"streamed cache/index differs from sealed closure: {namespace_id}",
        )
        for ordinal in ordinals_by_namespace[namespace_id]:
            parent_source = _parent_source_projection(
                ordinal=ordinal,
                run=run,
                replay=replay,
                source_row=run_rows[ordinal],
                judge_row=judge_rows[ordinal],
            )
            questions_by_ordinal[ordinal] = _specialist_or_passthrough_question(
                ordinal=ordinal,
                namespace_id=namespace_id,
                index=index,
                composition_row=composition_rows[ordinal],
                composition_sha256=composition.sha256,
                parent_source=parent_source,
            )
        lifecycle.append(
            {
                "cache_receipt_sha256": cache.cache_receipt_sha256,
                "content_row_count": cache.content_row_count,
                "database_read_passes": 1,
                "namespace_id": namespace_id,
                "physical_content_token_count": index.physical_content_tokens_indexed,
                "physical_store_row_count": cache.physical_store_row_count,
                "window_index_receipt_sha256": index.receipt_sha256,
            }
        )
        del index, cache
        gc.collect()

    _require(
        set(questions_by_ordinal) == set(ORDINALS),
        "streamed construction lost or duplicated a question",
    )
    questions = [questions_by_ordinal[ordinal] for ordinal in ORDINALS]
    specialist_count = sum(row["mode"] == "specialist" for row in questions)
    passthrough_count = EXPECTED_QUESTION_COUNT - specialist_count
    complete_envelopes = [
        row["terminal_prompt"]["full_chat_plus_output_tokens"]
        for row in questions
        if row["mode"] == "specialist"
    ]
    payload: dict[str, Any] = {
        "bindings": {
            "parent_composition_artifact_sha256": composition.sha256,
            "parent_full_store_input_artifact_sha256": closure.sha256,
            "parent_replay_artifact_sha256": replay.sha256,
            "parent_run_artifact_sha256": run.sha256,
        },
        "construction_is_posthoc_outcome_conditioned": False,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_terminal_complete_envelope_tokens": max(complete_envelopes, default=0),
        "new_provider_calls": 0,
        "ordinals": list(ORDINALS),
        "parent_passthrough_count": passthrough_count,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": questions,
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "receipts": lifecycle,
            "total_database_read_passes": len(lifecycle),
            "unique_namespace_count": len(lifecycle),
        },
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "specialist_provider_prompt_count": specialist_count,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="locked_specialist_final_construction")
    payload["construction_identity_sha256"] = identity_sha256(payload)
    return payload


def _validate_parent_source(
    value: object,
    *,
    ordinal: int,
    bindings: Mapping[str, Any],
    question: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _exact_dict(value, "parent source")
    body = dict(parent)
    declared = require_sha256(body.pop("receipt_sha256", None), "parent source")
    judge = _exact_dict(parent.get("parent_judge_row"), "parent judge seam")
    prediction = require_text(parent.get("prediction"), "parent prediction")
    _require(
        identity_sha256(body) == declared
        and parent.get("run_artifact_sha256")
        == bindings.get("parent_run_artifact_sha256")
        and parent.get("replay_artifact_sha256")
        == bindings.get("parent_replay_artifact_sha256")
        and parent.get("parent_judge_row_sha256") == identity_sha256(judge)
        and parent.get("source_row_sha256") == judge.get("source_row_sha256")
        and parent.get("prediction_sha256") == quote_sha256(prediction)
        and parent.get("prediction") == judge.get("prediction")
        and parent.get("prediction_sha256") == judge.get("prediction_sha256")
        and judge.get("ordinal") == ordinal
        and judge.get("question_id") == question.get("question_id")
        and judge.get("question_sha256") == question.get("question_sha256")
        and judge.get("dated_question_sha256")
        == question.get("dated_question_sha256"),
        f"parent judge seam changed at ordinal {ordinal}",
    )
    return parent


def validate_construction(
    artifact: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = tuple(
        _exact_dict(value, "specialist construction row")
        for value in _exact_list(payload.get("questions"), "specialist questions")
    )
    bindings = _exact_dict(payload.get("bindings"), "specialist bindings")
    lifecycle = _exact_dict(
        payload.get("resident_index_lifecycle"), "specialist lifecycle"
    )
    receipts = _exact_list(lifecycle.get("receipts"), "specialist index receipts")
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("construction_is_posthoc_outcome_conditioned") is False
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("selection_and_routing_frozen_before_target_plan_load") is True
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and tuple(payload.get("ordinals", ())) == ORDINALS
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and len(questions) == EXPECTED_QUESTION_COUNT
        and bindings
        == {
            "parent_composition_artifact_sha256": EXPECTED_PARENT_COMPOSITION_SHA256,
            "parent_full_store_input_artifact_sha256": EXPECTED_PARENT_CLOSURE_SHA256,
            "parent_replay_artifact_sha256": EXPECTED_PARENT_REPLAY_SHA256,
            "parent_run_artifact_sha256": EXPECTED_PARENT_RUN_SHA256,
        }
        and lifecycle.get("database_read_passes_per_used_namespace") == 1
        and lifecycle.get("maximum_simultaneous_namespace_indexes") == 1
        and lifecycle.get("total_database_read_passes") == EXPECTED_NAMESPACE_COUNT
        and lifecycle.get("unique_namespace_count") == EXPECTED_NAMESPACE_COUNT
        and len(receipts) == EXPECTED_NAMESPACE_COUNT
        and len({row.get("namespace_id") for row in receipts if type(row) is dict})
        == EXPECTED_NAMESPACE_COUNT,
        "locked specialist construction boundary changed",
    )
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("construction_identity_sha256", None),
        "specialist construction identity",
    )
    _require(identity_sha256(unsigned) == declared, "construction identity changed")
    specialist_count = 0
    terminal_tokens: list[int] = []
    question_ids: list[str] = []
    for ordinal, row in enumerate(questions):
        row_body = dict(row)
        row_declared = require_sha256(
            row_body.pop("question_receipt_sha256", None), "specialist question"
        )
        _require(
            identity_sha256(row_body) == row_declared
            and row.get("ordinal") == ordinal
            and type(row.get("applicable_specialist_ids")) is list
            and row.get("new_provider_calls") == 0
            and row.get("retained_transformer_token_state_bytes") == 0,
            f"specialist row seal/order changed at ordinal {ordinal}",
        )
        question_ids.append(require_text(row.get("question_id"), "question ID"))
        _validate_parent_source(
            row.get("parent_source"),
            ordinal=ordinal,
            bindings=bindings,
            question=row,
        )
        mode = row.get("mode")
        terminal = row.get("terminal_prompt")
        if mode == "parent_passthrough":
            _require(
                terminal is None and row.get("methods") == [],
                f"passthrough row exposed a provider prompt at ordinal {ordinal}",
            )
            continue
        _require(
            mode == "specialist"
            and bool(row.get("applicable_specialist_ids"))
            and type(terminal) is dict,
            f"specialist mode changed at ordinal {ordinal}",
        )
        assert type(terminal) is dict
        provider_input = _exact_dict(
            terminal.get("provider_input"), "specialist provider input"
        )
        advisories = _exact_list(
            provider_input.get("specialist_advisories"), "specialist advisories"
        )
        envelope = render_specialist_scoped_prompt(provider_input)
        _require(
            bool(advisories)
            and terminal.get("message_renderer_format") == SPECIALIST_PROMPT_FORMAT
            and terminal.get("messages_sha256")
            == identity_sha256(list(envelope.messages))
            and terminal.get("specialist_prompt_envelope_receipt_sha256")
            == envelope.receipt_sha256
            and terminal.get("prompt_token_proxy") == envelope.prompt_token_proxy
            and terminal.get("full_chat_plus_output_tokens")
            == envelope.prompt_token_proxy + specialist.OUTPUT_TOKEN_RESERVE
            and terminal.get("full_chat_plus_output_tokens")
            <= HARD_COMPLETE_CHAT_TOKEN_CAP,
            f"specialist prompt envelope changed at ordinal {ordinal}",
        )
        specialist_count += 1
        terminal_tokens.append(int(terminal["full_chat_plus_output_tokens"]))
    _require(
        len(set(question_ids)) == EXPECTED_QUESTION_COUNT
        and payload.get("specialist_provider_prompt_count") == specialist_count
        and payload.get("parent_passthrough_count")
        == EXPECTED_QUESTION_COUNT - specialist_count
        and payload.get("max_terminal_complete_envelope_tokens")
        == max(terminal_tokens, default=0),
        "specialist/passthrough accounting changed",
    )
    assert_gold_blind(payload, path="validated_locked_specialist_construction")
    return questions


def load_verified_construction(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    source = Path(path)
    if source.name != CONSTRUCTION_NAME:
        source = source / CONSTRUCTION_NAME
    artifact = read_sealed_json(source)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "locked specialist construction"),
        "locked specialist construction artifact changed",
    )
    return artifact, validate_construction(artifact)


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME, payload
    )
    validate_construction(artifact)
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "parent_passthrough_count": payload["parent_passthrough_count"],
        "question_count": EXPECTED_QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "specialist_provider_prompt_count": payload[
            "specialist_provider_prompt_count"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    specialist._add_store_args(parser)  # noqa: SLF001
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_construct(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "LockedSpecialistFinalConstructionError",
    "build_construction",
    "build_parser",
    "load_verified_construction",
    "main",
    "run_construct",
    "validate_construction",
]
