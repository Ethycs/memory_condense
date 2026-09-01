from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.typed_memory_final_arm import (
    COMPOSITION_FORMAT,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    VALIDATION_CONTRACT_FORMAT,
    VALIDATOR_POLICY_FORMAT,
)
from tools.matched_eval.typed_memory_final_judging import (
    JUDGE_FORMAT,
    SCORE_FORMAT,
)
import tools.run_locked_typed_memory_final_arm as typed_cli
import tools.run_locked_typed_memory_posthoc_miss_subset as subset


def _sha(value: str) -> str:
    return quote_sha256(value)


def _validation_contract() -> dict[str, Any]:
    return {
        "answer_shape": "direct",
        "by_handle": {"H001": {}},
        "cardinality": None,
        "comparison_mode": "none",
        "deterministic_execution_advisory": None,
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": False,
        "operation": "single_supported_fact",
        "operator_spec_receipt_sha256": _sha("spec"),
        "packet_receipt_sha256": _sha("packet"),
        "question_action_concepts": [],
        "question_terms": ["question"],
        "required_slot_ids": [],
        "required_slots": [],
        "requires_all_slots": True,
        "scalar_validation_advisory": None,
        "temporal_mode": "none",
    }


def _prompt_row(ordinal: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "Use only the supplied typed evidence."},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "dated_question": f"Question {ordinal}",
                    "typed_evidence": [{"handle": "H001", "summary": "blue"}],
                },
                sort_keys=True,
            ),
        },
    ]
    body = {
        "allowed_handle_ids": ["H001"],
        "composition_row_sha256": _sha(f"composition-row-{ordinal}"),
        "dated_question_sha256": _sha(f"dated-{ordinal}"),
        "handle_group_by_id": {"H001": "G001"},
        "messages": messages,
        "messages_sha256": identity_sha256(messages),
        "ordinal": ordinal,
        "parent_prediction": f"parent answer {ordinal}",
        "preservation_requirements": {
            "by_handle": {},
            "question_required_terms": [],
        },
        "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
        "question_id": f"question-{ordinal:03d}",
        "question_sha256": _sha(f"question-{ordinal}"),
        "route_id": "direct",
        "story_coherence": {"incompatible_group_pairs": []},
        "typed_composition_receipt_sha256": _sha(f"typed-{ordinal}"),
        "validation_contract": _validation_contract(),
    }
    return {**body, "prompt_row_receipt_sha256": identity_sha256(body)}


def _sealed_inputs(
    tmp_path: Path,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
]:
    source_root = tmp_path / "source"
    authority_root = tmp_path / "authority"
    composition, _ = publish_sealed_json(
        source_root / typed_cli.COMPOSITION_NAME,
        {"format": COMPOSITION_FORMAT, "questions": []},
    )
    rows = tuple(_prompt_row(ordinal) for ordinal in range(100))
    prompts = tuple(tuple(row["messages"]) for row in rows)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    source_payload = {
        "composition_artifact_sha256": composition.sha256,
        "format": typed_cli.PREFLIGHT_FORMAT,
        "gateway_url": "http://sealed-gateway",
        "gold_loaded": False,
        "hard_prompt_token_cap": 8_000,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": 3,
        "model": "terra-model",
        "observed_max_complete_envelope_tokens": max(
            row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in rows
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": list(rows),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": 100,
        "required_authorized_provider_calls": 100,
        "retained_transformer_token_state_bytes": 0,
        "source_hash_bindings": {"fixture_source_sha256": _sha("source")},
    }
    source, _ = publish_sealed_json(
        source_root / typed_cli.PREFLIGHT_NAME,
        source_payload,
    )

    judge_rows = []
    for ordinal, source_row in enumerate(rows):
        body = {
            "correct": ordinal not in subset.MISS_ORDINALS,
            "dated_question_sha256": source_row["dated_question_sha256"],
            "ordinal": ordinal,
            "question_id": source_row["question_id"],
            "question_sha256": source_row["question_sha256"],
            "route_id": source_row["route_id"],
        }
        judge_rows.append({**body, "judge_row_sha256": identity_sha256(body)})
    parent_run_sha = _sha("compact-parent-run")
    judge_payload = {
        "aggregate": {
            "accuracy": 0.73,
            "correct": 73,
            "question_count": 100,
        },
        "format": JUDGE_FORMAT,
        "gold_loaded": True,
        "judge_mode": "full100",
        "question_count": 100,
        "questions": judge_rows,
        "selected_question_count": 100,
        "typed_final_run_sha256": parent_run_sha,
    }
    judge, _ = publish_sealed_json(
        authority_root / "judge.json",
        judge_payload,
    )
    score, _ = publish_sealed_json(
        authority_root / "score.json",
        {
            "correct": 73,
            "format": SCORE_FORMAT,
            "judge_mode": "full100",
            "question_count": 100,
            "selected_accuracy": 0.73,
            "selected_question_count": 100,
            "typed_final_run_sha256": parent_run_sha,
        },
    )
    return source, composition, judge, score, rows


def _preflight_args(
    tmp_path: Path,
) -> tuple[SimpleNamespace, tuple[dict[str, Any], ...]]:
    source, composition, judge, score, rows = _sealed_inputs(tmp_path)
    return (
        SimpleNamespace(
            expected_selection_judge_sha256=judge.sha256,
            expected_selection_score_sha256=score.sha256,
            expected_source_preflight_sha256=source.sha256,
            output_root=tmp_path / "subset",
            selection_judge=judge.path,
            selection_score=score.path,
            source_composition=composition.path,
            source_preflight=source.path,
        ),
        rows,
    )


def _fake_batch(rows: tuple[dict[str, Any], ...]) -> SimpleNamespace:
    completions = tuple(
        json.dumps(
            {
                "decision": "replace",
                "prediction": row["parent_prediction"],
                "used_handle_ids": ["H001"],
            }
        )
        for row in rows
    )
    records = tuple(
        SimpleNamespace(
            call_key_sha256=_sha(f"call-{ordinal}"),
            checkpoint_hit=True,
            completion=completion,
            completion_sha256=_sha(completion),
            messages_sha256=row["messages_sha256"],
            physical_call=False,
            request_journal_sha256=_sha(f"request-{ordinal}"),
            response_journal_sha256=_sha(f"response-{ordinal}"),
        )
        for ordinal, row, completion in zip(
            subset.MISS_ORDINALS,
            rows,
            completions,
            strict=True,
        )
    )
    usage = SimpleNamespace(
        checkpoint_hits=27,
        logical_calls=27,
        physical_calls=0,
        unique_calls=27,
    )
    return SimpleNamespace(
        logical_completions=completions,
        model_dump=lambda: {
            "logical_completions": list(completions),
            "unique_records": [],
            "usage": {
                "checkpoint_hits": 27,
                "logical_calls": 27,
                "physical_calls": 0,
                "unique_calls": 27,
            },
        },
        unique_records=records,
        usage=usage,
    )


def test_preflight_seals_exact_outcome_plan_and_gold_free_prompt_view(
    tmp_path: Path,
) -> None:
    args, source_rows = _preflight_args(tmp_path)
    result = subset._preflight(args)
    plan = read_sealed_json(Path(args.output_root) / subset.SELECTION_PLAN_NAME)
    preflight = read_sealed_json(Path(args.output_root) / subset.PREFLIGHT_NAME)

    assert result["physical_provider_calls"] == 0
    assert result["question_count"] == 27
    assert plan.payload["selected_ordinals"] == list(subset.MISS_ORDINALS)
    assert plan.payload["selection_is_posthoc_outcome_conditioned"] is True
    assert plan.payload["provider_and_materialization_read_this_artifact"] is False
    assert preflight.payload["selection_plan_artifact_sha256"] == plan.sha256
    assert preflight.payload["original_ordinals"] == list(subset.MISS_ORDINALS)
    assert preflight.payload["required_authorized_provider_calls"] == 27
    assert preflight.payload["hard_prompt_token_cap"] == 8_000
    assert preflight.payload["retained_transformer_token_state_bytes"] == 0
    selected = preflight.payload["physical_prompt_rows"]
    assert selected == [source_rows[ordinal] for ordinal in subset.MISS_ORDINALS]
    assert [row["ordinal"] for row in selected] == list(subset.MISS_ORDINALS)
    assert [
        row["prompt_row_receipt_sha256"] for row in selected
    ] == preflight.payload["source_prompt_row_receipt_sha256s"]
    assert preflight.payload["prompt_population"]["logical_prompt_count"] == 27
    assert preflight.payload["prompt_population"]["unique_prompt_count"] == 27
    assert_gold_blind(preflight.payload)
    provider_surface = json.dumps(
        [row["messages"] for row in selected],
        sort_keys=True,
    ).casefold()
    assert '"reference"' not in provider_surface
    assert '"gold"' not in provider_surface


def test_provider_requires_exact_27_before_environment_and_has_no_authority_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_args, _rows = _preflight_args(tmp_path)
    result = subset._preflight(preflight_args)
    parsed = subset._parser().parse_args(
        [
            "provider-run",
            "--output-root",
            str(preflight_args.output_root),
            "--expected-preflight-sha256",
            result["preflight_sha256"],
        ]
    )
    assert not hasattr(parsed, "selection_judge")
    assert not hasattr(parsed, "selection_score")

    monkeypatch.setattr(
        subset,
        "load_dotenv",
        lambda: pytest.fail("authorization reached environment"),
    )
    parsed.enable_provider = True
    parsed.authorized_provider_calls = 26
    with pytest.raises(
        subset.LockedTypedMemoryPosthocSubsetError,
        match="exact authorization for 27",
    ):
        subset._provider(parsed)


def test_materialize_replay_and_public_reader_need_only_subset_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_args, source_rows = _preflight_args(tmp_path)
    preflight_result = subset._preflight(preflight_args)
    output_root = Path(preflight_args.output_root)
    preflight = read_sealed_json(output_root / subset.PREFLIGHT_NAME)
    _prompts, selected_rows = subset._validate_subset_preflight(preflight)
    assert selected_rows == tuple(
        source_rows[ordinal] for ordinal in subset.MISS_ORDINALS
    )
    batch = _fake_batch(selected_rows)
    monkeypatch.setattr(subset, "_checkpoint_batch", lambda *_args, **_kwargs: batch)

    allowed_reads = {
        subset.PREFLIGHT_NAME,
        subset.RUN_NAME,
        subset.REPLAY_NAME,
    }
    original_read = subset.read_sealed_json
    observed_reads: list[str] = []

    def guarded_read(path: str | Path):
        name = Path(path).name
        observed_reads.append(name)
        assert name in allowed_reads
        return original_read(path)

    monkeypatch.setattr(subset, "read_sealed_json", guarded_read)
    materialized = subset._materialize(
        SimpleNamespace(
            expected_preflight_sha256=preflight_result["preflight_sha256"],
            output_root=output_root,
        )
    )
    run = original_read(output_root / subset.RUN_NAME)
    assert run.payload["original_ordinals"] == list(subset.MISS_ORDINALS)
    assert [row["ordinal"] for row in run.payload["questions"]] == list(
        subset.MISS_ORDINALS
    )
    assert [row["ordinal"] for row in run.payload["judge_rows"]] == list(
        subset.MISS_ORDINALS
    )
    assert run.payload["physical_provider_calls_during_materialization"] == 0
    assert run.payload["validator_policy_format"] == VALIDATOR_POLICY_FORMAT
    assert all(
        row["validation_basis"] == "normalized_identical_replace"
        for row in run.payload["questions"]
    )

    replayed = subset._replay(
        SimpleNamespace(
            expected_preflight_sha256=preflight_result["preflight_sha256"],
            expected_run_sha256=materialized["run_sha256"],
            output_root=output_root,
        )
    )
    verified_run, verified_replay, judge_rows = subset.read_verified_subset_run(
        output_root,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_run_sha256=materialized["run_sha256"],
        expected_replay_sha256=replayed["replay_sha256"],
    )
    assert verified_run.sha256 == materialized["run_sha256"]
    assert verified_replay.payload["byte_identical"] is True
    assert tuple(row["ordinal"] for row in judge_rows) == subset.MISS_ORDINALS
    assert set(observed_reads) <= allowed_reads
    assert subset.SELECTION_PLAN_NAME not in observed_reads
    assert "judge.json" not in observed_reads
    assert "score.json" not in observed_reads


def test_output_root_must_be_distinct_from_source_and_authority_roots(
    tmp_path: Path,
) -> None:
    args, _rows = _preflight_args(tmp_path)
    args.output_root = Path(args.source_preflight).parent
    with pytest.raises(
        subset.LockedTypedMemoryPosthocSubsetError,
        match="output root must be distinct",
    ):
        subset._preflight(args)


def test_cli_exposes_exact_four_phase_artifact_lifecycle() -> None:
    parser = subset._parser()
    preflight = parser.parse_args(["preflight"])
    provider = parser.parse_args(
        ["provider-run", "--expected-preflight-sha256", "a" * 64]
    )
    materialize = parser.parse_args(
        ["materialize", "--expected-preflight-sha256", "a" * 64]
    )
    replay = parser.parse_args(
        [
            "replay",
            "--expected-preflight-sha256",
            "a" * 64,
            "--expected-run-sha256",
            "b" * 64,
        ]
    )
    assert [
        preflight.command,
        provider.command,
        materialize.command,
        replay.command,
    ] == ["preflight", "provider-run", "materialize", "replay"]
    assert subset.SELECTION_PLAN_NAME.endswith("selection-plan-v1.json")
    assert subset.PREFLIGHT_NAME.endswith("preflight-v1.json")
    assert subset.RUN_NAME.endswith("run-v1.json")
    assert subset.REPLAY_NAME.endswith("replay-v1.json")
