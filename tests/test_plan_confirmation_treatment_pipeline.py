from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import plan_confirmation_treatment_pipeline as planner
from tools.v4_population_firebreak.canonical import canonical_sha256
from tools.confirmation_treatment import (
    ConfirmationTreatmentInput,
    TreatmentQuestion,
    TreatmentSample,
)


def _samples(count: int) -> tuple[TreatmentSample, ...]:
    result: list[TreatmentSample] = []
    for index in range(count):
        question_id = f"synthetic-row-{index}"
        result.append(
            TreatmentSample(
                sample_id=question_id,
                turns=(
                    ("user", f"memory request {index}"),
                    ("assistant", f"memory fact {index}"),
                ),
                turn_source_ids=(f"source-{index}-a", f"source-{index}-b"),
                turn_created_at=(
                    datetime(2026, 1, index + 1, tzinfo=timezone.utc),
                    datetime(2026, 1, index + 1, 1, tzinfo=timezone.utc),
                ),
                questions=(
                    TreatmentQuestion(
                        question_id=question_id,
                        question=f"What was memory fact {index}?",
                        question_date=f"2026-02-{index + 1:02d}T00:00:00Z",
                    ),
                ),
            )
        )
    return tuple(result)


def _projection(sample: TreatmentSample) -> dict[str, object]:
    question = sample.questions[0]
    return {
        "sample_id": sample.sample_id,
        "turns": [list(turn) for turn in sample.turns],
        "turn_source_ids": list(sample.turn_source_ids),
        "turn_created_at": [
            value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if value is not None
            else None
            for value in sample.turn_created_at
        ],
        "questions": [
            {
                "question_id": question.question_id,
                "question": question.question,
                "question_date": question.question_date,
            }
        ],
    }


def _treatment(
    samples: tuple[TreatmentSample, ...], *, tag: str = "base"
) -> ConfirmationTreatmentInput:
    projections = [_projection(sample) for sample in samples]
    return ConfirmationTreatmentInput(
        file_sha256=canonical_sha256({"file": tag, "samples": projections}),
        sanitized_projection_sha256=canonical_sha256(projections),
        dataset_sha256=canonical_sha256({"dataset": "synthetic"}),
        split_manifest_sha256=canonical_sha256({"split": "synthetic"}),
        ordered_question_ids_sha256=canonical_sha256(
            [sample.sample_id for sample in samples]
        ),
        ordered_normalized_sample_bindings_sha256=canonical_sha256(
            {"normalized": tag}
        ),
        ordered_raw_record_bindings_sha256=canonical_sha256({"raw": tag}),
        samples=samples,
    )


def _stage(plan: dict[str, object], stage_id: str) -> dict[str, object]:
    stages = plan["stages"]
    assert isinstance(stages, list)
    return next(stage for stage in stages if stage["stage_id"] == stage_id)


def test_arbitrary_population_and_namespace_schedule_has_only_exact_known_calls(
) -> None:
    treatment = _treatment(_samples(5))
    plan = planner.compile_confirmation_pipeline_preflight(
        treatment, namespace_sizes=(2, 3)
    )

    assert plan["question_count"] == 5
    assert plan["namespace_count"] == 2
    assert plan["namespace_sizes"] == [2, 3]
    assert [
        row["question_id"] for row in plan["rows"]  # type: ignore[index]
    ] == [sample.sample_id for sample in treatment.samples]
    assert plan["physical_provider_calls"] == 0
    assert plan["provider_execution_available"] is False
    assert plan["known_would_call_count"] == 5
    assert plan["total_would_call_count_exact"] is False
    assert _stage(plan, "treatment_verification")["would_call_count"] == 0
    assert (
        _stage(plan, "memory_materialization_and_retrieval")["would_call_count"]
        == 0
    )
    assert _stage(plan, "upstream_parent_synthesis")["would_call_count"] is None
    assert _stage(plan, "terminal_synthesis")["would_call_count"] is None
    assert (
        _stage(plan, "official_full_population_judge")["would_call_count"] == 5
    )
    planner.assert_plan_gold_blind(plan)

    rendered = json.dumps(plan, sort_keys=True)
    assert "What was memory fact" not in rendered
    assert "memory request" not in rendered
    assert "reference_answer" not in rendered
    with pytest.raises(
        planner.ConfirmationPipelinePlanError, match="label-bearing"
    ):
        planner.assert_plan_gold_blind({"nested": {"reference_answer": "leak"}})


def test_uniform_schedule_allows_a_partial_final_namespace() -> None:
    assert planner.uniform_namespace_sizes(5, 2) == (2, 2, 1)
    plan = planner.compile_uniform_confirmation_pipeline_preflight(
        _treatment(_samples(5)), questions_per_namespace=2
    )
    assert plan["namespace_sizes"] == [2, 2, 1]
    assert plan["namespace_count"] == 3


def test_planner_import_does_not_load_tokenizer_or_provider_runtime() -> None:
    repository = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tools.plan_confirmation_treatment_pipeline; "
                "bad=[n for n in sys.modules if n == 'tiktoken' "
                "or n.endswith('._tokenizer') or 'llm_provider' in n]; "
                "assert not bad, bad"
            ),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_deterministic_seal_is_no_clobber_with_digest_sidecar(tmp_path: Path) -> None:
    treatment = _treatment(_samples(4))
    output = tmp_path / "preflight.json"

    first, first_created = planner.publish_confirmation_pipeline_preflight(
        output, treatment, namespace_sizes=(2, 2)
    )
    second, second_created = planner.publish_confirmation_pipeline_preflight(
        output, treatment, namespace_sizes=(2, 2)
    )

    assert first_created is True
    assert second_created is False
    assert first.sha256 == second.sha256
    assert planner.read_sealed_confirmation_pipeline_plan(output).sha256 == first.sha256
    assert (
        planner.read_sealed_confirmation_pipeline_plan(
            output, expected_sha256=first.sha256
        ).sha256
        == first.sha256
    )
    with pytest.raises(
        planner.ConfirmationPipelineSealError,
        match="does not match expected sha256",
    ):
        planner.read_sealed_confirmation_pipeline_plan(
            output, expected_sha256="0" * 64
        )
    assert output.with_name(output.name + ".sha256").is_file()

    changed = _treatment(_samples(5), tag="expanded")
    with pytest.raises(
        planner.ConfirmationPipelineSealError, match="refusing to replace"
    ):
        planner.publish_confirmation_pipeline_preflight(
            output, changed, namespace_sizes=(2, 3)
        )


def test_id_renumbering_cannot_change_policy_content_or_call_decisions() -> None:
    original_samples = _samples(4)
    renamed_samples = tuple(
        replace(
            sample,
            sample_id=f"foreign-id-{index}",
            questions=(
                replace(sample.questions[0], question_id=f"foreign-id-{index}"),
            ),
        )
        for index, sample in enumerate(original_samples)
    )
    original = planner.compile_confirmation_pipeline_preflight(
        _treatment(original_samples), namespace_sizes=(2, 2)
    )
    renamed = planner.compile_confirmation_pipeline_preflight(
        _treatment(renamed_samples, tag="renamed"), namespace_sizes=(2, 2)
    )

    assert original["policy"] == renamed["policy"]
    assert original["stages"] == renamed["stages"]
    assert [row["content_binding_sha256"] for row in original["rows"]] == [
        row["content_binding_sha256"] for row in renamed["rows"]
    ]
    assert [row["question_id"] for row in original["rows"]] != [
        row["question_id"] for row in renamed["rows"]
    ]
    assert [row["namespace_id"] for row in original["namespaces"]] == [
        row["namespace_id"] for row in renamed["namespaces"]
    ]


def test_namespace_block_permutation_only_reorders_the_schedule() -> None:
    samples = _samples(6)
    permuted = samples[4:6] + samples[0:2] + samples[2:4]
    original = planner.compile_confirmation_pipeline_preflight(
        _treatment(samples), namespace_sizes=(2, 2, 2)
    )
    reordered = planner.compile_confirmation_pipeline_preflight(
        _treatment(permuted, tag="permuted"), namespace_sizes=(2, 2, 2)
    )

    assert original["policy"] == reordered["policy"]
    assert original["stages"] == reordered["stages"]
    assert sorted(row["row_receipt_sha256"] for row in original["rows"]) == sorted(
        row["row_receipt_sha256"] for row in reordered["rows"]
    )
    assert sorted(
        row["namespace_receipt_sha256"] for row in original["namespaces"]
    ) == sorted(row["namespace_receipt_sha256"] for row in reordered["namespaces"])
    assert [row["namespace_id"] for row in original["namespaces"]] != [
        row["namespace_id"] for row in reordered["namespaces"]
    ]


def test_adding_a_foreign_namespace_does_not_rewrite_existing_work() -> None:
    base_samples = _samples(4)
    expanded_samples = _samples(6)
    base = planner.compile_confirmation_pipeline_preflight(
        _treatment(base_samples), namespace_sizes=(2, 2)
    )
    expanded = planner.compile_confirmation_pipeline_preflight(
        _treatment(expanded_samples, tag="expanded"), namespace_sizes=(2, 2, 2)
    )

    assert base["policy"] == expanded["policy"]
    assert base["rows"] == expanded["rows"][:4]
    assert base["namespaces"] == expanded["namespaces"][:2]
    assert expanded["known_would_call_count"] == 6
    assert _stage(expanded, "official_full_population_judge")["would_call_count"] == 6


def test_duplicate_namespace_content_is_disambiguated_without_id_branching() -> None:
    originals = _samples(2)
    duplicates = tuple(
        replace(
            sample,
            sample_id=f"duplicate-{index}",
            questions=(
                replace(sample.questions[0], question_id=f"duplicate-{index}"),
            ),
        )
        for index, sample in enumerate(originals)
    )
    plan = planner.compile_confirmation_pipeline_preflight(
        _treatment(originals + duplicates, tag="duplicates"),
        namespace_sizes=(2, 2),
    )

    namespaces = plan["namespaces"]
    assert namespaces[0]["content_population_sha256"] == namespaces[1][
        "content_population_sha256"
    ]
    assert [row["content_occurrence"] for row in namespaces] == [0, 1]
    assert len({row["namespace_id"] for row in namespaces}) == 2


@pytest.mark.parametrize("sizes", [(), (0, 4), (3,), (2, 2, 1)])
def test_namespace_schedule_must_be_positive_and_exact(sizes: tuple[int, ...]) -> None:
    with pytest.raises(planner.ConfirmationPipelinePlanError, match="namespace"):
        planner.compile_confirmation_pipeline_preflight(
            _treatment(_samples(4)), namespace_sizes=sizes
        )


def test_rebased_or_untyped_treatment_is_rejected() -> None:
    treatment = _treatment(_samples(3))
    with pytest.raises(planner.ConfirmationPipelinePlanError, match="projection"):
        planner.compile_confirmation_pipeline_preflight(
            replace(treatment, sanitized_projection_sha256="0" * 64),
            namespace_sizes=(3,),
        )
    with pytest.raises(planner.ConfirmationPipelinePlanError, match="must be"):
        planner.compile_confirmation_pipeline_preflight(  # type: ignore[arg-type]
            {"samples": []}, namespace_sizes=(1,)
        )


def test_cli_exposes_planning_only_not_provider_or_population_routing() -> None:
    parser = planner.build_parser()
    options = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert "--ordinal" not in options
    assert "--question-id" not in options
    assert "--enable-provider" not in options
    assert "--authorized-provider-calls" not in options
    assert "--treatment-input" in options
    assert "--questions-per-namespace" in options
