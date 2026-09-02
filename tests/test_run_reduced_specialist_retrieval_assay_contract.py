from __future__ import annotations

import json
from types import SimpleNamespace

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools import run_reduced_specialist_retrieval_assay as assay
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.numeric_operand_specialist import (
    MECHANISM_ID as NUMERIC_MECHANISM_ID,
    NumericOperandClosureResult,
    NumericOperandGroup,
)
from tools.matched_eval.profile_preference_specialist import (
    MECHANISM_ID as PROFILE_MECHANISM_ID,
)
from tools.matched_eval.temporal_insufficiency_specialist import (
    MECHANISM_ID as TEMPORAL_MECHANISM_ID,
    SpecialistRoute,
    TemporalEventBundle,
    TemporalInsufficiencyResult,
)
from tools.matched_eval.typed_memory_final_arm import render_final_messages


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _partial_result(result_type: type[object], **values: object) -> object:
    """Build only the exact result surface consumed by the advisory helper."""

    result = object.__new__(result_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _run(
    mechanism_id: str,
    result: object,
    rows: tuple[tuple[str, str, str], ...],
) -> assay._SpecialistRun:  # noqa: SLF001
    bindings = tuple(
        SimpleNamespace(handle_id=handle_id, source_group_handle=global_group)
        for _candidate_id, handle_id, global_group in rows
    )
    locals_ = tuple(
        SimpleNamespace(
            candidate_id=candidate_id,
            source_group_handle=f"G{offset + 1:04d}",
        )
        for offset, (candidate_id, _handle_id, _global_group) in enumerate(rows)
    )
    return assay._SpecialistRun(  # noqa: SLF001
        mechanism_id=mechanism_id,
        result=result,
        contribution=SimpleNamespace(bindings=bindings),
        local_bindings=locals_,
        provider_projection={},
        local_projection={},
        specialist_receipt_sha256=_sha(f"{mechanism_id}-receipt"),
    )


def test_specialist_ranges_follow_the_parent_stack_without_collision() -> None:
    assert assay.MECHANISM_HANDLE_START == {
        NUMERIC_MECHANISM_ID: 700_001,
        PROFILE_MECHANISM_ID: 800_001,
        TEMPORAL_MECHANISM_ID: 900_001,
    }
    assert {
        value // 100_000 for value in assay.MECHANISM_HANDLE_START.values()
    } == {7, 8, 9}


def test_advisories_close_over_only_fitted_handles_groups_and_bundle_members() -> None:
    numeric_groups = (
        NumericOperandGroup(
            operand_group_id=_sha("numeric-keep-group"),
            operation_mode="sum",
            action_class="purchase",
            entity_key="feed",
            operand_values=(5.0,),
            value_basis="explicit_numeric_mention",
            candidate_ids=("numeric-keep",),
            source_group_handles=("G0001",),
        ),
        NumericOperandGroup(
            operand_group_id=_sha("numeric-drop-group"),
            operation_mode="sum",
            action_class="purchase",
            entity_key="feed",
            operand_values=(8.0,),
            value_basis="explicit_numeric_mention",
            candidate_ids=("numeric-drop",),
            source_group_handles=("G0002",),
        ),
    )
    numeric_result = _partial_result(
        NumericOperandClosureResult,
        operand_groups=numeric_groups,
    )
    numeric_run = _run(
        NUMERIC_MECHANISM_ID,
        numeric_result,
        (
            ("numeric-keep", "H700001", "G700001"),
            ("numeric-drop", "H700002", "G700002"),
        ),
    )

    temporal_bundle = TemporalEventBundle(
        route=SpecialistRoute.TEMPORAL_ORDER.value,
        requested_cardinality=2,
        ordered_candidate_ids=("temporal-keep", "temporal-drop"),
        winner_candidate_id="temporal-keep",
        predecessor_candidate_id="temporal-drop",
        query_time=None,
        target_date=None,
        population_count=2,
        truncated=False,
    )
    temporal_result = _partial_result(
        TemporalInsufficiencyResult,
        routes=(SpecialistRoute.TEMPORAL_ORDER,),
        temporal_bundle=temporal_bundle,
    )
    temporal_run = _run(
        TEMPORAL_MECHANISM_ID,
        temporal_result,
        (
            ("temporal-keep", "H900001", "G900001"),
            ("temporal-drop", "H900002", "G900002"),
        ),
    )

    advisories = assay._specialist_advisories(  # noqa: SLF001
        (numeric_run, temporal_run),
        frozenset({"H700001", "H900001"}),
    )
    by_mechanism = {row["mechanism_id"]: row for row in advisories}

    # Numeric reduction is all-or-nothing: fitting removed the only witness
    # for one of the two sealed operand groups, so publishing the surviving
    # value 5 as a complete reduction would be unsound.
    assert NUMERIC_MECHANISM_ID not in by_mechanism

    temporal = by_mechanism[TEMPORAL_MECHANISM_ID]
    assert temporal["format"] == assay.SPECIALIST_ADVISORY_FORMAT
    assert temporal["handle_ids"] == ["H900001"]
    bundle = temporal["temporal_bundle"]
    assert type(bundle) is dict
    assert set(bundle.get("ordered_handle_ids", ())) <= {"H900001"}
    for key in ("winner_handle_id", "predecessor_handle_id"):
        assert bundle.get(key) in {None, "H900001"}

    rendered = json.dumps(advisories, sort_keys=True)
    assert "candidate" not in rendered
    assert "numeric-keep" not in rendered
    assert "temporal-keep" not in rendered
    assert numeric_run.local_bindings[0].candidate_id == "numeric-keep"
    assert temporal_run.local_bindings[0].candidate_id == "temporal-keep"
    for dangling in (
        "numeric-drop",
        "temporal-drop",
        "H700002",
        "H900002",
        "G700002",
        "G900002",
        "G0001",
        "G0002",
    ):
        assert dangling not in rendered


def test_numeric_advisory_omits_a_multi_group_candidate_fail_closed() -> None:
    repeated_candidate = "numeric-repeated"
    numeric_groups = tuple(
        NumericOperandGroup(
            operand_group_id=_sha(f"numeric-repeated-group-{value}"),
            operation_mode="sum",
            action_class="write",
            entity_key="pieces",
            operand_values=(value,),
            value_basis="explicit_numeric_mention",
            candidate_ids=(repeated_candidate,),
            source_group_handles=("G0001",),
        )
        for value in (7.0, 320.0, 400.0)
    )
    numeric_run = _run(
        NUMERIC_MECHANISM_ID,
        _partial_result(
            NumericOperandClosureResult,
            operand_groups=numeric_groups,
        ),
        ((repeated_candidate, "H700001", "G700001"),),
    )

    advisories = assay._specialist_advisories(  # noqa: SLF001
        (numeric_run,),
        ("H700001",),
    )

    assert advisories == []


def test_numeric_advisory_omits_an_unsupported_comparison_fail_closed() -> None:
    numeric_group = NumericOperandGroup(
        operand_group_id=_sha("numeric-comparison-group"),
        operation_mode="difference_or_compare",
        action_class="weigh",
        entity_key="change",
        operand_values=(10.0,),
        value_basis="explicit_numeric_mention",
        candidate_ids=("numeric-comparison",),
        source_group_handles=("G0001",),
    )
    numeric_run = _run(
        NUMERIC_MECHANISM_ID,
        _partial_result(
            NumericOperandClosureResult,
            operand_groups=(numeric_group,),
        ),
        (("numeric-comparison", "H700001", "G700001"),),
    )

    advisories = assay._specialist_advisories(  # noqa: SLF001
        (numeric_run,),
        ("H700001",),
    )

    assert advisories == []


def test_temporal_advisory_omits_a_handle_free_absence_claim() -> None:
    temporal_run = _run(
        TEMPORAL_MECHANISM_ID,
        _partial_result(
            TemporalInsufficiencyResult,
            routes=(SpecialistRoute.NUMERIC_SLOT_INSUFFICIENCY,),
            temporal_bundle=None,
        ),
        (),
    )

    advisories = assay._specialist_advisories(  # noqa: SLF001
        (temporal_run,),
        (),
    )

    assert advisories == []


def test_method_projection_seals_explicit_source_ids_for_posthoc_audit() -> None:
    source_id = "q36::profile-source"
    local = SimpleNamespace(
        candidate_id="profile-candidate",
        source_id=source_id,
        projection=lambda: {"source_id": source_id},
    )
    binding = SimpleNamespace(handle_id="H800001")
    contribution = SimpleNamespace(
        bindings=(binding,),
        parsed=SimpleNamespace(accepted_items=()),
        projection=lambda: {"format": "synthetic-contribution"},
    )
    run = assay._SpecialistRun(  # noqa: SLF001
        mechanism_id=PROFILE_MECHANISM_ID,
        result=object(),
        contribution=contribution,
        local_bindings=(local,),
        provider_projection={},
        local_projection={},
        specialist_receipt_sha256=_sha("profile-receipt"),
    )

    method = assay._method_projection(run)  # noqa: SLF001

    assert method["source_ids"] == [source_id]


def test_source_audit_consumes_explicit_source_ids_not_local_binding_payloads() -> None:
    method = {
        "source_ids": ["q36::selected", "other-history::other-source"],
        "local_bindings": [{"source_id": "q36::poison"}],
    }

    aliases = assay._binding_aliases(method, "q36")  # noqa: SLF001

    assert aliases == {
        "q36::selected",
        "selected",
        "other-history::other-source",
    }
    assert "poison" not in aliases


def test_terminal_receipt_chains_fitter_advisories_and_exact_final_chat() -> None:
    provider_input = {
        "dated_question": "[Question asked at 2023/05/30]\nRecommend a show.",
        "format": "synthetic-provider-input-v1",
    }
    advisories = [
        {
            "format": assay.SPECIALIST_ADVISORY_FORMAT,
            "handle_ids": ["H800001"],
            "mechanism_id": PROFILE_MECHANISM_ID,
            "purpose": "synthetic contract fixture",
        }
    ]
    fitted_receipt = _sha("fitted-prompt")
    terminal_projection = assay._terminal_projection(  # noqa: SLF001
        provider_input=provider_input,
        specialist_advisories=advisories,
        fitted_prompt_receipt_sha256=fitted_receipt,
    )

    expected_provider_input = {
        **provider_input,
        "specialist_advisories": advisories,
    }
    messages = render_final_messages(expected_provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    advisory_sha256 = identity_sha256(advisories)
    messages_sha256 = identity_sha256(list(messages))
    expected_receipt = identity_sha256(
        {
            "fitted_prompt_receipt_sha256": fitted_receipt,
            "messages_sha256": messages_sha256,
            "output_token_reserve": assay.OUTPUT_TOKEN_RESERVE,
            "prompt_token_proxy": prompt_tokens,
            "provider_input_sha256": identity_sha256(expected_provider_input),
            "specialist_advisories_sha256": advisory_sha256,
        }
    )

    assert terminal_projection["provider_input"] == expected_provider_input
    assert terminal_projection["fitted_prompt_receipt_sha256"] == fitted_receipt
    assert terminal_projection["specialist_advisories_sha256"] == advisory_sha256
    assert terminal_projection["messages_sha256"] == messages_sha256
    assert terminal_projection["prompt_token_proxy"] == prompt_tokens
    assert terminal_projection["full_chat_plus_output_tokens"] == (
        prompt_tokens + assay.OUTPUT_TOKEN_RESERVE
    )
    assert (
        terminal_projection["terminal_prompt_receipt_sha256"]
        == expected_receipt
    )
