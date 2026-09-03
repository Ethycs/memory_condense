from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools import run_reduced_semantic_global_terminal_assay as terminal_cli
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.semantic_global_terminal_adapter import (
    FORMAT as TERMINAL_COMPILATION_FORMAT,
    PLANE_ORDER,
    ExactSpanSupportAuthority,
    ExactSpanSupportPopulationReceipt,
    PlaneBudget,
    PlaneSelectionReceipt,
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
)
from tools.matched_eval.typed_memory_final_arm import render_final_messages


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _with_receipt(body: dict, key: str) -> dict:
    return {**body, key: identity_sha256(body)}


def _frozen_episode_payload() -> dict:
    return {
        "namespace_receipts": [
            {
                "episode_artifact_binding": {
                    "resolution_mode": (
                        "authenticated_namespace_fixed_interval_auto"
                    )
                }
            }
        ]
    }


def test_terminal_defaults_to_frozen_authenticated_episode_resolution() -> None:
    args = terminal_cli.build_parser().parse_args(
        ["construct", "--ordinals", *map(str, terminal_cli.EXACT_ORDINALS)]
    )

    assert args.auto_resolve_episode_artifact is True
    assert args.episode_artifact_id is None
    assert args.terminal_compilation_mode == terminal_cli.TERMINAL_COMPILATION_MODE_V2
    assert terminal_cli.output_root_for_args(args) == terminal_cli.DEFAULT_OUTPUT_ROOT
    terminal_cli._require_frozen_episode_resolution(  # noqa: SLF001
        _frozen_episode_payload(), args
    )


@pytest.mark.parametrize(
    ("mode", "expected_format", "expected_features"),
    (
        (
            terminal_cli.TERMINAL_COMPILATION_MODE_V4,
            terminal_cli.BACKFILL_FORMAT,
            (False, True),
        ),
        (
            terminal_cli.TERMINAL_COMPILATION_MODE_V5,
            terminal_cli.LINKED_BACKFILL_FORMAT,
            (True, True),
        ),
    ),
)
def test_backfill_cli_modes_use_distinct_successor_default_roots(
    mode: str,
    expected_format: str,
    expected_features: tuple[bool, bool],
) -> None:
    args = terminal_cli.build_parser().parse_args(
        [
            "construct",
            "--ordinals",
            *map(str, terminal_cli.EXACT_ORDINALS),
            "--terminal-compilation-mode",
            mode,
        ]
    )

    assert terminal_cli.terminal_compilation_format(args) == expected_format
    assert terminal_cli.terminal_compilation_features(mode) == expected_features
    assert args.output_root == terminal_cli.DEFAULT_OUTPUT_ROOT
    assert terminal_cli.output_root_for_args(args) == (
        terminal_cli.DEFAULT_OUTPUT_ROOT_BY_MODE[mode]
    )
    assert terminal_cli.output_root_for_args(args) != terminal_cli.DEFAULT_OUTPUT_ROOT


def test_backfill_successor_default_roots_cannot_collide() -> None:
    roots = {
        terminal_cli.DEFAULT_OUTPUT_ROOT_BY_MODE[mode]
        for mode in (
            terminal_cli.TERMINAL_COMPILATION_MODE_V2,
            terminal_cli.TERMINAL_COMPILATION_MODE_V4,
            terminal_cli.TERMINAL_COMPILATION_MODE_V5,
        )
    }
    assert len(roots) == 3


def test_terminal_rejects_disabled_episode_resolution_before_rebuild() -> None:
    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="must reproduce frozen V7 authenticated auto bindings",
    ):
        terminal_cli._require_frozen_episode_resolution(  # noqa: SLF001
            _frozen_episode_payload(),
            SimpleNamespace(
                auto_resolve_episode_artifact=False,
                episode_artifact_id=None,
            ),
        )


def _payload(
    *,
    base_v7_sha256: str | None = None,
    source_binding_shifted: bool = False,
) -> dict:
    policy = SemanticGlobalTerminalPolicy().projection()
    budget_by_plane = {
        row["plane"]: row for row in policy["plane_budgets"]
    }
    plane_selections = tuple(
        PlaneSelectionReceipt(
            plane=plane,
            candidate_receipt_sha256s=(),
            consideration_policy_id=policy["plane_consideration_policy"][plane],
            consideration_candidate_receipt_sha256s=(),
            consideration_priority_vectors=(),
            upstream_attempt_receipt_sha256s=(),
            selected_candidate_receipt_sha256s=(),
            skipped_candidate_receipt_sha256s=(),
            selected_evidence_tokens=0,
            evidence_token_cap=budget_by_plane[plane]["evidence_token_cap"],
            max_items=budget_by_plane[plane]["max_items"],
            minimum_items=budget_by_plane[plane]["minimum_items"],
            upstream_budget_unpacked_selected=0,
            completed_event_lane_selected=0,
            proposed_action_lane_selected=0,
            source_group_closure_lane_selected=0,
            selected_anchor_closure_lane_selected=0,
        )
        for plane in PLANE_ORDER
    )
    exact_span_support_population = ExactSpanSupportPopulationReceipt(
        plane_candidate_receipt_sha256s=tuple(
            (plane, ()) for plane in PLANE_ORDER
        ),
        plane_selection_receipt_sha256s=tuple(
            row.receipt_sha256 for row in plane_selections
        ),
        authorities=(),
    )
    r7_construction = _sha("r7-construction")
    r7_gate = _sha("r7-gate")
    bound_construction = (
        _sha("shifted-r7-construction")
        if source_binding_shifted
        else r7_construction
    )
    bound_gate = _sha("shifted-r7-gate") if source_binding_shifted else r7_gate
    sources = TerminalSealedSources(
        protected_owner_artifact_sha256=bound_construction,
        residual_artifact_sha256=bound_construction,
        parent_artifact_sha256=bound_gate,
    ).projection()
    questions: list[dict] = []
    question_receipts: list[str] = []
    for ordinal in terminal_cli.EXACT_ORDINALS:
        dated_question = f"[Question asked at 2026/08/29 12:00] Question {ordinal}?"
        parent = f"Parent prediction {ordinal}."
        local_result_receipt = _sha(f"local:{ordinal}")
        global_result_receipt = _sha(f"global:{ordinal}")
        query_receipt = _sha(f"query:{ordinal}")
        residual_index_receipt = _sha(f"index:{ordinal}")
        provider_input = {
            "dated_question": dated_question,
            "protected_parent_fallback": {
                "label": "fallback_not_evidence",
                "prediction": parent,
                "prediction_sha256": quote_sha256(parent),
            },
            "story_coherence": {},
            "typed_evidence": [],
        }
        messages = render_final_messages(provider_input)
        terminal_prompt = {
            "allowed_handle_ids": [],
            "messages_sha256": identity_sha256(list(messages)),
            "preservation_requirements": {},
            "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
            "provider_input": provider_input,
            "story_coherence": {},
            "validation_contract": {},
        }
        local_prompt = {**terminal_prompt, "handle_group_by_id": {}}
        local_rows: list[dict] = []
        mechanism_by_handle: dict[str, str] = {}
        local_audit = {
            "exact_span_support_population": (
                exact_span_support_population.projection()
            ),
            "local_rows": local_rows,
            "mechanism_by_handle": mechanism_by_handle,
            "packet": {},
            "terminal_prompt": local_prompt,
        }
        local_audit_receipt = identity_sha256(
            {
                "format": f"{TERMINAL_COMPILATION_FORMAT}-local-audit-v1",
                "exact_span_support_population": (
                    exact_span_support_population.projection()
                ),
                "local_rows": local_rows,
                "mechanism_by_handle": mechanism_by_handle,
            }
        )
        compilation_body = {
            "exact_span_support_population_receipt_sha256": (
                exact_span_support_population.receipt_sha256
            ),
            "format": TERMINAL_COMPILATION_FORMAT,
            "global_result_receipt_sha256": global_result_receipt,
            "local_audit_receipt_sha256": local_audit_receipt,
            "local_result_receipt_sha256": local_result_receipt,
            "new_provider_calls": 0,
            "plane_selections": [row.projection() for row in plane_selections],
            "policy": policy,
            "query_receipt_sha256": query_receipt,
            "residual_index_receipt_sha256": residual_index_receipt,
            "retained_transformer_token_state_bytes": 0,
            "sealed_sources": sources,
            "terminal_prompt": terminal_prompt,
        }
        compilation = {
            **compilation_body,
            "local_audit": local_audit,
            "receipt_sha256": identity_sha256(compilation_body),
        }
        plan_body = {
            "allowed_handle_ids": [],
            "dated_question": dated_question,
            "dated_question_sha256": quote_sha256(dated_question),
            "format": terminal_cli.ANSWER_PLAN_FORMAT,
            "handle_group_by_id": {},
            "hard_prompt_token_cap": 8_000,
            "messages_sha256": identity_sha256(list(messages)),
            "ordinal": ordinal,
            "output_token_reserve": 768,
            "parent_prediction": parent,
            "parent_prediction_sha256": quote_sha256(parent),
            "preservation_requirements": {},
            "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
            "provider_input": provider_input,
            "provider_input_sha256": identity_sha256(provider_input),
            "question_id": f"q-{ordinal}",
            "question_sha256": quote_sha256(f"Question {ordinal}?"),
            "route_id": terminal_cli.ROUTE_ID,
            "source_artifact_bindings": sources,
            "story_coherence": {},
            "terminal_compilation": compilation,
            "terminal_compilation_receipt_sha256": compilation["receipt_sha256"],
            "validation_contract": {},
        }
        plan = _with_receipt(plan_body, "answer_plan_receipt_sha256")
        question_body = {
            "dated_question_sha256": plan["dated_question_sha256"],
            "global_completion": {
                "new_provider_calls": 0,
                "query_receipt_sha256": query_receipt,
                "receipt_sha256": global_result_receipt,
                "residual_index_receipt_sha256": residual_index_receipt,
                "retained_transformer_token_state_bytes": 0,
            },
            "namespace_id": "namespace-a",
            "new_provider_calls": 0,
            "ordinal": ordinal,
            "question_id": plan["question_id"],
            "question_sha256": plan["question_sha256"],
            "retained_transformer_token_state_bytes": 0,
            "terminal_answer_plan": plan,
            "v6_result_receipt_sha256": local_result_receipt,
        }
        question = _with_receipt(question_body, "question_assay_receipt_sha256")
        questions.append(question)
        question_receipts.append(question["question_assay_receipt_sha256"])
    namespace_body = {
        "namespace_id": "namespace-a",
        "question_assay_receipt_sha256s": question_receipts,
    }
    body = {
        "base_v7_assay_sha256": (
            terminal_cli.DEFAULT_V7_ASSAY_SHA256
            if base_v7_sha256 is None
            else base_v7_sha256
        ),
        "base_v7_format": terminal_cli.v7_cli.FORMAT,
        "base_v7_replay_sha256": (
            terminal_cli.DEFAULT_V7_ASSAY_SHA256
            if base_v7_sha256 is None
            else base_v7_sha256
        ),
        "exact_terminal_ordinals": list(terminal_cli.EXACT_ORDINALS),
        "format": terminal_cli.FORMAT,
        "namespace_receipts": [
            _with_receipt(namespace_body, "namespace_assay_receipt_sha256")
        ],
        "new_provider_calls": 0,
        "question_count": len(questions),
        "questions": questions,
        "r7_bindings": {
            "construction_artifact_sha256": bound_construction,
            "gate_artifact_sha256": bound_gate,
            "query_vector_artifact_sha256": _sha("query-vectors"),
            "query_vector_replay_artifact_sha256": _sha("query-vectors"),
        },
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan_count": len(questions),
        "terminal_policy": policy,
        "terminal_source_artifact_bindings": sources,
    }
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def _publish_pair(root: Path, payload: dict) -> str:
    construction, _ = publish_sealed_json(
        root / terminal_cli.CONSTRUCTION_NAME, payload
    )
    replay, _ = publish_sealed_json(root / terminal_cli.REPLAY_NAME, payload)
    assert construction.sha256 == replay.sha256
    return construction.sha256


def _bound_payload(
    root: Path,
    *,
    claimed_base_sha256: str | None = None,
    source_binding_shifted: bool = False,
) -> tuple[dict, Path]:
    base_seed = _payload(base_v7_sha256=_sha("ignored-base-seed"))
    frozen_v7_payload = terminal_cli._project_base_v7(base_seed)  # noqa: SLF001
    frozen_v7_root = root / "frozen-v7"
    frozen_construction, _ = publish_sealed_json(
        frozen_v7_root / terminal_cli.v7_cli.CONSTRUCTION_NAME,
        frozen_v7_payload,
    )
    frozen_replay, _ = publish_sealed_json(
        frozen_v7_root / terminal_cli.v7_cli.REPLAY_NAME,
        frozen_v7_payload,
    )
    assert frozen_construction.sha256 == frozen_replay.sha256
    return (
        _payload(
            base_v7_sha256=(
                frozen_construction.sha256
                if claimed_base_sha256 is None
                else claimed_base_sha256
            ),
            source_binding_shifted=source_binding_shifted,
        ),
        frozen_v7_root,
    )


def _reseal(payload: dict) -> None:
    for question in payload["questions"]:
        plan = question["terminal_answer_plan"]
        compilation = plan["terminal_compilation"]
        compilation_body = {
            key: value
            for key, value in compilation.items()
            if key not in {"local_audit", "receipt_sha256"}
        }
        compilation["receipt_sha256"] = identity_sha256(compilation_body)
        plan["terminal_compilation_receipt_sha256"] = compilation[
            "receipt_sha256"
        ]
        plan_body = {
            key: value
            for key, value in plan.items()
            if key != "answer_plan_receipt_sha256"
        }
        plan["answer_plan_receipt_sha256"] = identity_sha256(plan_body)
        question_body = {
            key: value
            for key, value in question.items()
            if key != "question_assay_receipt_sha256"
        }
        question["question_assay_receipt_sha256"] = identity_sha256(question_body)
    receipts_by_namespace: dict[str, list[str]] = {}
    for question in payload["questions"]:
        receipts_by_namespace.setdefault(question["namespace_id"], []).append(
            question["question_assay_receipt_sha256"]
        )
    for namespace in payload["namespace_receipts"]:
        namespace["question_assay_receipt_sha256s"] = receipts_by_namespace.get(
            namespace["namespace_id"], []
        )
        namespace_body = {
            key: value
            for key, value in namespace.items()
            if key != "namespace_assay_receipt_sha256"
        }
        namespace["namespace_assay_receipt_sha256"] = identity_sha256(
            namespace_body
        )
    body = {
        key: value
        for key, value in payload.items()
        if key != "construction_identity_sha256"
    }
    payload["construction_identity_sha256"] = identity_sha256(body)


def _replace_exact_span_support_population(
    question: dict,
    population: dict,
) -> None:
    compilation = question["terminal_answer_plan"]["terminal_compilation"]
    local_audit = compilation["local_audit"]
    local_audit["exact_span_support_population"] = population
    compilation["exact_span_support_population_receipt_sha256"] = population[
        "receipt_sha256"
    ]
    compilation["local_audit_receipt_sha256"] = identity_sha256(
        {
            "format": f"{TERMINAL_COMPILATION_FORMAT}-local-audit-v1",
            "exact_span_support_population": population,
            "local_rows": local_audit["local_rows"],
            "mechanism_by_handle": local_audit["mechanism_by_handle"],
        }
    )


def test_verified_reader_returns_exact_inner_answer_plan_rows(tmp_path: Path) -> None:
    payload, v7_root = _bound_payload(tmp_path)
    digest = _publish_pair(tmp_path, payload)
    construction, replay, plans = terminal_cli.load_verified_terminal_assay(
        tmp_path, digest, digest, v7_source_root=v7_root
    )

    assert construction.sha256 == replay.sha256 == digest
    assert tuple(row["ordinal"] for row in plans) == terminal_cli.EXACT_ORDINALS
    assert all(set(row) == terminal_cli.ANSWER_PLAN_KEYS for row in plans)


def test_verified_reader_rejects_local_audit_receipt_that_omits_exact_span_support(
    tmp_path: Path,
) -> None:
    source, v7_root = _bound_payload(tmp_path)
    payload = deepcopy(source)
    compilation = payload["questions"][0]["terminal_answer_plan"][
        "terminal_compilation"
    ]
    local_audit = compilation["local_audit"]
    compilation["local_audit_receipt_sha256"] = identity_sha256(
        {
            "format": f"{TERMINAL_COMPILATION_FORMAT}-local-audit-v1",
            "local_rows": local_audit["local_rows"],
            "mechanism_by_handle": local_audit["mechanism_by_handle"],
        }
    )
    _reseal(payload)
    digest = _publish_pair(tmp_path, payload)

    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="self-authentication",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_resealed_exact_span_selection_crosslink_shift(
    tmp_path: Path,
) -> None:
    source, v7_root = _bound_payload(tmp_path)
    payload = deepcopy(source)
    question = payload["questions"][0]
    population = deepcopy(
        question["terminal_answer_plan"]["terminal_compilation"]["local_audit"][
            "exact_span_support_population"
        ]
    )
    population["plane_selection_receipt_sha256s"][0] = _sha(
        "foreign-plane-selection"
    )
    population_body = {
        key: value for key, value in population.items() if key != "receipt_sha256"
    }
    population["receipt_sha256"] = identity_sha256(population_body)
    _replace_exact_span_support_population(question, population)
    _reseal(payload)
    digest = _publish_pair(tmp_path, payload)

    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="self-authentication",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_resealed_exact_span_candidate_crosslink_shift(
    tmp_path: Path,
) -> None:
    source, v7_root = _bound_payload(tmp_path)
    payload = deepcopy(source)
    question = payload["questions"][0]
    compilation = question["terminal_answer_plan"]["terminal_compilation"]
    candidate_receipt = _sha("foreign-plane-candidate")
    authority = ExactSpanSupportAuthority(
        exact_span_identity_sha256=_sha("foreign-exact-span"),
        authority_candidate_receipt_sha256s=(candidate_receipt,),
        authority_source_planes=("P",),
        supported_obligation_ids=(),
        source_group_supported_obligation_ids=(),
        source_group_supported_kinds=(),
        matched_query_actions=(),
        exact_relation_support=False,
        query_temporal_support=False,
        past_event_witness=False,
        role="user",
        priority_prefix=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    )
    population = ExactSpanSupportPopulationReceipt(
        plane_candidate_receipt_sha256s=(
            ("P", (candidate_receipt,)),
            ("R", ()),
            ("L", ()),
            ("G", ()),
        ),
        plane_selection_receipt_sha256s=tuple(
            row["receipt_sha256"] for row in compilation["plane_selections"]
        ),
        authorities=(authority,),
    ).projection()
    _replace_exact_span_support_population(question, population)
    _reseal(payload)
    digest = _publish_pair(tmp_path, payload)

    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="self-authentication",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_self_consistent_provenance_shift(
    tmp_path: Path,
) -> None:
    payload, v7_root = _bound_payload(
        tmp_path, claimed_base_sha256=_sha("wrong-v7")
    )
    digest = _publish_pair(tmp_path, payload)
    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="independently sealed V7 ancestor",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_self_consistent_source_crosslink_shift(
    tmp_path: Path,
) -> None:
    payload, v7_root = _bound_payload(tmp_path, source_binding_shifted=True)
    digest = _publish_pair(tmp_path, payload)
    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="independently sealed V7 ancestor",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_resealed_prompt_mirror_drift(tmp_path: Path) -> None:
    source, v7_root = _bound_payload(tmp_path)
    payload = deepcopy(source)
    question = payload["questions"][0]
    plan = question["terminal_answer_plan"]
    plan["terminal_compilation"]["terminal_prompt"]["messages_sha256"] = _sha(
        "wrong-inner-messages"
    )
    _reseal(payload)
    digest = _publish_pair(tmp_path, payload)

    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="self-authentication",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_coherently_resealed_nondefault_policy(
    tmp_path: Path,
) -> None:
    source, v7_root = _bound_payload(tmp_path)
    payload = deepcopy(source)
    altered = SemanticGlobalTerminalPolicy(
        plane_budgets=(
            PlaneBudget("P", 16, 1_400),
            PlaneBudget("R", 16, 1_600),
            PlaneBudget("L", 16, 1_599),
            PlaneBudget("G", 24, 2_400),
        )
    ).projection()
    payload["terminal_policy"] = altered
    for question in payload["questions"]:
        question["terminal_answer_plan"]["terminal_compilation"]["policy"] = altered
    _reseal(payload)
    digest = _publish_pair(tmp_path, payload)

    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="construction identity/population changed",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=v7_root
        )


def test_verified_reader_rejects_orphan_namespace_even_if_ancestor_matches(
    tmp_path: Path,
) -> None:
    source, _original_v7_root = _bound_payload(tmp_path)
    payload = deepcopy(source)
    payload["namespace_receipts"][0]["namespace_id"] = "orphan-namespace"
    _reseal(payload)
    orphan_v7_payload = terminal_cli._project_base_v7(payload)  # noqa: SLF001
    orphan_v7_root = tmp_path / "orphan-v7"
    frozen, _ = publish_sealed_json(
        orphan_v7_root / terminal_cli.v7_cli.CONSTRUCTION_NAME,
        orphan_v7_payload,
    )
    frozen_replay, _ = publish_sealed_json(
        orphan_v7_root / terminal_cli.v7_cli.REPLAY_NAME,
        orphan_v7_payload,
    )
    assert frozen.sha256 == frozen_replay.sha256
    payload["base_v7_assay_sha256"] = frozen.sha256
    payload["base_v7_replay_sha256"] = frozen.sha256
    _reseal(payload)
    digest = _publish_pair(tmp_path, payload)

    with pytest.raises(
        terminal_cli.ReducedSemanticGlobalTerminalAssayError,
        match="namespace set differs",
    ):
        terminal_cli.load_verified_terminal_assay(
            tmp_path, digest, digest, v7_source_root=orphan_v7_root
        )


def test_project_base_v7_removes_only_terminal_question_additions() -> None:
    base_question_body = {
        "namespace_id": "n1",
        "ordinal": 14,
        "value": "base",
    }
    base_question = _with_receipt(
        base_question_body, "question_assay_receipt_sha256"
    )
    base_namespace_body = {
        "namespace_id": "n1",
        "question_assay_receipt_sha256s": [
            base_question["question_assay_receipt_sha256"]
        ],
    }
    base_body = {
        "format": terminal_cli.v7_cli.FORMAT,
        "namespace_receipts": [
            _with_receipt(base_namespace_body, "namespace_assay_receipt_sha256")
        ],
        "questions": [base_question],
    }
    base = {**base_body, "construction_identity_sha256": identity_sha256(base_body)}

    terminal_question_body = {
        **base_question_body,
        "terminal_answer_plan": {"sealed": True},
    }
    terminal_question = _with_receipt(
        terminal_question_body, "question_assay_receipt_sha256"
    )
    terminal_namespace_body = {
        "namespace_id": "n1",
        "question_assay_receipt_sha256s": [
            terminal_question["question_assay_receipt_sha256"]
        ],
    }
    terminal_body = {
        "format": terminal_cli.v7_cli.FORMAT,
        "namespace_receipts": [
            _with_receipt(
                terminal_namespace_body, "namespace_assay_receipt_sha256"
            )
        ],
        "questions": [terminal_question],
    }
    terminalized = {
        **terminal_body,
        "construction_identity_sha256": identity_sha256(terminal_body),
    }

    assert terminal_cli._project_base_v7(terminalized) == base  # noqa: SLF001
