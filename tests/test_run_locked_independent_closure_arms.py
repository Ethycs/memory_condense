from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_tokens
from tools import run_locked_independent_closure_arms as runner


POPULATION_SHA = "a" * 64
QUESTION_SHA = "b" * 64


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        retrieval=root / "retrieval.json",
        baseline_answers=root / "baseline.json",
        store_root=root / "stores",
        output_root=root / "output",
        runtime_source_root=root / "runtime-src",
        policy=root / "policy.json",
        qwen_prefix=root / "prefix",
        qwen_choice=root / "choice",
        device="cuda",
        expected_eligibility_sha256=None,
        expected_preflight_sha256=None,
        shard_offset=None,
        enable_expensive_retrieval=False,
        authorized_retrieval_questions=0,
        expected_question_count=2,
        expected_eligible_count=1,
    )


def _arm(
    label: str,
    *,
    discovered: tuple[str, ...],
    selected: tuple[str, ...],
    excluded: tuple[str, ...] = (),
    projected: tuple[str, ...] | None = None,
    admitted: tuple[str, ...] = (),
    identity_suffix: str = "",
    selection_overflow: bool = False,
) -> dict[str, Any]:
    def identity(atom_id: str) -> dict[str, str]:
        return {"atom_id": atom_id, "identity_suffix": identity_suffix}

    def rows(atom_ids: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {"atom_id": item, "identity": identity(item)} for item in atom_ids
        ]

    projected_ids = (
        tuple(item for item in selected if item not in set(excluded))
        if projected is None
        else projected
    )
    candidate_identities = [identity(item) for item in discovered]
    selected_rows = rows(selected)
    projected_identities = [identity(item) for item in projected_ids]
    admitted_rows = rows(admitted)
    if admitted:
        status, overflow_reason = "added", None
    elif selection_overflow:
        status, overflow_reason = "overflow_noop", "selected_before_dedup:test"
    elif selected and not projected_ids:
        status, overflow_reason = "no_novel_evidence", None
    elif selected:
        status, overflow_reason = "admission_budget_noop", None
    else:
        status, overflow_reason = "selection_budget_noop", None
    return {
        "arm_label": label,
        "candidate_pool": {
            "source_plan_sha256": ("c" if label == runner.REPRESENTATIVE_ARM else "d")
            * 64,
            "atom_count": len(candidate_identities),
            "atom_identities_sha256": runner.identity_sha256(
                candidate_identities
            ),
            "atom_identities": candidate_identities,
            "bundle_count": 0,
            "bundle_identities_sha256": runner.identity_sha256([]),
            "bundle_identities": [],
            "scope_witnesses_sha256": runner.identity_sha256([]),
        },
        "selected_before_dedup": (
            {
                "packet_receipt": {"receipt_sha256": "e" * 64},
                "atom_count": len(selected_rows),
                "atom_identities_sha256": runner.identity_sha256(
                    [item["identity"] for item in selected_rows]
                ),
                "atoms": selected_rows,
            }
            if selected
            else None
        ),
        "dedup": (
            None
            if selection_overflow or not selected
            else {
                "selected_plan_sha256": "f" * 64,
                "projection_receipt": {
                    "source_plan_sha256": "f" * 64,
                    "excluded_atom_ids": list(excluded),
                    "receipt_sha256": "1" * 64,
                },
                "excluded_atom_count": len(excluded),
                "excluded_atom_ids": list(excluded),
                "post_dedup_atom_count": len(projected_identities),
                "post_dedup_atom_identities_sha256": runner.identity_sha256(
                    projected_identities
                ),
                "post_dedup_atom_identities": projected_identities,
                "post_dedup_bundle_count": 0,
                "post_dedup_bundle_identities_sha256": runner.identity_sha256([]),
                "post_dedup_bundle_identities": [],
            }
        ),
        "admission": {
            "status": status,
            "overflow_reason": overflow_reason,
            "added_evidence": admitted_rows,
            "packet": (
                {
                    "packet_receipt": {"receipt_sha256": "2" * 64},
                    "atoms": admitted_rows,
                }
                if admitted
                else None
            ),
        },
    }


def _attribution(*arms: dict[str, Any]) -> dict[str, Any]:
    return runner._structural_candidate_attribution(
        population_identity_sha256=POPULATION_SHA,
        question_id="question-7",
        question_identity_sha256=QUESTION_SHA,
        arms=arms,
    )


def test_structural_attribution_is_total_and_separates_s0_coverage() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("shared", "bridge-only"),
        selected=("shared",),
        excluded=("shared",),
    )
    global_arm = _arm(
        runner.GLOBAL_ARM,
        discovered=("shared", "global-only"),
        selected=("global-only",),
        admitted=("global-only",),
    )

    manifest = _attribution(representative, global_arm)
    targets = {item["evidence_atom_id"]: item for item in manifest["targets"]}

    assert manifest["declared_structural_candidate_count"] == 3
    assert manifest["desired_target_union_completeness_claimed"] is False
    assert manifest["invariants"] == {
        "unattributed_structural_candidate_count": 0,
        "duplicate_primary_attribution_count": 0,
        "pairwise_primary_attribution_intersection_count": 0,
        "primary_attribution_union_equals_declared_structural_candidate_universe": True,
        "shared_atom_identity_mismatch_count": 0,
        "selected_terminal_disposition_missing_count": 0,
        "selected_discovery_credit_loss_count": 0,
    }
    shared = targets["shared"]
    assert shared["primary_attribution"] == runner.REPRESENTATIVE_ARM
    assert shared["discovering_methods"] == list(runner.ARM_LABELS)
    assert shared["secondary_reachability"] == [runner.GLOBAL_ARM]
    assert shared["discovery_credit_preserved_by"] == [
        runner.REPRESENTATIVE_ARM
    ]
    assert shared["admitted_after_dedup_by"] == []
    assert shared["exact_s0_overlap_discovered_by"] == [
        runner.REPRESENTATIVE_ARM
    ]
    assert shared["reachability"][0]["final_coverage_source"] == "S0_CONTROL"
    assert shared["primary_attribution_outcome"] == {
        "discovery_credit_preserved": True,
        "mechanism_admission_credit": False,
        "exact_s0_overlap_discovered": True,
        "secondary_route_only_admission": False,
    }
    assert targets["global-only"]["primary_attribution"] == runner.GLOBAL_ARM
    assert targets["global-only"]["admitted_after_dedup_by"] == [
        runner.GLOBAL_ARM
    ]
    attributed = {
        item
        for values in manifest["primary_attribution_sets"].values()
        for item in values
    }
    assert attributed == set(manifest["declared_structural_candidate_ids"])
    assert manifest["benchmark_target_tags_loaded"] is False


def test_secondary_only_admission_does_not_credit_primary_attribution() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("shared",),
        selected=(),
    )
    global_arm = _arm(
        runner.GLOBAL_ARM,
        discovered=("shared",),
        selected=("shared",),
        admitted=("shared",),
    )

    target = _attribution(representative, global_arm)["targets"][0]

    assert target["primary_attribution"] == runner.REPRESENTATIVE_ARM
    assert target["admitted_after_dedup_by"] == [runner.GLOBAL_ARM]
    assert target["primary_attribution_outcome"] == {
        "discovery_credit_preserved": False,
        "mechanism_admission_credit": False,
        "exact_s0_overlap_discovered": False,
        "secondary_route_only_admission": True,
    }


def test_shared_atom_id_requires_byte_identical_identity_payload() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("shared",),
        selected=(),
        identity_suffix="representative",
    )
    global_arm = _arm(
        runner.GLOBAL_ARM,
        discovered=("shared",),
        selected=(),
        identity_suffix="global",
    )

    with pytest.raises(RuntimeError, match="different identity payloads"):
        _attribution(representative, global_arm)


def test_selected_dispositions_are_exhaustive_and_discovery_credit_survives() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("s0", "projection", "admitted", "repack"),
        selected=("s0", "projection", "admitted", "repack"),
        excluded=("s0",),
        projected=("admitted", "repack"),
        admitted=("admitted",),
    )
    global_arm = _arm(runner.GLOBAL_ARM, discovered=(), selected=())

    projection = runner._arm_target_projection(representative)
    dispositions = {
        item["evidence_atom_id"]: item["terminal_disposition"]
        for item in projection["route_target_dispositions"]
    }

    assert dispositions == {
        "s0": "exact_s0_overlap_after_selection",
        "projection": "projection_drop_after_s0_dedup",
        "admitted": "admitted_after_dedup",
        "repack": "final_repack_budget_drop",
    }
    assert projection["preserved_discovery_credit_target_ids"] == [
        "s0",
        "projection",
        "admitted",
        "repack",
    ]
    _attribution(representative, global_arm)


def test_selection_overflow_keeps_discovery_credit_and_terminal_reason() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("overflow",),
        selected=("overflow",),
        selection_overflow=True,
    )

    projection = runner._arm_target_projection(representative)

    assert projection["preserved_discovery_credit_target_ids"] == ["overflow"]
    assert projection["route_target_dispositions"][0]["terminal_disposition"] == (
        "selection_overflow_noop"
    )
    assert projection["route_target_dispositions"][0][
        "discovery_credit_preserved"
    ] is True


def test_target_identity_binds_population_question_and_atom_payload() -> None:
    arms = (
        _arm(runner.REPRESENTATIVE_ARM, discovered=("x",), selected=()),
        _arm(runner.GLOBAL_ARM, discovered=(), selected=()),
    )
    first = _attribution(*arms)["targets"][0]["target_id"]
    changed_question = runner._structural_candidate_attribution(
        population_identity_sha256=POPULATION_SHA,
        question_id="question-8",
        question_identity_sha256="3" * 64,
        arms=arms,
    )["targets"][0]["target_id"]
    changed_population = runner._structural_candidate_attribution(
        population_identity_sha256="4" * 64,
        question_id="question-7",
        question_identity_sha256=QUESTION_SHA,
        arms=arms,
    )["targets"][0]["target_id"]
    assert len({first, changed_question, changed_population}) == 3


def test_target_projection_rejects_ids_outside_discovered_pool() -> None:
    arm = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("candidate",),
        selected=("not-a-candidate",),
    )
    with pytest.raises(RuntimeError, match="escaped its candidate universe"):
        runner._arm_target_projection(arm)


def test_aggregate_revalidates_each_question_manifest() -> None:
    arms = [
        _arm(runner.REPRESENTATIVE_ARM, discovered=("x",), selected=()),
        _arm(runner.GLOBAL_ARM, discovered=(), selected=()),
    ]
    manifest = _attribution(*arms)
    question = {
        "population_identity_sha256": POPULATION_SHA,
        "question_id": "question-7",
        "retrieval_question_part_sha256": QUESTION_SHA,
        "arms": arms,
        "structural_candidate_attribution": manifest,
    }
    aggregate = runner._aggregate_structural_candidate_attribution([question])
    assert aggregate["declared_structural_candidate_count"] == 1

    tampered = copy.deepcopy(question)
    tampered["structural_candidate_attribution"]["targets"][0][
        "secondary_reachability"
    ] = [runner.GLOBAL_ARM]
    with pytest.raises(RuntimeError, match="attribution changed"):
        runner._aggregate_structural_candidate_attribution([tampered])


def test_aggregate_rejects_declared_universe_or_route_receipt_tampering() -> None:
    arms = [
        _arm(runner.REPRESENTATIVE_ARM, discovered=("x",), selected=("x",)),
        _arm(runner.GLOBAL_ARM, discovered=(), selected=()),
    ]
    question = {
        "population_identity_sha256": POPULATION_SHA,
        "question_id": "question-7",
        "retrieval_question_part_sha256": QUESTION_SHA,
        "arms": arms,
        "structural_candidate_attribution": _attribution(*arms),
    }
    universe_tamper = copy.deepcopy(question)
    universe_tamper["structural_candidate_attribution"][
        "declared_structural_candidate_ids"
    ] = []
    with pytest.raises(RuntimeError, match="attribution changed"):
        runner._aggregate_structural_candidate_attribution([universe_tamper])

    receipt_tamper = copy.deepcopy(question)
    receipt_tamper["arms"][0]["selected_before_dedup"]["packet_receipt"][
        "receipt_sha256"
    ] = "9" * 64
    with pytest.raises(RuntimeError, match="attribution changed"):
        runner._aggregate_structural_candidate_attribution([receipt_tamper])


def test_historical_validation_precedes_any_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)

    def fail(**_kwargs: Any) -> Any:
        raise ValueError("historical validation failed")

    monkeypatch.setattr(runner, "_prepare_population", fail)
    with pytest.raises(ValueError, match="historical validation failed"):
        runner.run_preflight(args)
    assert not args.output_root.exists()


def test_runtime_binding_rejects_modified_src_and_accepts_frozen_src() -> None:
    repository = Path(runner.__file__).resolve().parents[1]
    current = repository / "src"
    frozen = repository / runner.DEFAULT_RUNTIME_SOURCE_ROOT

    with pytest.raises(ValueError, match="implementation changed"):
        runner._runtime_source_binding(
            current,
            repository=repository,
            require_imported_runtime=False,
        )

    binding = runner._runtime_source_binding(
        frozen,
        repository=repository,
        require_imported_runtime=False,
    )
    assert binding["runtime_source_root"] == (
        "eval_results/locked-campaign-a66ff05-worktree/src"
    )
    assert binding["retrieval_implementation_sha256"] == (
        runner.EXPECTED_RUNTIME_IMPLEMENTATION_SHA256
    )
    assert binding["runtime_source_binding_sha256"] == runner.identity_sha256(
        {
            key: value
            for key, value in binding.items()
            if key != "runtime_source_binding_sha256"
        }
    )


def test_preflight_publishes_only_provider_free_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    manifest = {
        "format": runner.ELIGIBILITY_FORMAT,
        "manifest_identity_sha256": "a" * 64,
        "provider_calls": 0,
        "gold_loaded": False,
    }
    population = SimpleNamespace(
        eligibility_manifest=manifest,
        eligibility_sha256=runner._digest(manifest),
    )
    preflight = {
        "format": runner.PREFLIGHT_FORMAT,
        "eligible_question_count": 1,
        "provider_calls": 0,
        "gold_loaded": False,
    }
    monkeypatch.setattr(runner, "_prepare_population", lambda **_kwargs: population)
    monkeypatch.setattr(runner, "_preflight", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(
        runner,
        "retrieve_recall_guarded_cumulative_packet",
        lambda *_args, **_kwargs: pytest.fail("preflight invoked retrieval"),
    )

    artifact, manifest_sha, preflight_sha = runner.run_preflight(args)

    assert artifact == preflight
    assert manifest_sha == runner._digest(manifest)
    assert preflight_sha == runner._digest(preflight)
    assert not (args.output_root / "questions").exists()
    assert not (args.output_root / "shards").exists()


def test_retrieve_requires_exact_missing_question_authorization_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.shard_offset = 0
    question = SimpleNamespace(shard_offset=0, ordinal=3)
    population = SimpleNamespace(eligible=(question,))
    store_report = {"shard_offset": 0}
    preflight = {
        "stores": [store_report],
        "policy": {"policy_receipt_sha256": "p" * 64},
        "runtime_source_binding": {
            "runtime_source_binding_sha256": "a" * 64,
            "retrieval_implementation_sha256": "b" * 64,
        },
        "source_surface_sha256s": {},
    }
    monkeypatch.setattr(
        runner,
        "_load_campaign",
        lambda _args: (population, preflight, "f" * 64, object()),
    )
    monkeypatch.setattr(runner, "_validate_store", lambda *_a, **_k: store_report)
    monkeypatch.setattr(
        runner,
        "_open_store",
        lambda *_a, **_k: pytest.fail("authorization guard opened a store"),
    )

    with pytest.raises(ValueError, match="enable-expensive-retrieval"):
        runner.run_retrieve(args)
    args.enable_expensive_retrieval = True
    with pytest.raises(ValueError, match="must exactly equal"):
        runner.run_retrieve(args)
    assert not args.output_root.exists()


@dataclass
class _Receipt:
    receipt_sha256: str = "3" * 64


@dataclass
class _ProjectionReceipt:
    source_plan_sha256: str = "b" * 64
    excluded_atom_ids: tuple[str, ...] = ()
    receipt_sha256: str = "d" * 64


class _Identity:
    def __init__(self, identity: str) -> None:
        self.atom_id = identity
        self.bundle_id = identity
        self.text = "new evidence"
        self.span = SimpleNamespace(
            chunk_id=f"chunk-{identity}", source_id=f"source-{identity}"
        )

    def identity_payload(self) -> dict[str, str]:
        return {"atom_id": self.atom_id}


def test_each_arm_selects_before_s0_dedup_and_reuses_the_same_isolated_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = _Identity("atom-1")
    bundle = _Identity("bundle-1")
    raw_plan = SimpleNamespace(
        plan_sha256="a" * 64,
        atoms=(atom,),
        bundles=(bundle,),
        visited_episode_ids=(),
        visited_unit_ids=(),
        visited_relation_ids=(),
        scope_witnesses=(),
        complete_claimed=False,
        stopping_reason="budget_impossible",
    )
    selected_plan = SimpleNamespace(plan_sha256="b" * 64)
    projected_plan = SimpleNamespace(atoms=(atom,), bundles=(bundle,))
    packet = SimpleNamespace(
        atoms=(atom,), bundles=(bundle,), context="new evidence", receipt=_Receipt()
    )
    question = SimpleNamespace(
        dated_question="[Question asked at now]\nWhat changed?",
        protected_context="[1] exact S0",
        protected_excerpts=(object(),),
        s0_messages=(
            {"role": "system", "content": runner.QA_SYSTEM_PROMPT},
            {"role": "user", "content": "sealed S0"},
        ),
    )
    calls: list[tuple[object, int]] = []

    def pack(_condenser: Any, plan: object, **kwargs: Any) -> Any:
        calls.append((plan, kwargs["max_context_tokens"]))
        return packet

    def project(plan: object, protected: object, admitted: object) -> Any:
        assert plan is selected_plan
        assert protected is question.protected_excerpts
        assert admitted == ()
        return SimpleNamespace(
            plan=projected_plan,
            receipt=_ProjectionReceipt(),
        )

    monkeypatch.setattr(runner, "_pack_additions", pack)
    monkeypatch.setattr(runner, "_selected_plan", lambda *_args: selected_plan)
    monkeypatch.setattr(runner, "_novel_closure_projection", project)

    arm = runner._build_arm(
        label=runner.REPRESENTATIVE_ARM,
        plan=raw_plan,
        question=question,
        condenser=object(),
    )

    cap = min(
        runner.MAX_CONTEXT_TOKENS,
        count_tokens(question.protected_context) + runner.ADDITION_TOKEN_CAP,
    )
    assert calls == [(raw_plan, cap), (projected_plan, cap)]
    assert arm["selected_before_dedup"]["atoms"][0]["atom_id"] == "atom-1"
    assert arm["admission"]["addition_token_cap"] == 2_048
    assert arm["admission"]["status"] == "added"
    projection = runner._arm_target_projection(arm)
    assert projection["admitted_target_ids_after_dedup"] == ["atom-1"]
