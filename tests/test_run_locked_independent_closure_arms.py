from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_tokens
from tools import run_locked_independent_closure_arms as runner


POPULATION_SHA = "a" * 64
QUESTION_SHA = "b" * 64
REAL_V3_ELIGIBILITY = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/independent-closure-v3/eligibility-manifest.json"
)
REAL_TARGET_OWNER_PLAN = Path(
    "docs/10 - Research Log/data/longmemeval-locked-100-target-owner-plan-v1.json"
)


def test_json_projection_handles_frozen_nested_dataclass_without_mutation() -> None:
    @dataclass(frozen=True)
    class Leaf:
        names: tuple[str, ...]

    @dataclass(frozen=True)
    class Receipt:
        values: MappingProxyType[str, object]
        leaf: Leaf

    frozen_values = MappingProxyType(
        {"reason": "budget", "nested": MappingProxyType({"count": 2})}
    )
    receipt = Receipt(values=frozen_values, leaf=Leaf(("a", "b")))

    first = runner._json_projection(receipt)
    second = runner._json_projection(receipt)

    assert first == second == {
        "values": {"reason": "budget", "nested": {"count": 2}},
        "leaf": {"names": ["a", "b"]},
    }
    assert receipt.values is frozen_values
    assert receipt.values["nested"] is frozen_values["nested"]
    with pytest.raises(TypeError):
        receipt.values["new"] = "mutation"  # type: ignore[index]


def test_json_projection_is_canonically_equivalent_to_asdict_when_supported() -> None:
    @dataclass(frozen=True)
    class Ordinary:
        names: tuple[str, ...]
        metadata: dict[str, int]

    value = Ordinary(names=("a", "b"), metadata={"count": 2})

    assert runner.canonical_json_bytes(
        runner._json_projection(value)
    ) == runner.canonical_json_bytes(asdict(value))


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


def _test_atom_identity(
    alias: str, *, identity_suffix: str = "", atom_label: str = "test-candidate"
) -> dict[str, Any]:
    source_key = f"{alias}{identity_suffix}"
    text = f"sealed source evidence for {source_key}"
    text_sha = runner.quote_sha256(text)
    span = {
        "chunk_id": f"chunk-{source_key}",
        "start_char": 0,
        "end_char": len(text),
        "quote_sha256": text_sha,
        "ordinal": 0,
        "source_id": f"source-{source_key}",
        "turn_start_char": 0,
        "turn_id": f"turn-{source_key}",
        "role": "user",
        "created_at": "2023-01-01T00:00:00Z",
    }
    return {
        "atom_id": f"atom-{runner.identity_sha256(span)[:24]}",
        "span": span,
        "text_sha256": text_sha,
        "label": atom_label,
        "role": span["role"],
        "created_at": span["created_at"],
    }


def _test_atom_id(alias: str, *, identity_suffix: str = "") -> str:
    return str(
        _test_atom_identity(alias, identity_suffix=identity_suffix)["atom_id"]
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
    atom_label: str = "test-candidate",
    selection_overflow: bool = False,
) -> dict[str, Any]:
    def identity(alias: str) -> dict[str, Any]:
        return _test_atom_identity(
            alias,
            identity_suffix=identity_suffix,
            atom_label=atom_label,
        )

    def rows(aliases: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {"atom_id": identity(item)["atom_id"], "identity": identity(item)}
            for item in aliases
        ]

    def atom_ids(aliases: tuple[str, ...]) -> list[str]:
        return [str(identity(item)["atom_id"]) for item in aliases]

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
                    "excluded_atom_ids": atom_ids(excluded),
                    "receipt_sha256": "1" * 64,
                },
                "excluded_atom_count": len(excluded),
                "excluded_atom_ids": atom_ids(excluded),
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


@pytest.mark.parametrize(
    "dated_question",
    (
        "[Question asked at 2023/04/05 (Wed) 19:25]\n"
        "What is the artist that I started to listen to last Friday?",
        "[Question asked at 2023/05/26 (Fri) 00:18]\n"
        "Which streaming service did I start using most recently?",
        "[Question asked at 2023/05/05 (Fri) 16:42]\n"
        "What gardening-related activity did I do two weeks ago?",
        "[Question asked at 2023/03/28 (Tue) 20:35]\n"
        "What was the significant buisiness milestone I mentioned four weeks ago?",
    ),
)
def test_relative_time_questions_are_closure_eligible(dated_question: str) -> None:
    route = runner.route_question(dated_question)

    assert route.modifiers.requires_temporal_metadata is True
    assert route.modifiers.requires_complete_frontier is False
    assert runner._question_only_closure_eligible(route) is True


def test_timeless_point_lookup_is_not_closure_eligible() -> None:
    route = runner.route_question(
        "[Question asked at 2023/05/30 (Tue) 23:40]\n"
        "What degree did I graduate with?"
    )

    assert route.modifiers.requires_temporal_metadata is False
    assert route.modifiers.requires_complete_frontier is False
    assert runner._question_only_closure_eligible(route) is False


@pytest.mark.skipif(
    not REAL_V3_ELIGIBILITY.is_file(), reason="sealed v3 eligibility is absent"
)
def test_v9_question_only_rule_is_a_79_question_superset() -> None:
    payload = json.loads(REAL_V3_ELIGIBILITY.read_text(encoding="utf-8"))
    eligible = tuple(
        row["ordinal"]
        for row in payload["questions"]
        if runner._question_only_closure_eligible(
            runner.route_question(row["dated_question"])
        )
    )

    assert len(eligible) == runner.EXPECTED_ELIGIBLE_COUNT == 79
    assert {6, 21, 43, 93} <= set(eligible)


@pytest.mark.skipif(
    not REAL_V3_ELIGIBILITY.is_file() or not REAL_TARGET_OWNER_PLAN.is_file(),
    reason="sealed eligibility or posthoc target-owner plan is absent",
)
def test_v9_eligibility_covers_every_bridge_and_global_owned_question() -> None:
    eligibility = json.loads(REAL_V3_ELIGIBILITY.read_text(encoding="utf-8"))
    target_plan = json.loads(REAL_TARGET_OWNER_PLAN.read_text(encoding="utf-8"))
    eligible_ordinals = {
        row["ordinal"]
        for row in eligibility["questions"]
        if runner._question_only_closure_eligible(
            runner.route_question(row["dated_question"])
        )
    }
    owned_targets = [
        row
        for row in target_plan["desired_targets"]
        if row["primary_owner"] in {"representative", "global"}
    ]

    # This is a posthoc preflight audit only. The target plan never enters the
    # runtime predicate or any retrieval/answer prompt.
    assert len(owned_targets) == 51
    assert {row["ordinal"] for row in owned_targets} <= eligible_ordinals


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
        "shared_structural_source_identity_mismatch_count": 0,
        "selected_terminal_disposition_missing_count": 0,
        "selected_discovery_credit_loss_count": 0,
    }
    shared = targets[_test_atom_id("shared")]
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
    assert shared["reachability"][0]["selection_packet_receipt_sha256"] == (
        "e" * 64
    )
    assert shared["reachability"][1]["selection_packet_receipt_sha256"] is None
    assert shared["primary_attribution_outcome"] == {
        "discovery_credit_preserved": True,
        "mechanism_admission_credit": False,
        "exact_s0_overlap_discovered": True,
        "secondary_route_only_admission": False,
    }
    assert targets[_test_atom_id("global-only")]["primary_attribution"] == (
        runner.GLOBAL_ARM
    )
    assert targets[_test_atom_id("global-only")]["admitted_after_dedup_by"] == [
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


def test_shared_atom_id_allows_only_plan_local_label_drift() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("shared",),
        selected=(),
        atom_label="representative_bridge",
    )
    global_arm = _arm(
        runner.GLOBAL_ARM,
        discovered=("shared",),
        selected=(),
        atom_label="artifact_global",
    )

    target = _attribution(representative, global_arm)["targets"][0]
    relabeled = _attribution(
        _arm(
            runner.REPRESENTATIVE_ARM,
            discovered=("shared",),
            selected=(),
            atom_label="another_representative_label",
        ),
        _arm(
            runner.GLOBAL_ARM,
            discovered=("shared",),
            selected=(),
            atom_label="another_global_label",
        ),
    )["targets"][0]

    assert target["discovering_methods"] == list(runner.ARM_LABELS)
    assert (
        target["primary_route_atom_identity"]["label"]
        == "representative_bridge"
    )
    assert "label" not in target["structural_source_identity"]
    assert (
        target["structural_source_identity_sha256"]
        == runner.identity_sha256(target["structural_source_identity"])
    )
    assert len(
        {row["atom_identity_sha256"] for row in target["reachability"]}
    ) == 2
    assert target["target_id"] == relabeled["target_id"]
    assert (
        target["primary_route_atom_identity_sha256"]
        != relabeled["primary_route_atom_identity_sha256"]
    )
    assert set(target["route_atom_identity_sha256s"]) == set(runner.ARM_LABELS)


def test_shared_atom_id_rejects_non_label_source_tampering() -> None:
    representative = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("shared",),
        selected=(),
    )
    global_arm = _arm(
        runner.GLOBAL_ARM,
        discovered=("shared",),
        selected=(),
    )
    global_identity = global_arm["candidate_pool"]["atom_identities"][0]
    global_identity["text_sha256"] = "9" * 64
    global_arm["candidate_pool"]["atom_identities_sha256"] = (
        runner.identity_sha256(global_arm["candidate_pool"]["atom_identities"])
    )

    with pytest.raises(RuntimeError, match="text digest does not bind"):
        _attribution(representative, global_arm)


@pytest.mark.parametrize("field", ("label", "role"))
def test_structural_source_identity_rejects_missing_identity_field(field: str) -> None:
    arm = _arm(runner.REPRESENTATIVE_ARM, discovered=("x",), selected=())
    del arm["candidate_pool"]["atom_identities"][0][field]
    arm["candidate_pool"]["atom_identities_sha256"] = runner.identity_sha256(
        arm["candidate_pool"]["atom_identities"]
    )

    with pytest.raises(RuntimeError, match="identity fields changed"):
        _attribution(arm, _arm(runner.GLOBAL_ARM, discovered=(), selected=()))


def test_structural_source_identity_rejects_extra_identity_field() -> None:
    arm = _arm(runner.REPRESENTATIVE_ARM, discovered=("x",), selected=())
    arm["candidate_pool"]["atom_identities"][0]["unexpected"] = True
    arm["candidate_pool"]["atom_identities_sha256"] = runner.identity_sha256(
        arm["candidate_pool"]["atom_identities"]
    )

    with pytest.raises(RuntimeError, match="identity fields changed"):
        _attribution(arm, _arm(runner.GLOBAL_ARM, discovered=(), selected=()))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("atom_id", "atom ID does not bind its exact span"),
        ("role", "atom metadata does not bind its span"),
    ),
)
def test_structural_source_identity_rejects_non_label_linkage_drift(
    mutation: str, message: str
) -> None:
    arm = _arm(runner.REPRESENTATIVE_ARM, discovered=("x",), selected=())
    identity = arm["candidate_pool"]["atom_identities"][0]
    if mutation == "atom_id":
        identity["atom_id"] = "atom-000000000000000000000000"
    else:
        identity["role"] = "assistant"
    arm["candidate_pool"]["atom_identities_sha256"] = runner.identity_sha256(
        arm["candidate_pool"]["atom_identities"]
    )

    with pytest.raises(RuntimeError, match=message):
        _attribution(arm, _arm(runner.GLOBAL_ARM, discovered=(), selected=()))


def test_route_local_full_identity_remains_byte_exact() -> None:
    arm = _arm(
        runner.REPRESENTATIVE_ARM,
        discovered=("x",),
        selected=("x",),
    )
    arm["selected_before_dedup"]["atoms"][0]["identity"]["label"] = (
        "changed-after-selection"
    )
    arm["selected_before_dedup"]["atom_identities_sha256"] = (
        runner.identity_sha256(
            [
                row["identity"]
                for row in arm["selected_before_dedup"]["atoms"]
            ]
        )
    )

    with pytest.raises(RuntimeError, match="identity changed across projections"):
        runner._arm_target_projection(arm)


def test_v9_policy_seals_label_free_union_and_exact_route_identities() -> None:
    policy = SimpleNamespace(retrieval_policy_sha256="1" * 64)
    population = SimpleNamespace(eligibility_sha256="2" * 64)
    runtime = {
        "runtime_source_binding_sha256": "3" * 64,
        "retrieval_implementation_sha256": "4" * 64,
    }

    receipt = runner._policy_receipt(
        population,
        policy,
        runtime_source_binding=runtime,
    )
    attribution = receipt["structural_candidate_attribution"]

    assert "label-free source identity" in attribution["shared_candidate_rule"]
    assert "plan-local labels may differ" in attribution["shared_candidate_rule"]
    assert attribution["route_atom_identity_rule"] == (
        "each route retains its exact full atom identity and hash"
    )
    assert (
        "shared_atom_ids_have_identical_label_free_source_identities"
        in attribution["required_invariants"]
    )
    assert receipt["policy_receipt_sha256"] == runner.identity_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "policy_receipt_sha256"
        }
    )


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
        _test_atom_id("s0"): "exact_s0_overlap_after_selection",
        _test_atom_id("projection"): "projection_drop_after_s0_dedup",
        _test_atom_id("admitted"): "admitted_after_dedup",
        _test_atom_id("repack"): "final_repack_budget_drop",
    }
    assert projection["preserved_discovery_credit_target_ids"] == [
        _test_atom_id("s0"),
        _test_atom_id("projection"),
        _test_atom_id("admitted"),
        _test_atom_id("repack"),
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

    assert projection["preserved_discovery_credit_target_ids"] == [
        _test_atom_id("overflow")
    ]
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
    changed_source = _attribution(
        _arm(
            runner.REPRESENTATIVE_ARM,
            discovered=("x",),
            selected=(),
            identity_suffix="changed-source",
        ),
        _arm(runner.GLOBAL_ARM, discovered=(), selected=()),
    )["targets"][0]["target_id"]
    assert len({first, changed_question, changed_population, changed_source}) == 4


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
class _TimedPredecessorReceipt:
    stable_field: str = "stable"
    coverage_candidate_trace_sha256: str = "3" * 64
    coverage_selector_report_sha256: str = "4" * 64
    receipt_sha256: str = "5" * 64


@dataclass
class _TimedRootStageReceipt:
    selected_evidence_ids: tuple[str, ...] = ("evidence-1",)
    method_evidence_sha256: str = "6" * 64
    receipt_sha256: str = "7" * 64


def test_stable_s0_projection_excludes_report_and_derived_receipt_fields() -> None:
    predecessor = _TimedPredecessorReceipt()
    replayed_predecessor = _TimedPredecessorReceipt(
        coverage_selector_report_sha256="8" * 64,
        receipt_sha256="9" * 64,
    )
    changed_predecessor = _TimedPredecessorReceipt(stable_field="changed")
    stage = _TimedRootStageReceipt()
    replayed_stage = _TimedRootStageReceipt(
        method_evidence_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )
    changed_stage = _TimedRootStageReceipt(selected_evidence_ids=("evidence-2",))

    assert runner._stable_predecessor_projection(predecessor) == (
        runner._stable_predecessor_projection(replayed_predecessor)
    )
    assert runner._stable_predecessor_projection(predecessor) != (
        runner._stable_predecessor_projection(changed_predecessor)
    )
    assert runner._stable_s0_stage_projection(stage) == (
        runner._stable_s0_stage_projection(replayed_stage)
    )
    assert runner._stable_s0_stage_projection(stage) != (
        runner._stable_s0_stage_projection(changed_stage)
    )


def _coverage_report(*, elapsed: float, output_candidates: int = 3) -> dict[str, Any]:
    return {
        "elapsed_s": elapsed,
        "output_candidates": output_candidates,
        "selection_status": "applied",
        "other_runtime": {"elapsed_s": 99.0},
        "score_provider_report": {
            "elapsed_s": elapsed / 2,
            "model_id": "Qwen/Qwen3-0.6B",
            "output_candidates": output_candidates,
        },
    }


def _bypassed_coverage_report(*, elapsed: float) -> dict[str, Any]:
    return {
        "elapsed_s": elapsed,
        "output_candidates": 3,
        "selection_status": "bypassed",
        "bypass_reason": "not a set query",
        "requires_completeness": False,
        "score_provider_fallback": "",
        "fallback_reason": "",
        "other_runtime": {"elapsed_s": 99.0},
        "score_provider_report": {
            "model_id": "Qwen/Qwen3-0.6B",
            "model_revision": "",
            "checkpoint_sha256": "a" * 64,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "runtime": "memory_condense.search.QwenScoreProvider",
            "retained_transformer_state_bytes": 0,
        },
    }


def test_normalized_coverage_report_removes_invoked_timing_paths() -> None:
    report = _coverage_report(elapsed=4.0)
    original = copy.deepcopy(report)

    normalized, removed = runner._coverage_report_normalization(report)

    assert report == original
    assert removed == ["elapsed_s", "score_provider_report.elapsed_s"]
    assert "elapsed_s" not in normalized
    assert "elapsed_s" not in normalized["score_provider_report"]
    assert normalized["other_runtime"]["elapsed_s"] == 99.0


def test_normalized_coverage_report_accepts_exact_scalar_bypass_identity() -> None:
    report = _bypassed_coverage_report(elapsed=4.0)
    original = copy.deepcopy(report)

    normalized, removed = runner._coverage_report_normalization(report)

    assert report == original
    assert removed == ["elapsed_s"]
    assert "elapsed_s" not in normalized
    assert normalized["score_provider_report"] == report["score_provider_report"]
    assert normalized["other_runtime"]["elapsed_s"] == 99.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.update(selection_status="applied"),
            "invoked score-provider report is missing timing telemetry",
        ),
        (
            lambda report: report.update(bypass_reason=""),
            "invoked score-provider report is missing timing telemetry",
        ),
        (
            lambda report: report.update(requires_completeness=True),
            "invoked score-provider report is missing timing telemetry",
        ),
        (
            lambda report: report.update(score_provider_fallback="partial"),
            "invoked score-provider report is missing timing telemetry",
        ),
        (
            lambda report: report.update(fallback_reason="empty candidates"),
            "invoked score-provider report is missing timing telemetry",
        ),
        (
            lambda report: report["score_provider_report"].update(extra=1),
            "identity fields changed",
        ),
        (
            lambda report: report["score_provider_report"].pop("runtime"),
            "identity fields changed",
        ),
        (
            lambda report: report["score_provider_report"].update(
                retained_transformer_state_bytes=True
            ),
            "retained transformer state",
        ),
    ],
)
def test_normalized_coverage_report_rejects_noncanonical_bypass(
    mutation: Any,
    message: str,
) -> None:
    report = _bypassed_coverage_report(elapsed=4.0)
    mutation(report)

    with pytest.raises(RuntimeError, match=message):
        runner._coverage_report_normalization(report)


def test_normalized_coverage_report_rejects_bypass_with_nested_timing() -> None:
    report = _bypassed_coverage_report(elapsed=4.0)
    report["score_provider_report"]["elapsed_s"] = 0.5

    with pytest.raises(RuntimeError, match="unexpectedly invoked"):
        runner._coverage_report_normalization(report)


def test_normalized_coverage_report_rejects_missing_or_boolean_top_timing() -> None:
    missing = _bypassed_coverage_report(elapsed=4.0)
    missing.pop("elapsed_s")
    with pytest.raises(RuntimeError, match="missing timing telemetry"):
        runner._coverage_report_normalization(missing)

    boolean = _coverage_report(elapsed=4.0)
    boolean["elapsed_s"] = True
    with pytest.raises(RuntimeError, match="missing timing telemetry"):
        runner._coverage_report_normalization(boolean)


def test_timing_only_report_drift_preserves_normalized_identity() -> None:
    first = _coverage_report(elapsed=4.0)
    second = _coverage_report(elapsed=8.0)

    assert runner.identity_sha256(first) != runner.identity_sha256(second)
    assert runner.identity_sha256(
        runner._normalized_coverage_report(first)
    ) == runner.identity_sha256(runner._normalized_coverage_report(second))


def _s0_validation_fixture() -> tuple[Any, Any, dict[str, Any]]:
    report = _coverage_report(elapsed=4.0)
    expected_predecessor = _TimedPredecessorReceipt()
    observed_predecessor = _TimedPredecessorReceipt(
        coverage_selector_report_sha256=runner.identity_sha256(report),
        receipt_sha256="8" * 64,
    )
    expected_stage = _TimedRootStageReceipt(
        method_evidence_sha256=expected_predecessor.receipt_sha256,
    )
    observed_stage = _TimedRootStageReceipt(
        method_evidence_sha256=observed_predecessor.receipt_sha256,
        receipt_sha256="9" * 64,
    )
    question = SimpleNamespace(
        predecessor_receipt=expected_predecessor,
        s0_stage_receipt=expected_stage,
        s0_messages=({"role": "user", "content": "same prompt"},),
        protected_excerpts=("same excerpt",),
    )
    result = SimpleNamespace(
        predecessor=SimpleNamespace(
            receipt=observed_predecessor,
            messages=question.s0_messages,
            excerpts=question.protected_excerpts,
        ),
        ladder=SimpleNamespace(stages=(observed_stage,)),
    )
    return question, result, report


def test_fresh_s0_attestation_binds_report_and_bilateral_stable_fields() -> None:
    question, result, report = _s0_validation_fixture()

    validation = runner._validate_exact_s0_result(
        question,
        result,
        observed_coverage_report=report,
    )

    assert validation["stable_predecessor_fields_exact"] is True
    assert validation["stable_root_stage_fields_exact"] is True
    assert validation["coverage_report_hash_exact_match"] is False
    assert validation["observed_root_method_evidence_sha256"] == (
        validation["observed_predecessor_receipt_sha256"]
    )
    assert validation["expected_stable_predecessor_projection_sha256"] == (
        validation["observed_stable_predecessor_projection_sha256"]
    )
    assert validation["fresh_report_normalization_removed_fields"] == [
        "elapsed_s",
        "score_provider_report.elapsed_s",
    ]


def test_fresh_s0_attestation_records_scalar_bypass_normalization() -> None:
    question, result, _report = _s0_validation_fixture()
    report = _bypassed_coverage_report(elapsed=4.0)
    result.predecessor.receipt = _TimedPredecessorReceipt(
        coverage_selector_report_sha256=runner.identity_sha256(report),
        receipt_sha256=result.predecessor.receipt.receipt_sha256,
    )

    validation = runner._validate_exact_s0_result(
        question,
        result,
        observed_coverage_report=report,
    )

    assert validation["fresh_report_normalization_removed_fields"] == ["elapsed_s"]
    assert validation["observed_normalized_coverage_selector_report_sha256"] == (
        runner.identity_sha256(runner._normalized_coverage_report(report))
    )


def test_fresh_s0_attestation_rejects_report_drift_and_broken_linkage() -> None:
    question, result, report = _s0_validation_fixture()
    drifted_report = _coverage_report(elapsed=4.0, output_candidates=4)
    with pytest.raises(RuntimeError, match="report payload"):
        runner._validate_exact_s0_result(
            question,
            result,
            observed_coverage_report=drifted_report,
        )

    broken_stage = _TimedRootStageReceipt(
        method_evidence_sha256="a" * 64,
        receipt_sha256=result.ladder.stages[0].receipt_sha256,
    )
    broken_result = SimpleNamespace(
        predecessor=result.predecessor,
        ladder=SimpleNamespace(stages=(broken_stage,)),
    )
    with pytest.raises(RuntimeError, match="receipt linkage"):
        runner._validate_exact_s0_result(
            question,
            broken_result,
            observed_coverage_report=report,
        )

    changed_receipt = _TimedPredecessorReceipt(
        stable_field="changed",
        coverage_selector_report_sha256=(
            result.predecessor.receipt.coverage_selector_report_sha256
        ),
        receipt_sha256=result.predecessor.receipt.receipt_sha256,
    )
    changed_result = SimpleNamespace(
        predecessor=SimpleNamespace(
            receipt=changed_receipt,
            messages=result.predecessor.messages,
            excerpts=result.predecessor.excerpts,
        ),
        ladder=result.ladder,
    )
    with pytest.raises(RuntimeError, match="stable sealed S0 receipt"):
        runner._validate_exact_s0_result(
            question,
            changed_result,
            observed_coverage_report=report,
        )


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
