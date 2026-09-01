from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from tools import audit_locked_semantic_global_terminal_postseal as audit
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import LocalCitationBinding
from tools.matched_eval.typed_memory_final_arm import render_final_messages
from tools.matched_eval.typed_operator_adapter import (
    ConflictPolicy,
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProviderPayloadMode,
    ProvenanceGrade,
    build_typed_evidence_packet,
    compact_typed_evidence_projection,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


TARGETS_PER_ORDINAL = {
    14: 4,
    28: 3,
    40: 3,
    49: 1,
    53: 3,
    54: 1,
    67: 3,
    69: 3,
    82: 1,
    94: 2,
    97: 2,
}
EXTRA_DIRECT_TARGETS = {(14, 0), (14, 1), (14, 2), (14, 3), (28, 0)}
LINK_ONLY_TARGETS = {(53, 2), (67, 2)}
SOURCE_DATE_UTC = "2026-08-29T12:00:00+00:00"


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _question_id(ordinal: int) -> str:
    return f"q-{ordinal}" if ordinal in TARGETS_PER_ORDINAL else f"other-{ordinal}"


def _target_plan_payload() -> dict:
    targets: list[dict] = []
    for ordinal in audit.terminal_cli.EXACT_ORDINALS:
        for index in range(TARGETS_PER_ORDINAL[ordinal]):
            source_id = f"answer_source_{ordinal}_{index}"
            body = {
                "ordinal": ordinal,
                "primary_owner": "em",
                "question_id": _question_id(ordinal),
                "target_id": source_id,
                "target_kind": "source_id",
            }
            basis = {"rule": "test-source", "source_position": index}
            targets.append(
                {
                    **body,
                    "assignment_basis": basis,
                    "assignment_basis_sha256": identity_sha256(basis),
                    "target_sha256": identity_sha256(body),
                }
            )
    body = {
        "answer_run_or_judge_inputs_loaded": False,
        "desired_target_count": len(targets),
        "desired_targets": targets,
        "format": audit.TARGET_PLAN_FORMAT,
        "gold_target_tags_posthoc_only": True,
        "ordered_question_keys": [
            {"ordinal": ordinal, "question_id": _question_id(ordinal)}
            for ordinal in range(100)
        ],
        "provider_calls": 0,
        "population_identity_sha256": _sha("test-population"),
        "question_count": 100,
        "runtime_use_forbidden": True,
    }
    return {**body, "plan_sha256": identity_sha256(body)}


def _publish_target_plan(tmp_path: Path):
    artifact, _ = publish_sealed_json(
        tmp_path / "target-plan.json", _target_plan_payload()
    )
    return audit._verified_target_plan(  # noqa: SLF001
        artifact.path,
        artifact.sha256,
        artifact.payload["plan_sha256"],
    )


def _candidate_and_binding(
    *, ordinal: int, index: int, source_id: str, quote: str
) -> tuple[dict, dict]:
    quote_sha = quote_sha256(quote)
    local = LocalCitationBinding(
        candidate_id=_sha(f"candidate:{ordinal}:{index}"),
        source_group_handle=f"G8{ordinal:02d}{index:02d}",
        namespace_id=_sha(f"namespace:{ordinal}"),
        cache_receipt_sha256=_sha(f"cache:{ordinal}"),
        source_database_sha256=_sha(f"database:{ordinal}"),
        source_store_receipt_sha256=_sha(f"store:{ordinal}"),
        source_id=source_id,
        partition_id=f"partition-{ordinal}",
        span=EvidenceSpan(
            chunk_id=f"chunk-{ordinal}-{index}",
            created_at=SOURCE_DATE_UTC,
            start_char=0,
            end_char=len(quote),
            quote_sha256=quote_sha,
            ordinal=index,
            role="user",
            source_id=source_id,
        ),
        quote_sha256=quote_sha,
    )
    candidate_body = {
        "binding_receipt_sha256": local.receipt_sha256,
        "created_at": SOURCE_DATE_UTC,
        "format": "test-terminal-candidate-v1",
        "plane": "G",
        "quote_sha256": quote_sha,
    }
    candidate = {
        **candidate_body,
        "receipt_sha256": identity_sha256(candidate_body),
    }
    return candidate, local.projection()


def _terminal_plan(
    ordinal: int,
    target_rows: list[dict],
    *,
    include_extra_raw_witnesses: bool = True,
    provider_date: str = SOURCE_DATE_UTC,
    source_question_id: str | None = None,
) -> dict:
    question_id = _question_id(ordinal)
    source_question_id = source_question_id or question_id
    dated_question = (
        f"[Question asked at 2026/08/29 12:00] What facts were recorded for {ordinal}?"
    )
    spec = compile_typed_operator_spec(dated_question)
    candidates: list[dict] = []
    local_bindings: list[dict] = []
    evidence_bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict] = []
    candidate_index = 0
    for source_index, target in enumerate(target_rows):
        witness_count = (
            2
            if include_extra_raw_witnesses
            and (ordinal, source_index) in EXTRA_DIRECT_TARGETS
            else 1
        )
        for source_turn in range(witness_count):
            quote = (
                f"Relevant fact {source_turn + 1} for source {source_index + 1} "
                "was recorded for this question."
            )
            candidate, local = _candidate_and_binding(
                ordinal=ordinal,
                index=candidate_index,
                source_id=f"{source_question_id}::{target['target_id']}",
                quote=quote,
            )
            handle = f"H4{ordinal:02d}{candidate_index:03d}"
            group = f"G8{ordinal:02d}{source_index:02d}"
            evidence_bindings.append(
                EvidenceHandleBinding(
                    handle_id=handle,
                    origin=EvidenceOrigin.MAP,
                    provenance_grade=ProvenanceGrade.EXACT_CITATION,
                    source_group_handle=group,
                    sealed_artifact_sha256=_sha("sealed-terminal-source"),
                    parent_receipt_sha256=_sha(f"parent:{ordinal}"),
                    evidence_receipt_sha256=candidate["receipt_sha256"],
                    payload_sha256=identity_sha256(candidate),
                    citation_sha256=candidate["quote_sha256"],
                    citation_char_count=len(quote),
                    local_source_locator_sha256=local["receipt_sha256"],
                )
            )
            raw_items.append(
                {
                    "date": provider_date,
                    "handle_ids": [handle],
                    "included": True,
                    "kind": "direct",
                    "status": "completed",
                    "summary": quote,
                    "value_authority": "explicit",
                }
            )
            candidates.append(candidate)
            local_bindings.append(local)
            candidate_index += 1
    bindings = tuple(evidence_bindings)
    parsed = parse_typed_items(raw_items, operator_spec=spec, bindings=bindings)
    assert not parsed.rejected_items
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("sealed-terminal-source"),),
        frontier_mode=FrontierMode.OPEN,
        conflict_policy=ConflictPolicy.QUARANTINE,
        output_token_reserve=(
            audit.PACKET_CONSTRUCTION_OUTPUT_TOKEN_RESERVE
        ),
        truncated=True,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )
    packet_projection = packet.projection()
    local_rows: list[dict] = []
    for candidate, local, binding in zip(
        candidates, local_bindings, packet.local_bindings, strict=True
    ):
        local_rows.append(
            {
                "binding": local,
                "candidate": candidate,
                "retained_after_post_selection_dedup": True,
                "selected_by_independent_plane_budget": True,
                "typed_terminal": {
                    "admitted_to_compact_packet": True,
                    "binding": local,
                    "candidate": candidate,
                    "final_handle_id": binding.handle_id,
                    "retained_in_final_prompt": True,
                },
            }
        )
    local_audit = {
        # Discovery/plane order is independent from compact consideration
        # order; the audit must prove an exact retained-population bijection.
        "local_rows": list(reversed(local_rows)),
        "packet": packet_projection,
    }
    provider_input = {
        "dated_question": dated_question,
        "typed_evidence": compact_typed_evidence_projection(packet),
    }
    messages = render_final_messages(provider_input)
    compilation = {
        "local_audit": local_audit,
        "new_provider_calls": 0,
        "packet": packet_projection,
        "retained_row_receipt_sha256s": [
            row["receipt_sha256"] for row in candidates
        ],
        "retained_transformer_token_state_bytes": 0,
    }
    return {
        "allowed_handle_ids": [row.handle_id for row in packet.handles],
        "hard_prompt_token_cap": audit.HARD_PROMPT_TOKEN_CAP,
        "ordinal": ordinal,
        "output_token_reserve": audit.OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
        "provider_input": provider_input,
        "question_id": question_id,
        "terminal_compilation": compilation,
    }


def _fixture(tmp_path: Path):
    target_plan, targets = _publish_target_plan(tmp_path)
    by_ordinal = {
        ordinal: [row for row in targets if row["ordinal"] == ordinal]
        for ordinal in audit.terminal_cli.EXACT_ORDINALS
    }
    plans = tuple(
        _terminal_plan(ordinal, by_ordinal[ordinal])
        for ordinal in audit.terminal_cli.EXACT_ORDINALS
    )
    terminal_sha = _sha("terminal-construction")
    construction = SealedArtifact(tmp_path / "construction.json", terminal_sha, {})
    replay = SealedArtifact(tmp_path / "replay.json", terminal_sha, {})
    return construction, replay, plans, target_plan, targets


def _publish_manifest(
    tmp_path: Path,
    target_plan: SealedArtifact,
    targets: tuple[dict, ...],
    plans: tuple[dict, ...],
    *,
    wrong_target_index: int | None = None,
):
    positive_rows: list[dict] = []
    for plan in plans:
        turns_by_source: dict[str, int] = {}
        for row in plan["terminal_compilation"]["local_audit"]["local_rows"]:
            terminal_source_id = row["binding"]["source_id"]
            source_index, source_id = next(
                (index, target["target_id"])
                for index, target in enumerate(
                    target
                    for target in targets
                    if target["ordinal"] == plan["ordinal"]
                )
                if (
                    f"{plan['question_id']}::{target['target_id']}"
                    == terminal_source_id
                )
            )
            turn_index = turns_by_source.get(source_id, 0)
            turns_by_source[source_id] = turn_index + 1
            link_only = (plan["ordinal"], source_index) in LINK_ONLY_TARGETS
            row_body = {
                "content_char_count": 48,
                "content_sha256": row["candidate"]["quote_sha256"],
                "format": f"{audit.WITNESS_MANIFEST_FORMAT}-witness-v1",
                "has_answer": not link_only,
                "ordinal": plan["ordinal"],
                "question_id": plan["question_id"],
                "role": "user",
                "session_turn_index": turn_index,
                "target_source_id": source_id,
                "witness_kind": "relation_link" if link_only else "answer_atom",
            }
            positive_rows.append(
                {
                    **row_body,
                    "witness_receipt_sha256": identity_sha256(row_body),
                }
            )
    if wrong_target_index is not None:
        changed = dict(positive_rows[wrong_target_index])
        changed["content_sha256"] = _sha("authenticated-but-wrong-span")
        body = {
            key: value
            for key, value in changed.items()
            if key != "witness_receipt_sha256"
        }
        positive_rows[wrong_target_index] = {
            **body,
            "witness_receipt_sha256": identity_sha256(body),
        }
    q67_target = [row for row in targets if row["ordinal"] == 67][2]
    negative_body = {
        "content_char_count": 47,
        "content_sha256": _sha("january-museum-confounder"),
        "exclusion_reason": "outside the February query window",
        "format": f"{audit.WITNESS_MANIFEST_FORMAT}-negative-witness-v1",
        "has_answer": False,
        "ordinal": 67,
        "question_id": q67_target["question_id"],
        "role": "user",
        "session_turn_index": 0,
        "target_source_id": q67_target["target_id"],
        "witness_kind": "negative_temporal_confounder",
    }
    negative_rows = [
        {
            **negative_body,
            "witness_receipt_sha256": identity_sha256(negative_body),
        }
    ]
    policy_body = {
        "format": f"{audit.WITNESS_MANIFEST_FORMAT}-policy-v1",
        "rule": "test exact positive witnesses and one excluded confounder",
    }
    witness_policy = {
        **policy_body,
        "receipt_sha256": identity_sha256(policy_body),
    }
    body = {
        "analysis_is_posthoc_only": True,
        "dataset_file_sha256": audit.LOCKED_DATASET_FILE_SHA256,
        "direct_answer_source_count": 24,
        "direct_answer_witness_count": audit.DIRECT_ANSWER_WITNESS_COUNT,
        "exact_ordinals": list(audit.terminal_cli.EXACT_ORDINALS),
        "format": audit.WITNESS_MANIFEST_FORMAT,
        "gold_loaded": True,
        "link_only_source_count": 2,
        "negative_witness_count": audit.NEGATIVE_WITNESS_COUNT,
        "negative_witnesses": negative_rows,
        "positive_witness_count": audit.POSITIVE_WITNESS_COUNT,
        "positive_witnesses": positive_rows,
        "provider_calls": 0,
        "relation_link_witness_count": audit.RELATION_LINK_WITNESS_COUNT,
        "runtime_use_forbidden": True,
        "source_target_count": audit.SOURCE_TARGET_COUNT,
        "target_plan_file_sha256": target_plan.sha256,
        "target_plan_identity_sha256": target_plan.payload["plan_sha256"],
        "target_plan_population_identity_sha256": target_plan.payload[
            "population_identity_sha256"
        ],
        "witness_policy": witness_policy,
    }
    payload = {**body, "manifest_identity_sha256": identity_sha256(body)}
    artifact, _ = publish_sealed_json(
        tmp_path / f"witness-{wrong_target_index}.json", payload
    )
    return audit._verified_witness_manifest(  # noqa: SLF001
        artifact.path,
        artifact.sha256,
        target_plan=target_plan,
        source_targets=targets,
    )


def _publish_semantic_atom_manifest(
    tmp_path: Path,
    target_plan: SealedArtifact,
    targets: tuple[dict, ...],
    plans: tuple[dict, ...],
    witness_manifest: SealedArtifact,
    *,
    locator_source_id_overrides: dict[tuple[int, str], str] | None = None,
    wrong_atom_index: int | None = None,
    wrong_source_date_index: int | None = None,
):
    locator_source_id_overrides = locator_source_id_overrides or {}
    plan_by_ordinal = {plan["ordinal"]: plan for plan in plans}
    positives = witness_manifest.payload["positive_witnesses"]
    atoms: list[dict] = []
    for atom_index, target in enumerate(targets):
        ordinal = target["ordinal"]
        question_id = target["question_id"]
        source_id = target["target_id"]
        locator_source_id = locator_source_id_overrides.get(
            (ordinal, source_id), source_id
        )
        terminal_source_id = f"{question_id}::{locator_source_id}"
        local_row = [
            row
            for row in plan_by_ordinal[ordinal]["terminal_compilation"][
                "local_audit"
            ]["local_rows"]
            if row["binding"]["source_id"] == terminal_source_id
        ][-1]
        quote_sha = (
            _sha("authenticated-but-wrong-atom-span")
            if atom_index == wrong_atom_index
            else local_row["candidate"]["quote_sha256"]
        )
        locator_body = {
            "content_char_count": 48,
            "content_sha256": quote_sha,
            "format": audit.atom_cli.LOCATOR_FORMAT,
            "has_answer": True,
            "ordinal": ordinal,
            "question_id": question_id,
            "role": "user",
            "session_turn_index": 0,
            "source_date_utc": (
                "2026-08-28T12:00:00+00:00"
                if atom_index == wrong_source_date_index
                else "2026-08-29T12:00:00+00:00"
            ),
            "source_id": locator_source_id,
        }
        locator = {
            **locator_body,
            "locator_receipt_sha256": identity_sha256(locator_body),
        }
        raw_receipts = [
            row["witness_receipt_sha256"]
            for row in positives
            if row["ordinal"] == ordinal
            and row["question_id"] == question_id
            and row["target_source_id"] == source_id
        ]
        atom_body = {
            "acceptable_evidence_locators": [locator],
            "atom_key": f"test-atom-{atom_index:02d}",
            "canonical_claim": f"Test semantic claim {atom_index + 1}.",
            "format": audit.atom_cli.ATOM_FORMAT,
            "ordinal": ordinal,
            "question_id": question_id,
            "raw_witness_receipt_sha256s": raw_receipts,
            "semantic_role": "set_member",
        }
        atoms.append(
            {
                **atom_body,
                "atom_receipt_sha256": identity_sha256(atom_body),
            }
        )
    atom_receipts = [row["atom_receipt_sha256"] for row in atoms]
    policy_body = {
        "acceptable_evidence_rule": (
            "an atom is usable only when a preregistered exact question/source/"
            "turn/content/date locator reaches a provider-usable final item"
        ),
        "builder_allowed_inputs": [
            "pinned_longmemeval_dataset",
            "pinned_target_owner_plan",
            "pinned_raw31_witness_manifest",
            "static_reviewed_declarations",
        ],
        "builder_forbidden_inputs": [
            "terminal_construction",
            "terminal_replay",
            "answer_artifact",
            "judge_artifact",
            "provider_response",
        ],
        "equivalence_rule": (
            "OR is allowed only across exact locators declared in the same atom; "
            "one locator may satisfy multiple atoms only through explicit edges"
        ),
        "format": audit.atom_cli.POLICY_FORMAT,
        "fuzzy_or_llm_equivalence_forbidden": True,
        "manifest_must_precede_next_terminal_construction": True,
        "raw_witness_association_rule": (
            "raw witnesses remain fully assigned for diagnostics, but an "
            "associated witness authorizes an atom only when it is also an "
            "acceptable exact locator"
        ),
        "runtime_routing_use_forbidden": True,
    }
    policy = {
        **policy_body,
        "receipt_sha256": identity_sha256(policy_body),
    }
    body = {
        "analysis_is_posthoc_only": True,
        "atom_count": audit.SEMANTIC_ATOM_COUNT,
        "atom_population_sha256": identity_sha256(atom_receipts),
        "atoms": atoms,
        "dataset_file_sha256": audit.LOCKED_DATASET_FILE_SHA256,
        "exact_locator_count": len(
            {
                locator["locator_receipt_sha256"]
                for atom in atoms
                for locator in atom["acceptable_evidence_locators"]
            }
        ),
        "exact_ordinals": list(audit.terminal_cli.EXACT_ORDINALS),
        "format": audit.atom_cli.FORMAT,
        "gold_loaded": True,
        "negative_witness_count": audit.NEGATIVE_WITNESS_COUNT,
        "policy": policy,
        "provider_calls": 0,
        "raw_witness_assignment_edge_count": sum(
            len(row["raw_witness_receipt_sha256s"]) for row in atoms
        ),
        "raw_witness_count": audit.POSITIVE_WITNESS_COUNT,
        "raw_witness_manifest_file_sha256": witness_manifest.sha256,
        "raw_witness_manifest_identity_sha256": witness_manifest.payload[
            "manifest_identity_sha256"
        ],
        "runtime_use_forbidden": True,
        "target_plan_file_sha256": target_plan.sha256,
        "target_plan_identity_sha256": target_plan.payload["plan_sha256"],
        "terminal_answer_judge_artifacts_loaded": False,
    }
    payload = {**body, "manifest_identity_sha256": identity_sha256(body)}
    artifact, _ = publish_sealed_json(tmp_path / "semantic-atoms.json", payload)
    verified, verified_atoms = audit._verified_semantic_atom_manifest(  # noqa: SLF001
        artifact.path,
        artifact.sha256,
        payload["manifest_identity_sha256"],
        payload["atom_population_sha256"],
        target_plan=target_plan,
        witness_manifest=witness_manifest,
    )
    return verified, verified_atoms


def test_source_only_audit_is_explicitly_semantically_indeterminate(
    tmp_path: Path,
) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=plans,
        target_plan=target_plan,
        source_targets=targets,
    )

    assert result["source_level_gate_passed"] is True
    assert result["totals"]["source_final_usable_count"] == 26
    assert result["totals"]["fact_final_usable_count"] is None
    assert result["promotion_gate_passed"] is False
    assert result["semantic_fact_coverage_status"].startswith("indeterminate_")
    assert all(row["final_provider_chains"] for row in result["target_rows"])
    assert all(
        chain["provider_visible_quote_sha256"]
        == quote_sha256(chain["provider_visible_quote"])
        for row in result["target_rows"]
        for chain in row["final_provider_chains"]
    )


def test_date_only_evidence_stays_distinct_from_exact_source_instant() -> None:
    targets = [
        row
        for row in _target_plan_payload()["desired_targets"]
        if row["ordinal"] == 14
    ]
    result = audit._audit_terminal_plan(  # noqa: SLF001
        _terminal_plan(14, targets, provider_date="2026-08-29")
    )
    chains = [
        row["final_chain"]
        for row in result["selected_rows"]
        if row["final_chain"] is not None
    ]

    assert chains
    assert all(
        chain["provider_visible_calendar_dates"] == ["2026-08-29"]
        and chain["provider_visible_utc_instants"] == []
        and chain["terminal_source_date_utc"] == SOURCE_DATE_UTC
        for chain in chains
    )


def test_offsetless_datetime_is_not_downgraded_to_calendar_date() -> None:
    with pytest.raises(
        audit.SemanticGlobalTerminalPostSealAuditError,
        match="must include a UTC offset",
    ):
        audit._evidence_temporal_identity(  # noqa: SLF001
            "2026-08-29T12:00:00",
            "test evidence date",
        )


def test_repository_witness_manifest_authenticates_31_plus_1() -> None:
    target_plan, targets = audit._verified_target_plan(  # noqa: SLF001
        audit.DEFAULT_TARGET_PLAN,
        audit.DEFAULT_TARGET_PLAN_SHA256,
        audit.DEFAULT_TARGET_PLAN_IDENTITY_SHA256,
    )
    manifest, witness_index = audit._verified_witness_manifest(  # noqa: SLF001
        audit.DEFAULT_WITNESS_MANIFEST,
        audit.DEFAULT_WITNESS_MANIFEST_SHA256,
        target_plan=target_plan,
        source_targets=targets,
    )
    positives = [
        row
        for rows in witness_index["positive_by_target"].values()
        for row in rows
    ]
    negatives = [
        row
        for rows in witness_index["negative_by_target"].values()
        for row in rows
    ]

    assert manifest.payload["manifest_identity_sha256"] == (
        audit.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
    )
    assert len(positives) == 31
    assert len(negatives) == 1
    assert {
        row["content_sha256"]
        for row in positives
        if row["witness_kind"] == "relation_link"
    } == {
        "af5b78872c00d3220eeb536df70b4f93fa2c9e5d93c784af0a817f0995000c98",
        "a720bd59171b5017431b89e00d76ad14e9424f78ba00970f1385c3a16703e0af",
    }
    assert negatives[0]["content_sha256"] == (
        "7763bc0082f3c69f650a4eb75aaf11ac13f9523633f1387f7998333d0973e066"
    )
    assert negatives[0]["content_sha256"] not in {
        row["content_sha256"] for row in positives
    }


def test_authenticated_semantic_atoms_promote_with_raw31_reported(
    tmp_path: Path,
) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    manifest, witness_index = _publish_manifest(
        tmp_path, target_plan, targets, plans
    )
    atom_manifest, semantic_atoms = _publish_semantic_atom_manifest(
        tmp_path, target_plan, targets, plans, manifest
    )
    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )

    assert result["totals"]["fact_selected_count"] == 31
    assert result["totals"]["fact_final_visible_count"] == 31
    assert result["totals"]["fact_final_usable_count"] == 31
    assert result["totals"]["raw_witness_final_usable_count"] == 31
    assert result["totals"]["semantic_atom_final_usable_count"] == 26
    assert result["witness_manifest_identity_sha256"] == manifest.payload[
        "manifest_identity_sha256"
    ]
    assert result["semantic_fact_coverage_status"] == "proven_all_26_semantic_atoms"
    assert result["promotion_gate_passed"] is True

    promotion, _ = publish_sealed_json(tmp_path / "promotion.json", result)
    verified = audit.load_verified_promotion_audit(
        promotion.path,
        promotion.sha256,
        expected_terminal_construction_sha256=construction.sha256,
        expected_terminal_replay_sha256=replay.sha256,
        expected_target_plan_sha256=target_plan.sha256,
        expected_target_plan_identity_sha256=target_plan.payload["plan_sha256"],
        expected_witness_manifest_sha256=manifest.sha256,
        expected_witness_manifest_identity_sha256=manifest.payload[
            "manifest_identity_sha256"
        ],
        expected_semantic_atom_manifest_sha256=atom_manifest.sha256,
        expected_semantic_atom_manifest_identity_sha256=atom_manifest.payload[
            "manifest_identity_sha256"
        ],
        expected_semantic_atom_population_sha256=atom_manifest.payload[
            "atom_population_sha256"
        ],
    )
    assert verified.sha256 == promotion.sha256


def test_foreign_namespace_suffix_match_cannot_pass_source_gate(
    tmp_path: Path,
) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    changed = list(plans)
    q14_targets = [row for row in targets if row["ordinal"] == 14]
    changed[0] = _terminal_plan(
        14,
        q14_targets,
        source_question_id="foreign-q-14",
    )

    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=tuple(changed),
        target_plan=target_plan,
        source_targets=targets,
    )

    assert result["totals"]["source_final_usable_count"] == 22
    assert result["source_level_gate_passed"] is False
    assert all(
        row["source_selected_after_dedup"] is False
        for row in result["target_rows"]
        if row["ordinal"] == 14
    )


def test_resealed_promotion_aggregate_forgery_fails_closed(tmp_path: Path) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    manifest, witness_index = _publish_manifest(
        tmp_path, target_plan, targets, plans
    )
    atom_manifest, semantic_atoms = _publish_semantic_atom_manifest(
        tmp_path, target_plan, targets, plans, manifest
    )
    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )
    forged = deepcopy(result)
    forged["totals"]["fact_final_usable_count"] = 30
    unsigned = dict(forged)
    unsigned.pop("audit_identity_sha256")
    forged["audit_identity_sha256"] = identity_sha256(unsigned)
    artifact, _ = publish_sealed_json(tmp_path / "forged-promotion.json", forged)

    with pytest.raises(
        audit.SemanticGlobalTerminalPostSealAuditError,
        match="aggregate closure",
    ):
        audit.load_verified_promotion_audit(
            artifact.path,
            artifact.sha256,
            expected_terminal_construction_sha256=construction.sha256,
            expected_terminal_replay_sha256=replay.sha256,
            expected_target_plan_sha256=target_plan.sha256,
            expected_target_plan_identity_sha256=target_plan.payload["plan_sha256"],
            expected_witness_manifest_sha256=manifest.sha256,
            expected_witness_manifest_identity_sha256=manifest.payload[
                "manifest_identity_sha256"
            ],
            expected_semantic_atom_manifest_sha256=atom_manifest.sha256,
            expected_semantic_atom_manifest_identity_sha256=atom_manifest.payload[
                "manifest_identity_sha256"
            ],
            expected_semantic_atom_population_sha256=atom_manifest.payload[
                "atom_population_sha256"
            ],
        )


def test_same_source_wrong_span_cannot_pass_semantic_gate(tmp_path: Path) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    manifest, witness_index = _publish_manifest(
        tmp_path,
        target_plan,
        targets,
        plans,
        wrong_target_index=0,
    )
    atom_manifest, semantic_atoms = _publish_semantic_atom_manifest(
        tmp_path,
        target_plan,
        targets,
        plans,
        manifest,
        wrong_atom_index=0,
    )
    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )

    assert result["totals"]["source_final_usable_count"] == 26
    assert result["totals"]["fact_final_usable_count"] == 30
    assert result["totals"]["semantic_atom_final_usable_count"] == 25
    assert result["promotion_gate_passed"] is False
    assert (
        result["semantic_fact_coverage_status"]
        == "failed_semantic_atom_visibility_or_usability"
    )


def test_atom26_promotes_while_raw31_remains_an_incomplete_diagnostic(
    tmp_path: Path,
) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    manifest, witness_index = _publish_manifest(
        tmp_path, target_plan, targets, plans
    )
    atom_manifest, semantic_atoms = _publish_semantic_atom_manifest(
        tmp_path, target_plan, targets, plans, manifest
    )
    by_ordinal = {
        ordinal: [row for row in targets if row["ordinal"] == ordinal]
        for ordinal in audit.terminal_cli.EXACT_ORDINALS
    }
    deduped_plans = tuple(
        _terminal_plan(
            ordinal,
            by_ordinal[ordinal],
            include_extra_raw_witnesses=False,
        )
        for ordinal in audit.terminal_cli.EXACT_ORDINALS
    )

    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=deduped_plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )

    assert result["totals"]["raw_witness_final_usable_count"] == 26
    assert result["totals"]["source_final_usable_count"] == 26
    assert result["totals"]["semantic_atom_final_usable_count"] == 26
    assert result["raw_witness_coverage_status"] == (
        "reported_incomplete_raw_witness_visibility"
    )
    assert result["promotion_gate_passed"] is True


def test_exact_atom_source_date_is_part_of_usable_equivalence(
    tmp_path: Path,
) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    manifest, witness_index = _publish_manifest(
        tmp_path, target_plan, targets, plans
    )
    atom_manifest, semantic_atoms = _publish_semantic_atom_manifest(
        tmp_path,
        target_plan,
        targets,
        plans,
        manifest,
        wrong_source_date_index=0,
    )

    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )

    assert result["totals"]["semantic_atom_final_visible_count"] == 26
    assert result["totals"]["semantic_atom_final_usable_count"] == 25
    assert result["promotion_gate_passed"] is False


def test_verified_atom26_promotion_keeps_source24_and_raw29_diagnostic(
    tmp_path: Path,
) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    manifest, witness_index = _publish_manifest(
        tmp_path, target_plan, targets, plans
    )
    q53 = [row for row in targets if row["ordinal"] == 53]
    q67 = [row for row in targets if row["ordinal"] == 67]
    omitted = {
        (53, q53[-1]["target_id"]): q53[0]["target_id"],
        (67, q67[-1]["target_id"]): q67[0]["target_id"],
    }
    atom_manifest, semantic_atoms = _publish_semantic_atom_manifest(
        tmp_path,
        target_plan,
        targets,
        plans,
        manifest,
        locator_source_id_overrides=omitted,
    )
    reduced_plans = tuple(
        _terminal_plan(
            ordinal,
            [
                row
                for row in targets
                if row["ordinal"] == ordinal
                and (ordinal, row["target_id"]) not in omitted
            ],
        )
        for ordinal in audit.terminal_cli.EXACT_ORDINALS
    )

    result = audit.build_audit(
        construction=construction,
        replay=replay,
        plans=reduced_plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )
    assert result["totals"]["source_final_usable_count"] == 24
    assert result["totals"]["raw_witness_final_usable_count"] == 29
    assert result["totals"]["semantic_atom_final_usable_count"] == 26
    assert result["source_level_gate_passed"] is False
    assert result["promotion_gate_passed"] is True

    promotion, _ = publish_sealed_json(
        tmp_path / "diagnostic-incomplete-promotion.json", result
    )
    verified = audit.load_verified_promotion_audit(
        promotion.path,
        promotion.sha256,
        expected_terminal_construction_sha256=construction.sha256,
        expected_terminal_replay_sha256=replay.sha256,
        expected_target_plan_sha256=target_plan.sha256,
        expected_target_plan_identity_sha256=target_plan.payload["plan_sha256"],
        expected_witness_manifest_sha256=manifest.sha256,
        expected_witness_manifest_identity_sha256=manifest.payload[
            "manifest_identity_sha256"
        ],
        expected_semantic_atom_manifest_sha256=atom_manifest.sha256,
        expected_semantic_atom_manifest_identity_sha256=atom_manifest.payload[
            "manifest_identity_sha256"
        ],
        expected_semantic_atom_population_sha256=atom_manifest.payload[
            "atom_population_sha256"
        ],
    )
    assert verified.sha256 == promotion.sha256


def test_provider_quote_mutation_breaks_exact_provenance_chain(tmp_path: Path) -> None:
    construction, replay, plans, target_plan, targets = _fixture(tmp_path)
    changed = list(deepcopy(plans))
    changed[0]["provider_input"]["typed_evidence"]["items"][0]["summary"] = (
        "A different span from the same source."
    )

    with pytest.raises(
        audit.SemanticGlobalTerminalPostSealAuditError,
        match="byte-identically provider-visible",
    ):
        audit.build_audit(
            construction=construction,
            replay=replay,
            plans=tuple(changed),
            target_plan=target_plan,
            source_targets=targets,
        )


def test_target_plan_is_not_opened_before_terminal_seal_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def fail_terminal(*_args, **_kwargs):
        raise audit.SemanticGlobalTerminalPostSealAuditError("terminal not sealed")

    def observe_plan(_path):
        nonlocal opened
        opened = True
        raise AssertionError("target plan must remain unopened")

    monkeypatch.setattr(audit.terminal_cli, "load_verified_terminal_assay", fail_terminal)
    monkeypatch.setattr(audit, "read_sealed_json", observe_plan)
    args = argparse.Namespace(
        expected_construction_sha256=_sha("construction"),
        expected_semantic_atom_manifest_identity_sha256=None,
        expected_semantic_atom_manifest_sha256=None,
        expected_semantic_atom_population_sha256=None,
        expected_witness_manifest_sha256=None,
        expected_replay_sha256=_sha("replay"),
        expected_target_plan_identity_sha256=_sha("plan identity"),
        expected_target_plan_sha256=_sha("plan artifact"),
        witness_manifest=None,
        output=tmp_path / "audit.json",
        promotion_gate=False,
        semantic_atom_manifest=None,
        target_plan=tmp_path / "target-plan.json",
        terminal_root=tmp_path / "terminal",
        v7_source_root=tmp_path / "v7",
    )

    with pytest.raises(
        audit.SemanticGlobalTerminalPostSealAuditError,
        match="terminal not sealed",
    ):
        audit.run_audit(args)
    assert opened is False


def test_cli_promotion_mode_fails_closed_without_fact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    construction, replay, plans, target_plan, _targets = _fixture(tmp_path)
    monkeypatch.setattr(
        audit.terminal_cli,
        "load_verified_terminal_assay",
        lambda *_args, **_kwargs: (construction, replay, plans),
    )
    output = tmp_path / "source-audit.json"
    exit_code = audit.main(
        [
            "--terminal-root",
            str(tmp_path / "terminal"),
            "--expected-construction-sha256",
            construction.sha256,
            "--expected-replay-sha256",
            replay.sha256,
            "--target-plan",
            str(target_plan.path),
            "--expected-target-plan-sha256",
            target_plan.sha256,
            "--expected-target-plan-identity-sha256",
            target_plan.payload["plan_sha256"],
            "--output",
            str(output),
            "--promotion-gate",
        ]
    )

    assert exit_code == 2
    assert output.is_file()
