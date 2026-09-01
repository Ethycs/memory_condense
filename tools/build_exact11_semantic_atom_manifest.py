#!/usr/bin/env python3
"""Seal the exact11 semantic-atom/equivalence promotion manifest.

The manifest is an evaluation-only, gold-informed declaration built solely
from the pinned LongMemEval dataset, target-owner plan, and raw 31-message
witness manifest.  It cannot read a terminal construction, answer artifact,
judge artifact, or provider response.  Retrieval and answer code may bind its
identity, but must never consume its claims or locators as runtime routing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import build_exact11_target_witness_manifest as raw_cli  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-exact11-semantic-atom-manifest-v1"
ATOM_FORMAT = f"{FORMAT}-atom-v1"
LOCATOR_FORMAT = f"{FORMAT}-exact-locator-v1"
POLICY_FORMAT = f"{FORMAT}-policy-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = raw_cli.DEFAULT_DATASET
DEFAULT_TARGET_PLAN = raw_cli.DEFAULT_TARGET_PLAN
DEFAULT_RAW_WITNESS_MANIFEST = raw_cli.DEFAULT_OUTPUT
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "docs/10 - Research Log/data/"
    "longmemeval-exact11-semantic-atom-manifest-v1.json"
)
PINNED_DATASET_SHA256 = raw_cli.PINNED_DATASET_SHA256
PINNED_TARGET_PLAN_FILE_SHA256 = raw_cli.PINNED_TARGET_PLAN_FILE_SHA256
PINNED_TARGET_PLAN_IDENTITY_SHA256 = raw_cli.PINNED_TARGET_PLAN_IDENTITY_SHA256
PINNED_RAW_WITNESS_MANIFEST_FILE_SHA256 = (
    "f6add6368971d9b0b827bc0042c5e2a2e409f26df4f2a30ef18224c34c64bd60"
)
PINNED_RAW_WITNESS_MANIFEST_IDENTITY_SHA256 = (
    "3b39b8fba2ee0bc67cb6413883973c6da3b9ee4afbe6517aed28ed0b217ee935"
)
EXACT_ORDINALS = raw_cli.EXACT_ORDINALS
EXPECTED_ATOM_COUNT = 26
EXPECTED_RAW_WITNESS_COUNT = raw_cli.EXPECTED_POSITIVE_WITNESS_COUNT
EXPECTED_NEGATIVE_WITNESS_COUNT = raw_cli.EXPECTED_NEGATIVE_WITNESS_COUNT


class Exact11SemanticAtomManifestError(MatchedEvalContractError):
    """A pinned input, exact locator, atom, or equivalence declaration changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Exact11SemanticAtomManifestError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EvidenceDeclaration:
    source_id: str
    session_turn_index: int


@dataclass(frozen=True, slots=True)
class AtomDeclaration:
    ordinal: int
    question_id: str
    atom_key: str
    canonical_claim: str
    semantic_role: str
    acceptable: tuple[EvidenceDeclaration, ...]
    raw_witnesses: tuple[EvidenceDeclaration, ...]


def _e(source_id: str, turn: int) -> EvidenceDeclaration:
    return EvidenceDeclaration(source_id=source_id, session_turn_index=turn)


# These declarations are deliberately complete and static.  They describe
# unique answer operands/anchors, not messages: repeated mentions of the same
# bike, ring, venue, or plant share one atom.  Every acceptable member remains
# an exact source-turn locator in the pinned dataset.
ATOM_DECLARATIONS: tuple[AtomDeclaration, ...] = (
    AtomDeclaration(14, "d23cf73b", "cuisine_ethio", "The user tried Ethiopian cuisine.", "set_member", (_e("answer_5a0d28f8_4", 0),), (_e("answer_5a0d28f8_4", 0),)),
    AtomDeclaration(14, "d23cf73b", "cuisine_indian", "The user learned to cook Indian cuisine.", "set_member", (_e("answer_5a0d28f8_2", 0),), (_e("answer_5a0d28f8_2", 0),)),
    AtomDeclaration(14, "d23cf73b", "cuisine_korean", "The user tried cooking Korean cuisine.", "set_member", (_e("answer_5a0d28f8_3", 0),), (_e("answer_5a0d28f8_3", 0),)),
    AtomDeclaration(14, "d23cf73b", "cuisine_vegan", "The user learned to cook vegan cuisine.", "set_member", (_e("answer_5a0d28f8_1", 0), _e("answer_5a0d28f8_1", 8)), (_e("answer_5a0d28f8_1", 0),)),
    AtomDeclaration(28, "a9f6b44c", "bike_commuter_service", "The commuter bike was planned for tire service in March.", "set_member", (_e("answer_cc021f81_2", 0),), (_e("answer_cc021f81_2", 0),)),
    AtomDeclaration(28, "a9f6b44c", "bike_road_service", "The road bike was serviced in March.", "set_member", (_e("answer_cc021f81_1", 0), _e("answer_cc021f81_1", 6), _e("answer_cc021f81_3", 0)), (_e("answer_cc021f81_1", 0), _e("answer_cc021f81_1", 6), _e("answer_cc021f81_3", 0))),
    AtomDeclaration(40, "9d25d4e0", "jewelry_earrings", "The user acquired emerald earrings.", "set_member", (_e("answer_fcff2dc4_2", 0), _e("answer_fcff2dc4_2", 8)), (_e("answer_fcff2dc4_2", 0), _e("answer_fcff2dc4_2", 8))),
    AtomDeclaration(40, "9d25d4e0", "jewelry_necklace", "The user acquired a silver necklace.", "set_member", (_e("answer_fcff2dc4_1", 0),), (_e("answer_fcff2dc4_1", 0),)),
    AtomDeclaration(40, "9d25d4e0", "jewelry_ring", "The user acquired an engagement ring.", "set_member", (_e("answer_fcff2dc4_1", 8), _e("answer_fcff2dc4_3", 0)), (_e("answer_fcff2dc4_1", 8), _e("answer_fcff2dc4_3", 0))),
    AtomDeclaration(49, "a89d7624", "denver_live_music", "The user values Denver's live-music scene and concerts.", "preference_anchor", (_e("answer_8f15ac24", 0), _e("answer_8f15ac24", 2), _e("answer_8f15ac24", 4), _e("answer_8f15ac24", 6)), (_e("answer_8f15ac24", 4), _e("answer_8f15ac24", 6))),
    AtomDeclaration(49, "a89d7624", "brandon_flowers", "The user met Brandon Flowers after The Killers' Denver concert.", "preference_anchor", (_e("answer_8f15ac24", 2), _e("answer_8f15ac24", 4)), (_e("answer_8f15ac24", 4),)),
    AtomDeclaration(53, "3a704032", "plant_peace_lily", "The user acquired a peace lily within the last month.", "set_member", (_e("answer_c2204106_2", 0), _e("answer_c2204106_2", 2)), (_e("answer_c2204106_2", 2), _e("answer_c2204106_3", 0))),
    AtomDeclaration(53, "3a704032", "plant_succulent", "The user acquired a succulent.", "set_member", (_e("answer_c2204106_2", 0), _e("answer_c2204106_2", 2)), (_e("answer_c2204106_2", 2),)),
    AtomDeclaration(53, "3a704032", "plant_snake", "The user acquired a snake plant.", "set_member", (_e("answer_c2204106_1", 4),), (_e("answer_c2204106_1", 4),)),
    AtomDeclaration(54, "gpt4_8279ba03", "appliance_smoker", "The appliance acquired ten days before the question was a smoker.", "temporal_anchor", (_e("answer_56521e66_1", 0),), (_e("answer_56521e66_1", 0),)),
    AtomDeclaration(67, "80ec1f4f", "venue_art_cube", "The user visited The Art Cube in February.", "set_member", (_e("answer_990c8992_2", 0), _e("answer_990c8992_2", 8), _e("answer_990c8992_2", 10)), (_e("answer_990c8992_2", 0), _e("answer_990c8992_3", 4))),
    AtomDeclaration(67, "80ec1f4f", "venue_natural_history", "The user visited the Natural History Museum in February.", "set_member", (_e("answer_990c8992_1", 0),), (_e("answer_990c8992_1", 0),)),
    AtomDeclaration(69, "0a995998", "clothing_blazer_pickup", "The navy blazer needed to be picked up from dry cleaning.", "action_obligation", (_e("answer_afa9873b_2", 0), _e("answer_afa9873b_2", 10)), (_e("answer_afa9873b_2", 10),)),
    AtomDeclaration(69, "0a995998", "clothing_boots_return", "The too-small Zara boots needed to be returned.", "action_obligation", (_e("answer_afa9873b_3", 6),), (_e("answer_afa9873b_3", 6),)),
    AtomDeclaration(69, "0a995998", "clothing_boots_pickup", "The replacement Zara boots needed to be picked up.", "action_obligation", (_e("answer_afa9873b_1", 4), _e("answer_afa9873b_3", 6)), (_e("answer_afa9873b_1", 4), _e("answer_afa9873b_3", 6))),
    AtomDeclaration(82, "1d4e3b97", "bike_garmin", "The user has a new Garmin bike computer and plans to track rides with it.", "causal_anchor", (_e("answer_e6b6353d", 2),), (_e("answer_e6b6353d", 2),)),
    AtomDeclaration(82, "1d4e3b97", "bike_chain_cassette", "Replacing the bike chain and cassette improved performance.", "causal_anchor", (_e("answer_e6b6353d", 4),), (_e("answer_e6b6353d", 4),)),
    AtomDeclaration(94, "9a707b81", "baking_class_date", "The baking class occurred the day before its source session.", "temporal_anchor", (_e("answer_dba89487_2", 2),), (_e("answer_dba89487_2", 2),)),
    AtomDeclaration(94, "9a707b81", "birthday_cake_date", "The friend's birthday cake was made on its source-session day.", "temporal_anchor", (_e("answer_dba89487_1", 10),), (_e("answer_dba89487_1", 10),)),
    AtomDeclaration(97, "7405e8b1", "hellofresh_discount", "The first HelloFresh order received a 40 percent discount.", "numeric_operand", (_e("answer_80323f3f_1", 0),), (_e("answer_80323f3f_1", 0),)),
    AtomDeclaration(97, "7405e8b1", "ubereats_discount", "The user received 20 percent off an UberEats order; the source does not state that it was the first order.", "numeric_operand", (_e("answer_80323f3f_2", 0),), (_e("answer_80323f3f_2", 0),)),
)


_LONGMEMEVAL_WEEKDAY_RE = re.compile(r"\([^)]*\)")


def _source_date_utc(value: str) -> str:
    cleaned = _LONGMEMEVAL_WEEKDAY_RE.sub(" ", require_text(value, "source date"))
    cleaned = " ".join(cleaned.split())
    for format_string in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, format_string).replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            continue
    raise Exact11SemanticAtomManifestError("source date format changed")


def _load_raw_witness_manifest(path: Path) -> SealedArtifact:
    artifact = read_sealed_json(path)
    payload = artifact.payload
    body = {key: value for key, value in payload.items() if key != "manifest_identity_sha256"}
    _require(
        artifact.sha256 == PINNED_RAW_WITNESS_MANIFEST_FILE_SHA256
        and payload.get("format") == raw_cli.FORMAT
        and payload.get("manifest_identity_sha256")
        == PINNED_RAW_WITNESS_MANIFEST_IDENTITY_SHA256
        and payload.get("manifest_identity_sha256") == identity_sha256(body)
        and payload.get("positive_witness_count") == EXPECTED_RAW_WITNESS_COUNT
        and payload.get("negative_witness_count") == EXPECTED_NEGATIVE_WITNESS_COUNT
        and payload.get("runtime_use_forbidden") is True,
        "raw witness manifest binding or population changed",
    )
    return artifact


def _session_projection(
    dataset_row: Mapping[str, Any], source_id: str
) -> tuple[str, list[dict[str, Any]]]:
    source_ids = _exact_list(dataset_row.get("haystack_session_ids"), "source IDs")
    sessions = _exact_list(dataset_row.get("haystack_sessions"), "sessions")
    dates = _exact_list(dataset_row.get("haystack_dates"), "source dates")
    _require(len(source_ids) == len(sessions) == len(dates), "source session population changed")
    matches = [index for index, value in enumerate(source_ids) if value == source_id]
    _require(len(matches) == 1, "declared atom source is absent or duplicated")
    index = matches[0]
    messages = [
        _exact_dict(value, "source message")
        for value in _exact_list(sessions[index], "source messages")
    ]
    return _source_date_utc(require_text(dates[index], "source date")), messages


def _locator_row(
    declaration: AtomDeclaration,
    evidence: EvidenceDeclaration,
    dataset_by_question: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = dataset_by_question.get(declaration.question_id)
    _require(row is not None, "declared atom question is absent from dataset")
    source_date, messages = _session_projection(row, evidence.source_id)
    _require(
        0 <= evidence.session_turn_index < len(messages),
        "declared atom turn escaped its source",
    )
    message = messages[evidence.session_turn_index]
    _require(
        set(message) == {"content", "has_answer", "role"}
        and message.get("role") == "user"
        and type(message.get("has_answer")) is bool,
        "declared atom evidence message schema/role changed",
    )
    content = require_text(message.get("content"), "atom evidence content")
    content_sha = quote_sha256(content)
    # The terminal provenance carries source ID and exact quote bytes, but not
    # the benchmark's session-local turn index.  Prove at declaration time
    # that those two fields identify exactly one message in this source, so an
    # audited source/content/date match cannot silently resolve to a different
    # turn with duplicate text.
    _require(
        sum(
            quote_sha256(
                require_text(
                    _exact_dict(value, "source message").get("content"),
                    "source message content",
                )
            )
            == content_sha
            for value in messages
        )
        == 1,
        "declared atom content is not unique within its source",
    )
    body = {
        "content_char_count": len(content),
        "content_sha256": content_sha,
        "format": LOCATOR_FORMAT,
        "has_answer": message.get("has_answer"),
        "ordinal": declaration.ordinal,
        "question_id": declaration.question_id,
        "role": "user",
        "session_turn_index": evidence.session_turn_index,
        "source_date_utc": source_date,
        "source_id": evidence.source_id,
    }
    return {**body, "locator_receipt_sha256": identity_sha256(body)}


def build_manifest(
    dataset_path: str | Path,
    target_plan_path: str | Path,
    raw_witness_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build the complete static atom manifest without any runtime artifact."""

    dataset = raw_cli._load_dataset(Path(dataset_path))  # noqa: SLF001
    target_plan, target_plan_file_sha256 = raw_cli._load_target_plan(  # noqa: SLF001
        Path(target_plan_path)
    )
    raw_witness = _load_raw_witness_manifest(Path(raw_witness_manifest_path))
    dataset_by_question = {
        require_text(row.get("question_id"), "dataset question ID"): row
        for row in dataset
    }
    source_targets = {
        (int(row["ordinal"]), str(row["question_id"]), str(row["target_id"]))
        for row in _exact_list(target_plan.get("desired_targets"), "target plan targets")
        if type(row) is dict
        and row.get("target_kind") == "source_id"
        and row.get("ordinal") in EXACT_ORDINALS
    }
    raw_positive = tuple(
        _exact_dict(row, "raw positive witness")
        for row in _exact_list(
            raw_witness.payload.get("positive_witnesses"), "raw positive witnesses"
        )
    )
    raw_negative = tuple(
        _exact_dict(row, "raw negative witness")
        for row in _exact_list(
            raw_witness.payload.get("negative_witnesses"), "raw negative witnesses"
        )
    )
    raw_by_locator = {
        (
            int(row["ordinal"]),
            str(row["question_id"]),
            str(row["target_source_id"]),
            int(row["session_turn_index"]),
        ): row
        for row in raw_positive
    }
    _require(
        len(ATOM_DECLARATIONS) == EXPECTED_ATOM_COUNT
        and len({(row.ordinal, row.atom_key) for row in ATOM_DECLARATIONS})
        == EXPECTED_ATOM_COUNT
        and {row.ordinal for row in ATOM_DECLARATIONS} == set(EXACT_ORDINALS),
        "static atom declaration population changed",
    )

    atoms: list[dict[str, Any]] = []
    assigned_raw_receipts: set[str] = set()
    accepted_locator_receipts: set[str] = set()
    for declaration in ATOM_DECLARATIONS:
        _require(
            declaration.semantic_role
            in {
                "action_obligation",
                "causal_anchor",
                "numeric_operand",
                "preference_anchor",
                "set_member",
                "temporal_anchor",
            }
            and bool(require_text(declaration.atom_key, "atom key"))
            and bool(require_text(declaration.canonical_claim, "canonical claim")),
            "atom semantics changed",
        )
        acceptable = tuple(
            _locator_row(declaration, evidence, dataset_by_question)
            for evidence in declaration.acceptable
        )
        _require(
            len(acceptable) == len(declaration.acceptable)
            == len({row["locator_receipt_sha256"] for row in acceptable})
            and all(
                (declaration.ordinal, declaration.question_id, row["source_id"])
                in source_targets
                for row in acceptable
            ),
            "atom acceptable locator population changed",
        )
        raw_receipts: list[str] = []
        for evidence in declaration.raw_witnesses:
            raw = raw_by_locator.get(
                (
                    declaration.ordinal,
                    declaration.question_id,
                    evidence.source_id,
                    evidence.session_turn_index,
                )
            )
            raw_locator = _locator_row(declaration, evidence, dataset_by_question)
            _require(
                raw is not None
                and raw.get("role") == raw_locator.get("role")
                and raw.get("content_sha256")
                == raw_locator.get("content_sha256"),
                "raw witness assignment is not the declared exact source turn",
            )
            receipt = require_sha256(
                raw.get("witness_receipt_sha256"), "raw witness receipt"
            )
            raw_receipts.append(receipt)
            assigned_raw_receipts.add(receipt)
        body = {
            "acceptable_evidence_locators": list(acceptable),
            "atom_key": declaration.atom_key,
            "canonical_claim": declaration.canonical_claim,
            "format": ATOM_FORMAT,
            "ordinal": declaration.ordinal,
            "question_id": declaration.question_id,
            "raw_witness_receipt_sha256s": list(dict.fromkeys(raw_receipts)),
            "semantic_role": declaration.semantic_role,
        }
        atom = {**body, "atom_receipt_sha256": identity_sha256(body)}
        atoms.append(atom)
        accepted_locator_receipts.update(
            row["locator_receipt_sha256"] for row in acceptable
        )

    raw_receipt_population = {
        require_sha256(row.get("witness_receipt_sha256"), "raw witness receipt")
        for row in raw_positive
    }
    negative_hashes = {
        require_sha256(row.get("content_sha256"), "negative witness content")
        for row in raw_negative
    }
    accepted_hashes = {
        row["content_sha256"]
        for atom in atoms
        for row in atom["acceptable_evidence_locators"]
    }
    _require(
        len(raw_receipt_population) == EXPECTED_RAW_WITNESS_COUNT
        and assigned_raw_receipts == raw_receipt_population
        and not (accepted_hashes & negative_hashes),
        "semantic atoms do not cover raw31 exactly or admit a negative witness",
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
        "format": POLICY_FORMAT,
        "fuzzy_or_llm_equivalence_forbidden": True,
        "manifest_must_precede_next_terminal_construction": True,
        "raw_witness_association_rule": (
            "raw witnesses remain fully assigned for diagnostics, but an "
            "associated witness authorizes an atom only when it is also an "
            "acceptable exact locator"
        ),
        "runtime_routing_use_forbidden": True,
    }
    policy = {**policy_body, "receipt_sha256": identity_sha256(policy_body)}
    body = {
        "analysis_is_posthoc_only": True,
        "atom_count": len(atoms),
        "atom_population_sha256": identity_sha256(atom_receipts),
        "atoms": atoms,
        "dataset_file_sha256": PINNED_DATASET_SHA256,
        "exact_locator_count": len(accepted_locator_receipts),
        "exact_ordinals": list(EXACT_ORDINALS),
        "format": FORMAT,
        "gold_loaded": True,
        "negative_witness_count": len(raw_negative),
        "policy": policy,
        "provider_calls": 0,
        "raw_witness_assignment_edge_count": sum(
            len(row["raw_witness_receipt_sha256s"]) for row in atoms
        ),
        "raw_witness_count": len(raw_receipt_population),
        "raw_witness_manifest_file_sha256": raw_witness.sha256,
        "raw_witness_manifest_identity_sha256": raw_witness.payload[
            "manifest_identity_sha256"
        ],
        "runtime_use_forbidden": True,
        "target_plan_file_sha256": target_plan_file_sha256,
        "target_plan_identity_sha256": target_plan["plan_sha256"],
        "terminal_answer_judge_artifacts_loaded": False,
    }
    return {**body, "manifest_identity_sha256": identity_sha256(body)}


def load_verified_manifest(
    path: str | Path,
    expected_file_sha256: str,
    *,
    expected_target_plan_sha256: str = PINNED_TARGET_PLAN_FILE_SHA256,
    expected_target_plan_identity_sha256: str = PINNED_TARGET_PLAN_IDENTITY_SHA256,
    expected_raw_witness_manifest_sha256: str = PINNED_RAW_WITNESS_MANIFEST_FILE_SHA256,
    expected_raw_witness_manifest_identity_sha256: str = PINNED_RAW_WITNESS_MANIFEST_IDENTITY_SHA256,
) -> SealedArtifact:
    """Authenticate manifest structure and its complete immutable bindings."""

    artifact = read_sealed_json(Path(path))
    _require(
        artifact.sha256
        == require_sha256(expected_file_sha256, "semantic atom manifest artifact"),
        "semantic atom manifest artifact binding changed",
    )
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "manifest_identity_sha256"
    }
    atoms = tuple(
        _exact_dict(row, "semantic atom")
        for row in _exact_list(payload.get("atoms"), "semantic atoms")
    )
    atom_receipts: list[str] = []
    raw_receipts: set[str] = set()
    locator_receipts: set[str] = set()
    atom_keys: set[tuple[int, str, str]] = set()
    raw_assignment_edges = 0
    for atom in atoms:
        atom_body = {
            key: value
            for key, value in atom.items()
            if key != "atom_receipt_sha256"
        }
        atom_receipt = require_sha256(
            atom.get("atom_receipt_sha256"), "semantic atom"
        )
        locators = tuple(
            _exact_dict(row, "semantic atom locator")
            for row in _exact_list(
                atom.get("acceptable_evidence_locators"), "semantic atom locators"
            )
        )
        raw_values = _exact_list(
            atom.get("raw_witness_receipt_sha256s"), "atom raw witnesses"
        )
        ordinal = atom.get("ordinal")
        question_id = require_text(
            atom.get("question_id"), "semantic atom question ID"
        )
        atom_key = require_text(atom.get("atom_key"), "semantic atom key")
        _require(
            set(atom)
            == {
                "acceptable_evidence_locators",
                "atom_key",
                "atom_receipt_sha256",
                "canonical_claim",
                "format",
                "ordinal",
                "question_id",
                "raw_witness_receipt_sha256s",
                "semantic_role",
            }
            and atom.get("format") == ATOM_FORMAT
            and atom_receipt == identity_sha256(atom_body)
            and type(ordinal) is int
            and ordinal in EXACT_ORDINALS
            and bool(question_id)
            and bool(atom_key)
            and bool(require_text(atom.get("canonical_claim"), "canonical claim"))
            and atom.get("semantic_role")
            in {
                "action_obligation",
                "causal_anchor",
                "numeric_operand",
                "preference_anchor",
                "set_member",
                "temporal_anchor",
            }
            and (ordinal, question_id, atom_key) not in atom_keys
            and len(locators) > 0
            and len(raw_values) > 0,
            "semantic atom schema, identity, role, or self-authentication changed",
        )
        atom_keys.add((ordinal, question_id, atom_key))
        atom_receipts.append(atom_receipt)
        raw_receipts.update(
            require_sha256(value, "semantic atom raw witness")
            for value in raw_values
        )
        raw_assignment_edges += len(raw_values)
        for locator in locators:
            locator_body = {
                key: value
                for key, value in locator.items()
                if key != "locator_receipt_sha256"
            }
            receipt = require_sha256(
                locator.get("locator_receipt_sha256"), "semantic atom locator"
            )
            require_sha256(
                locator.get("content_sha256"), "semantic atom locator content"
            )
            source_date = require_text(
                locator.get("source_date_utc"), "semantic atom locator source date"
            )
            try:
                parsed_source_date = datetime.fromisoformat(source_date)
            except ValueError as exc:
                raise Exact11SemanticAtomManifestError(
                    "semantic atom locator source date changed"
                ) from exc
            _require(
                set(locator)
                == {
                    "content_char_count",
                    "content_sha256",
                    "format",
                    "has_answer",
                    "locator_receipt_sha256",
                    "ordinal",
                    "question_id",
                    "role",
                    "session_turn_index",
                    "source_date_utc",
                    "source_id",
                }
                and locator.get("format") == LOCATOR_FORMAT
                and locator.get("ordinal") == ordinal
                and locator.get("question_id") == question_id
                and locator.get("role") == "user"
                and type(locator.get("has_answer")) is bool
                and type(locator.get("content_char_count")) is int
                and locator.get("content_char_count") > 0
                and type(locator.get("session_turn_index")) is int
                and locator.get("session_turn_index") >= 0
                and bool(require_text(locator.get("source_id"), "atom source ID"))
                and parsed_source_date.tzinfo is not None
                and parsed_source_date.astimezone(timezone.utc).isoformat()
                == source_date
                and receipt == identity_sha256(locator_body),
                "semantic atom locator schema or authentication changed",
            )
            locator_receipts.add(receipt)
    policy = _exact_dict(payload.get("policy"), "semantic atom policy")
    policy_body = {
        key: value for key, value in policy.items() if key != "receipt_sha256"
    }
    _require(
        set(payload)
        == {
            "analysis_is_posthoc_only",
            "atom_count",
            "atom_population_sha256",
            "atoms",
            "dataset_file_sha256",
            "exact_locator_count",
            "exact_ordinals",
            "format",
            "gold_loaded",
            "manifest_identity_sha256",
            "negative_witness_count",
            "policy",
            "provider_calls",
            "raw_witness_assignment_edge_count",
            "raw_witness_count",
            "raw_witness_manifest_file_sha256",
            "raw_witness_manifest_identity_sha256",
            "runtime_use_forbidden",
            "target_plan_file_sha256",
            "target_plan_identity_sha256",
            "terminal_answer_judge_artifacts_loaded",
        }
        and payload.get("format") == FORMAT
        and payload.get("manifest_identity_sha256") == identity_sha256(body)
        and payload.get("dataset_file_sha256") == PINNED_DATASET_SHA256
        and payload.get("target_plan_file_sha256")
        == require_sha256(expected_target_plan_sha256, "target plan artifact")
        and payload.get("target_plan_identity_sha256")
        == require_sha256(
            expected_target_plan_identity_sha256, "target plan identity"
        )
        and payload.get("raw_witness_manifest_file_sha256")
        == require_sha256(
            expected_raw_witness_manifest_sha256, "raw witness manifest artifact"
        )
        and payload.get("raw_witness_manifest_identity_sha256")
        == require_sha256(
            expected_raw_witness_manifest_identity_sha256,
            "raw witness manifest identity",
        )
        and payload.get("exact_ordinals") == list(EXACT_ORDINALS)
        and payload.get("atom_count") == len(atoms) == EXPECTED_ATOM_COUNT
        and payload.get("atom_population_sha256") == identity_sha256(atom_receipts)
        and payload.get("exact_locator_count") == len(locator_receipts)
        and payload.get("raw_witness_assignment_edge_count")
        == raw_assignment_edges
        and payload.get("raw_witness_count") == len(raw_receipts)
        == EXPECTED_RAW_WITNESS_COUNT
        and payload.get("negative_witness_count") == EXPECTED_NEGATIVE_WITNESS_COUNT
        and payload.get("provider_calls") == 0
        and payload.get("runtime_use_forbidden") is True
        and payload.get("analysis_is_posthoc_only") is True
        and payload.get("gold_loaded") is True
        and payload.get("terminal_answer_judge_artifacts_loaded") is False
        and set(policy)
        == {
            "acceptable_evidence_rule",
            "builder_allowed_inputs",
            "builder_forbidden_inputs",
            "equivalence_rule",
            "format",
            "fuzzy_or_llm_equivalence_forbidden",
            "manifest_must_precede_next_terminal_construction",
            "raw_witness_association_rule",
            "receipt_sha256",
            "runtime_routing_use_forbidden",
        }
        and policy.get("format") == POLICY_FORMAT
        and policy.get("builder_allowed_inputs")
        == [
            "pinned_longmemeval_dataset",
            "pinned_target_owner_plan",
            "pinned_raw31_witness_manifest",
            "static_reviewed_declarations",
        ]
        and policy.get("builder_forbidden_inputs")
        == [
            "terminal_construction",
            "terminal_replay",
            "answer_artifact",
            "judge_artifact",
            "provider_response",
        ]
        and policy.get("fuzzy_or_llm_equivalence_forbidden") is True
        and policy.get("manifest_must_precede_next_terminal_construction") is True
        and policy.get("runtime_routing_use_forbidden") is True
        and policy.get("receipt_sha256") == identity_sha256(policy_body),
        "semantic atom manifest population, policy, or immutable binding changed",
    )
    return artifact


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_manifest(args.dataset, args.target_plan, args.raw_witness_manifest)
    artifact, created = publish_sealed_json(args.output, payload)
    return {
        "artifact": str(artifact.path),
        "atom_count": payload["atom_count"],
        "atom_population_sha256": payload["atom_population_sha256"],
        "created": created,
        "manifest_identity_sha256": payload["manifest_identity_sha256"],
        "provider_calls": 0,
        "raw_witness_count": payload["raw_witness_count"],
        "sha256": artifact.sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument(
        "--raw-witness-manifest", type=Path, default=DEFAULT_RAW_WITNESS_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATOM_DECLARATIONS",
    "ATOM_FORMAT",
    "DEFAULT_DATASET",
    "DEFAULT_OUTPUT",
    "DEFAULT_RAW_WITNESS_MANIFEST",
    "DEFAULT_TARGET_PLAN",
    "EXPECTED_ATOM_COUNT",
    "EXPECTED_RAW_WITNESS_COUNT",
    "Exact11SemanticAtomManifestError",
    "FORMAT",
    "LOCATOR_FORMAT",
    "PINNED_RAW_WITNESS_MANIFEST_FILE_SHA256",
    "PINNED_RAW_WITNESS_MANIFEST_IDENTITY_SHA256",
    "POLICY_FORMAT",
    "build_manifest",
    "build_parser",
    "load_verified_manifest",
    "main",
    "run_build",
]
