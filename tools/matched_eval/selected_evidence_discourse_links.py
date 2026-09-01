"""Provider-free typed discourse links over already selected exact evidence.

This adapter is intentionally downstream of retrieval and post-selection
deduplication.  It gives the existing conservative discourse linker only the
exact source spans that survived selection, then replaces its local unit IDs
with opaque evidence handles for the provider-visible projection.  Exact
source coordinates and linker relation identities remain prompt-external.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import (
    EvidenceAtom,
    EvidenceSpan,
    identity_sha256 as discourse_identity_sha256,
    make_atom_id,
    quote_sha256,
)
from memory_condense.ingest.discourse_linker import (
    LinkerInput,
    RuleBasedDiscourseLinker,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-selected-evidence-discourse-links-v1"
INPUT_FORMAT = f"{FORMAT}-input-v1"
LINK_FORMAT = f"{FORMAT}-provider-link-v1"
BINDING_FORMAT = f"{FORMAT}-local-binding-v1"
LINKER_ID = "rule-based-discourse-linker-v1"


class SelectedEvidenceDiscourseLinkError(MatchedEvalContractError):
    """Selected evidence or its linker projection changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SelectedEvidenceDiscourseLinkError(message)


@dataclass(frozen=True, slots=True)
class SelectedEvidenceLinkInput:
    """One post-dedup opaque handle with exact authoritative provenance."""

    handle_id: str
    span: EvidenceSpan
    quote: str
    source_binding_receipt_sha256: str
    selected_evidence_receipt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.handle_id, "selected discourse-link handle")
        _require(
            self.handle_id.startswith("H") and self.handle_id[1:].isdigit(),
            "selected discourse-link handle must be opaque",
        )
        _require(
            type(self.span) is EvidenceSpan,
            "selected discourse-link span must be exact",
        )
        require_text(self.quote, "selected discourse-link quote")
        _require(
            quote_sha256(self.quote) == self.span.quote_sha256,
            "selected discourse-link quote escaped its exact span",
        )
        for value, label in (
            (self.source_binding_receipt_sha256, "selected source binding"),
            (self.selected_evidence_receipt_sha256, "selected evidence"),
        ):
            require_sha256(value, label)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "selected discourse-link input receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": INPUT_FORMAT,
            "handle_id": self.handle_id,
            "quote_sha256": quote_sha256(self.quote),
            "selected_evidence_receipt_sha256": (
                self.selected_evidence_receipt_sha256
            ),
            "source_binding_receipt_sha256": (
                self.source_binding_receipt_sha256
            ),
            "span": self.span.identity_payload(),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SelectedEvidenceDiscourseLink:
    """One provider-safe typed relation between opaque selected handles."""

    link_id: str
    relation: str
    members: tuple[Mapping[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require(
            self.link_id.startswith("D") and self.link_id[1:].isalnum(),
            "selected discourse link ID must be opaque",
        )
        require_text(self.relation, "selected discourse relation")
        _require(
            type(self.members) is tuple and len(self.members) >= 2,
            "selected discourse link requires at least two members",
        )
        handles: list[str] = []
        for member in self.members:
            _require(
                type(member) is dict
                and set(member)
                == {"evidence_role", "handle_id", "ordinal", "role"},
                "selected discourse link member schema changed",
            )
            handle = member.get("handle_id")
            _require(
                type(handle) is str
                and handle.startswith("H")
                and handle[1:].isdigit()
                and type(member.get("role")) is str
                and bool(member.get("role"))
                and member.get("evidence_role") in {"user", "assistant", "system"}
                and type(member.get("ordinal")) is int
                and member.get("ordinal") >= 0,
                "selected discourse link member changed",
            )
            handles.append(handle)
        _require(
            len(set(handles)) == len(handles),
            "selected discourse link repeats a handle",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "selected discourse link receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="selected_evidence_discourse_link")

    @property
    def handle_ids(self) -> tuple[str, ...]:
        return tuple(str(row["handle_id"]) for row in self.members)

    def projection(self, *, include_receipt: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": LINK_FORMAT,
            "link_id": self.link_id,
            "members": [dict(row) for row in self.members],
            "relation": self.relation,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SelectedEvidenceDiscourseLinks:
    """Authenticated linker output with provider and local projections."""

    input_receipt_sha256s: tuple[str, ...]
    links: tuple[SelectedEvidenceDiscourseLink, ...]
    local_bindings: tuple[Mapping[str, Any], ...]
    receipt_sha256: str = ""
    retained_transformer_token_state_bytes: int = 0

    def __post_init__(self) -> None:
        _require(
            type(self.input_receipt_sha256s) is tuple
            and len(set(self.input_receipt_sha256s))
            == len(self.input_receipt_sha256s),
            "selected discourse-link input population changed",
        )
        for value in self.input_receipt_sha256s:
            require_sha256(value, "selected discourse-link input")
        _require(
            type(self.links) is tuple
            and all(type(row) is SelectedEvidenceDiscourseLink for row in self.links)
            and len({row.link_id for row in self.links}) == len(self.links),
            "selected discourse-link population changed",
        )
        _require(
            type(self.local_bindings) is tuple
            and len(self.local_bindings) == len(self.links),
            "selected discourse-link local bindings changed",
        )
        for link, binding in zip(self.links, self.local_bindings, strict=True):
            _require(
                type(binding) is dict
                and binding.get("format") == BINDING_FORMAT
                and binding.get("link_id") == link.link_id
                and binding.get("provider_link_receipt_sha256")
                == link.receipt_sha256,
                "selected discourse link lost its local binding",
            )
            declared = require_sha256(
                binding.get("receipt_sha256"),
                "selected discourse-link local binding",
            )
            _require(
                declared
                == identity_sha256(
                    {
                        key: value
                        for key, value in binding.items()
                        if key != "receipt_sha256"
                    }
                ),
                "selected discourse-link local binding receipt changed",
            )
        _require(
            self.retained_transformer_token_state_bytes == 0,
            "selected discourse linker retained transformer token state",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "selected discourse-link compilation changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": FORMAT,
            "input_receipt_sha256s": list(self.input_receipt_sha256s),
            "link_receipt_sha256s": [row.receipt_sha256 for row in self.links],
            "linker_id": LINKER_ID,
            "local_binding_receipt_sha256s": [
                row["receipt_sha256"] for row in self.local_bindings
            ],
            "provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def link_selected_evidence(
    inputs: Sequence[SelectedEvidenceLinkInput],
) -> SelectedEvidenceDiscourseLinks:
    """Run the conservative rule linker over one exact selected population."""

    rows = tuple(inputs)
    _require(
        all(type(row) is SelectedEvidenceLinkInput for row in rows)
        and len({row.handle_id for row in rows}) == len(rows)
        and len({discourse_identity_sha256(row.span.identity_payload()) for row in rows})
        == len(rows),
        "selected discourse-link inputs must be exact and deduplicated",
    )
    if not rows:
        return SelectedEvidenceDiscourseLinks((), (), ())
    artifact_id = "selected-evidence-" + identity_sha256(
        {
            "format": FORMAT,
            "input_receipt_sha256s": [row.receipt_sha256 for row in rows],
        }
    )[:24]
    linker_inputs = tuple(
        LinkerInput(
            atom=EvidenceAtom(
                atom_id=make_atom_id(row.span),
                span=row.span,
                text=row.quote,
                label="selected_terminal_evidence",
                role=row.span.role,
                created_at=row.span.created_at,
            )
        )
        for row in rows
    )
    output = RuleBasedDiscourseLinker().link(artifact_id, linker_inputs)
    _require(
        output.retained_request_token_state_bytes == 0,
        "selected discourse linker retained request token state",
    )
    input_by_span = {
        discourse_identity_sha256(row.span.identity_payload()): row for row in rows
    }
    unit_by_id = {row.unit_id: row for row in output.units}
    links: list[SelectedEvidenceDiscourseLink] = []
    bindings: list[Mapping[str, Any]] = []
    for relation in sorted(
        output.relations,
        key=lambda row: (
            row.created_ordinal,
            row.relation_type,
            row.relation_id,
        ),
    ):
        members: list[dict[str, Any]] = []
        local_members: list[dict[str, Any]] = []
        for member in relation.members:
            unit = unit_by_id.get(member.unit_id)
            _require(
                unit is not None and len(unit.evidence) == 1,
                "selected discourse relation lost its exact unit",
            )
            assert unit is not None
            span_identity = discourse_identity_sha256(
                unit.evidence[0].identity_payload()
            )
            source = input_by_span.get(span_identity)
            _require(
                source is not None and unit.evidence[0].role is not None,
                "selected discourse relation escaped its input provenance",
            )
            assert source is not None
            members.append(
                {
                    "evidence_role": unit.evidence[0].role,
                    "handle_id": source.handle_id,
                    "ordinal": member.ordinal,
                    "role": member.role,
                }
            )
            local_members.append(
                {
                    "handle_id": source.handle_id,
                    "selected_input_receipt_sha256": source.receipt_sha256,
                    "source_binding_receipt_sha256": (
                        source.source_binding_receipt_sha256
                    ),
                    "span_identity_sha256": span_identity,
                    "unit_id": member.unit_id,
                }
            )
        provider_body = {
            "members": members,
            "relation": relation.relation_type,
        }
        link_id = "D" + identity_sha256(provider_body)[:24]
        link = SelectedEvidenceDiscourseLink(
            link_id=link_id,
            relation=relation.relation_type,
            members=tuple(members),
        )
        local_body: dict[str, Any] = {
            "format": BINDING_FORMAT,
            "link_id": link.link_id,
            "linker_id": LINKER_ID,
            "members": local_members,
            "provider_link_receipt_sha256": link.receipt_sha256,
            "relation_identity_sha256": discourse_identity_sha256(
                {
                    "artifact_id": relation.artifact_id,
                    "confidence": relation.confidence,
                    "created_ordinal": relation.created_ordinal,
                    "evidence": [
                        span.identity_payload() for span in relation.evidence
                    ],
                    "members": [
                        {
                            "ordinal": member.ordinal,
                            "role": member.role,
                            "unit_id": member.unit_id,
                            "weight": member.weight,
                        }
                        for member in relation.members
                    ],
                    "metadata": dict(relation.metadata),
                    "relation_id": relation.relation_id,
                    "relation_type": relation.relation_type,
                }
            ),
            "relation_id": relation.relation_id,
        }
        links.append(link)
        bindings.append(
            {**local_body, "receipt_sha256": identity_sha256(local_body)}
        )
    return SelectedEvidenceDiscourseLinks(
        tuple(row.receipt_sha256 for row in rows),
        tuple(links),
        tuple(bindings),
    )


__all__ = [
    "SelectedEvidenceDiscourseLink",
    "SelectedEvidenceDiscourseLinkError",
    "SelectedEvidenceDiscourseLinks",
    "SelectedEvidenceLinkInput",
    "link_selected_evidence",
]
