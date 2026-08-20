"""Atomic hard-budget packing for episodic discourse closure plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from memory_condense.domain._discourse_identity import exact_int
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosureReceipt,
    EvidenceAtom,
    EvidenceBundle,
    EvidencePacket,
    EvidenceSpan,
    QueryProgram,
    evidence_span_sort_key,
    identity_sha256,
)


_HEADER = "## Source-grounded evidence\n"


@dataclass(frozen=True, slots=True)
class _BeamState:
    bundle_ids: tuple[str, ...]
    atom_ids: frozenset[str]
    obligation_ids: frozenset[str]
    required_weight: float
    desired_weight: float
    utility: float
    direct_raw_count: int
    token_count: int
    prompt_token_count: int | None


@dataclass(frozen=True, slots=True)
class EvidencePromptBudget:
    base_messages: tuple[tuple[str, str], ...]
    evidence_message_role: str
    evidence_prefix: str
    evidence_suffix: str
    max_prompt_tokens: int
    output_token_reserve: int

    def __post_init__(self) -> None:
        if type(self.base_messages) is not tuple or any(
            type(message) is not tuple
            or len(message) != 2
            or any(type(value) is not str for value in message)
            for message in self.base_messages
        ):
            raise TypeError(
                "base_messages must be an exact tuple of (role, content) strings"
            )
        if any(
            not role or role != role.strip()
            for role, _content in self.base_messages
        ):
            raise ValueError("base message roles must be non-empty and unpadded")
        for name in (
            "evidence_message_role",
            "evidence_prefix",
            "evidence_suffix",
        ):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact string")
        if (
            not self.evidence_message_role
            or self.evidence_message_role != self.evidence_message_role.strip()
        ):
            raise ValueError(
                "evidence_message_role must be non-empty and unpadded"
            )
        for name in ("max_prompt_tokens", "output_token_reserve"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            object.__setattr__(
                self,
                name,
                exact_int(value, name, minimum=0),
            )

    def messages(self, context: str) -> tuple[dict[str, str], ...]:
        if type(context) is not str:
            raise TypeError("prompt context must be an exact string")
        return (
            *(
                {"role": role, "content": content}
                for role, content in self.base_messages
            ),
            {
                "role": self.evidence_message_role,
                "content": self.evidence_prefix + context + self.evidence_suffix,
            },
        )

    def prompt_tokens(self, context: str, *, encoding: str) -> int:
        return count_chat_prompt_token_proxy(
            self.messages(context),
            encoding=encoding,
        )

    @property
    def base_messages_sha256(self) -> str:
        return identity_sha256(
            [
                {"role": role, "content": content}
                for role, content in self.base_messages
            ]
        )

    @property
    def evidence_prefix_sha256(self) -> str:
        return hashlib.sha256(self.evidence_prefix.encode("utf-8")).hexdigest()

    @property
    def evidence_suffix_sha256(self) -> str:
        return hashlib.sha256(self.evidence_suffix.encode("utf-8")).hexdigest()

    def prompt_messages_sha256(self, context: str) -> str:
        return identity_sha256(list(self.messages(context)))


def _atom_sort_key(
    atom: EvidenceAtom,
) -> tuple[int, str, int, int, int, str, str, str]:
    return (*evidence_span_sort_key(atom.span), atom.atom_id)


def _label_scalar(value: object) -> str:
    """Keep untrusted provenance metadata inside one inert label field."""
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("|", "/")
        .replace("]", ")")
        .strip()
    )


def _bundle_labels(
    bundles: Sequence[EvidenceBundle],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    labels = {bundle.bundle_id: f"B{index}" for index, bundle in enumerate(bundles, 1)}
    atom_bundles: dict[str, list[str]] = {}
    for bundle in bundles:
        label = labels[bundle.bundle_id]
        for atom_id in bundle.atom_ids:
            atom_bundles.setdefault(atom_id, []).append(label)
    return labels, {key: tuple(value) for key, value in atom_bundles.items()}


def _validate_render_inputs(
    atoms: tuple[EvidenceAtom, ...],
    bundles: tuple[EvidenceBundle, ...],
) -> None:
    for atom in atoms:
        if type(atom) is not EvidenceAtom:
            raise TypeError("renderer atoms must be exact EvidenceAtom values")
        if type(atom.span) is not EvidenceSpan:
            raise TypeError("renderer atom spans must be exact EvidenceSpan values")
        for name, value in (
            ("atom_id", atom.atom_id),
            ("text", atom.text),
            ("label", atom.label),
        ):
            if type(value) is not str:
                raise TypeError(f"renderer atom {name} must be an exact string")
        for name, value in (
            ("role", atom.role),
            ("created_at", atom.created_at),
            ("source_id", atom.span.source_id),
            ("turn_id", atom.span.turn_id),
            ("span_role", atom.span.role),
            ("span_created_at", atom.span.created_at),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(
                    f"renderer atom {name} must be an exact optional string"
                )
        for name, value in (
            ("start_char", atom.span.start_char),
            ("end_char", atom.span.end_char),
            ("ordinal", atom.span.ordinal),
            ("turn_start_char", atom.span.turn_start_char),
        ):
            if type(value) is not int:
                raise TypeError(f"renderer atom {name} must be an exact integer")
        for name, value in (
            ("chunk_id", atom.span.chunk_id),
            ("quote_sha256", atom.span.quote_sha256),
        ):
            if type(value) is not str:
                raise TypeError(f"renderer span {name} must be an exact string")
    for bundle in bundles:
        if type(bundle) is not EvidenceBundle:
            raise TypeError("renderer bundles must be exact EvidenceBundle values")
        if type(bundle.bundle_id) is not str:
            raise TypeError("renderer bundle IDs must be exact strings")
        for name in (
            "atom_ids",
            "obligation_ids",
            "unit_ids",
            "relation_ids",
        ):
            values = getattr(bundle, name)
            if type(values) is not tuple or any(
                type(value) is not str for value in values
            ):
                raise TypeError(
                    f"renderer bundle {name} must be an exact tuple of strings"
                )


def render_evidence_context(
    atoms: Sequence[EvidenceAtom],
    bundles: Sequence[EvidenceBundle],
) -> str:
    """Render one deduplicated packet while preserving bundle membership."""
    normalized_atoms = tuple(atoms)
    normalized_bundles = tuple(bundles)
    _validate_render_inputs(normalized_atoms, normalized_bundles)
    ordered = tuple(sorted(normalized_atoms, key=_atom_sort_key))
    groups = () if not ordered else (ordered,)
    return _render_evidence_groups(
        groups,
        normalized_bundles,
        group_headings=False,
    )


def _render_evidence_groups(
    atom_groups: tuple[tuple[EvidenceAtom, ...], ...],
    bundles: tuple[EvidenceBundle, ...],
    *,
    group_headings: bool,
) -> str:
    """Shared byte grammar for canonical and explicit grouped rendering."""
    if not atom_groups:
        return ""
    labels, atom_bundles = _bundle_labels(bundles)
    lines = [_HEADER.rstrip("\n")]
    evidence_index = 1
    for group_index, group in enumerate(atom_groups, 1):
        if group_headings:
            lines.append(f"### Evidence group G{group_index}")
        for atom in group:
            fields = [f"E{evidence_index}"]
            memberships = atom_bundles.get(atom.atom_id, ())
            if memberships:
                fields.append(f"bundles={','.join(memberships)}")
            fields.extend(
                (
                    f"source={_label_scalar(atom.span.source_id or 'unknown')}",
                    f"ordinal={int(atom.span.ordinal)}",
                    f"chunk={_label_scalar(atom.span.chunk_id)}",
                )
            )
            if atom.role:
                fields.append(f"role={_label_scalar(atom.role)}")
            if atom.created_at:
                fields.append(f"date={_label_scalar(atom.created_at)}")
            fields.append(f"label={_label_scalar(atom.label)}")
            lines.append(f"[{' | '.join(fields)}]\n{atom.text}")
            evidence_index += 1
    if labels:
        membership_lines = []
        for bundle in bundles:
            obligation_text = ",".join(
                _label_scalar(item) for item in bundle.obligation_ids
            ) or "none"
            membership_lines.append(
                f"{labels[bundle.bundle_id]}={_label_scalar(bundle.bundle_id)}; "
                f"obligations={obligation_text}"
            )
        lines.append("Bundle map:\n" + "\n".join(membership_lines))
    return "\n\n".join(lines)


def render_grouped_evidence_context(
    atoms: Sequence[EvidenceAtom],
    bundles: Sequence[EvidenceBundle],
    atom_groups: Sequence[Sequence[str]],
) -> str:
    """Render exact atoms in an explicit one-time grouped order.

    This is the narrow post-retrieval fusion seam.  Unlike
    :func:`render_evidence_context`, it never applies the canonical source
    sort: group and atom order are consumed exactly as supplied.  The fixed
    group headings are ordinal presentation structure only; they expose no
    latent slot, score, label, or inferred relation.
    """
    normalized_atoms = tuple(atoms)
    normalized_bundles = tuple(bundles)
    normalized_groups = tuple(tuple(group) for group in atom_groups)
    _validate_render_inputs(normalized_atoms, normalized_bundles)
    if not normalized_atoms:
        if normalized_groups:
            raise ValueError("empty evidence cannot have render groups")
        return ""
    if not normalized_groups or any(not group for group in normalized_groups):
        raise ValueError("grouped renderer requires non-empty atom groups")
    if any(type(atom_id) is not str for group in normalized_groups for atom_id in group):
        raise TypeError("grouped renderer atom IDs must be exact strings")
    atom_by_id = {atom.atom_id: atom for atom in normalized_atoms}
    if len(atom_by_id) != len(normalized_atoms):
        raise ValueError("grouped renderer atom IDs must be unique")
    rendered_ids = tuple(atom_id for group in normalized_groups for atom_id in group)
    if (
        len(rendered_ids) != len(normalized_atoms)
        or len(set(rendered_ids)) != len(rendered_ids)
        or set(rendered_ids) != set(atom_by_id)
    ):
        raise ValueError("render groups must partition the exact atom set once")

    ordered_groups = tuple(
        tuple(atom_by_id[atom_id] for atom_id in atom_ids)
        for atom_ids in normalized_groups
    )
    return _render_evidence_groups(
        ordered_groups,
        normalized_bundles,
        group_headings=True,
    )


def _state_order_key(
    state: _BeamState,
    *,
    required_total: frozenset[str],
) -> tuple[int, float, float, float, int, int, tuple[str, ...]]:
    """Ascending beam objective: lexicographically smaller states are better."""
    return (
        -int(required_total <= state.obligation_ids),
        -state.required_weight,
        -state.desired_weight,
        -state.utility,
        -state.direct_raw_count,
        _state_budget_cost(state),
        state.bundle_ids,
    )


def _prune_beam(
    states: Iterable[_BeamState],
    *,
    required_total: frozenset[str],
    beam_width: int,
) -> list[_BeamState]:
    # Two different paths can reach the same selected bundle set.  Retain only
    # the best canonical state before the bounded sort.  The dedup dict keys on
    # bundle_ids, so the key's bundle-ID tie-break element is inert in this
    # comparison and only the numeric prefix decides.
    unique: dict[tuple[str, ...], _BeamState] = {}
    for state in states:
        current = unique.get(state.bundle_ids)
        if current is None or _state_order_key(
            state, required_total=required_total
        ) < _state_order_key(current, required_total=required_total):
            unique[state.bundle_ids] = state
    ordered = sorted(
        unique.values(),
        key=lambda state: _state_order_key(state, required_total=required_total),
    )
    return ordered[:beam_width]


def _state_budget_cost(state: _BeamState) -> int:
    return (
        state.prompt_token_count
        if state.prompt_token_count is not None
        else state.token_count
    )


def _measure(
    atoms: Sequence[EvidenceAtom],
    bundles: Sequence[EvidenceBundle],
    *,
    encoding: str,
    prompt_budget: EvidencePromptBudget | None,
    max_context_tokens: int,
) -> tuple[str, int, int | None]:
    """Render one candidate context and measure both budgets exactly once.

    The chat-prompt proxy is skipped for contexts that already exceed the hard
    context ceiling: every caller rejects those on the context count alone.
    """
    context = render_evidence_context(atoms, bundles)
    tokens = count_tokens(context, encoding=encoding)
    prompt_tokens = (
        None
        if prompt_budget is None or tokens > max_context_tokens
        else prompt_budget.prompt_tokens(context, encoding=encoding)
    )
    return context, tokens, prompt_tokens


def _prompt_overflow(
    prompt_tokens: int | None,
    prompt_budget: EvidencePromptBudget | None,
) -> bool:
    """True when the exact chat prompt plus output reserve exceeds its ceiling."""
    return (
        prompt_budget is not None
        and prompt_tokens is not None
        and prompt_tokens + prompt_budget.output_token_reserve
        > prompt_budget.max_prompt_tokens
    )


def _required_proof_ids(program: QueryProgram) -> frozenset[str]:
    """Return required obligations plus their transitive dependencies."""

    by_id = {
        obligation.obligation_id: obligation for obligation in program.obligations
    }
    required = {
        obligation.obligation_id
        for obligation in program.obligations
        if obligation.required
    }
    pending = list(required)
    while pending:
        obligation_id = pending.pop()
        for dependency in by_id[obligation_id].dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return frozenset(required)


def _proved_obligation_ids(
    selected_bundle_ids: Iterable[str],
    *,
    plan: ClosurePlan,
    bundle_by_id: Mapping[str, EvidenceBundle],
) -> frozenset[str]:
    """Bind packet proof to the exact owners selected by closure results."""

    selected = set(selected_bundle_ids)
    result_by_id = {
        result.obligation_id: result for result in plan.obligation_results
    }
    base_proved: set[str] = set()
    for obligation in plan.query_program.obligations:
        result = result_by_id[obligation.obligation_id]
        if result.status != "satisfied":
            continue
        eligible = tuple(
            bundle_by_id[bundle_id]
            for bundle_id in result.bundle_ids
            if bundle_id in selected
            and bundle_id in bundle_by_id
            and obligation.obligation_id
            in bundle_by_id[bundle_id].obligation_ids
        )
        supported_units = {
            unit_id for bundle in eligible for unit_id in bundle.unit_ids
        }
        supported_relations = {
            relation_id for bundle in eligible for relation_id in bundle.relation_ids
        }
        expected_units = set(result.unit_ids)
        expected_relations = set(result.relation_ids)
        if not expected_units <= supported_units:
            continue
        if not expected_relations <= supported_relations:
            continue
        if expected_units:
            support_count = len(expected_units)
        elif expected_relations:
            support_count = len(expected_relations)
        else:
            expected_bundles = set(result.bundle_ids)
            selected_eligible_ids = {bundle.bundle_id for bundle in eligible}
            if not expected_bundles or not expected_bundles <= selected_eligible_ids:
                continue
            support_count = len(expected_bundles)
        if support_count >= obligation.min_count:
            base_proved.add(obligation.obligation_id)

    # QueryProgram validates an acyclic dependency graph. Iterate so program
    # declaration order does not affect which supported obligations are proved.
    proved: set[str] = set()
    changed = True
    while changed:
        changed = False
        for obligation in plan.query_program.obligations:
            if (
                obligation.obligation_id in base_proved
                and obligation.obligation_id not in proved
                and set(obligation.dependencies) <= proved
            ):
                proved.add(obligation.obligation_id)
                changed = True
    return frozenset(proved)


def _direct_raw_bundle_ids(
    plan: ClosurePlan,
    atom_by_id: Mapping[str, EvidenceAtom],
) -> frozenset[str]:
    """Identify fail-open direct bundles without treating them as proof."""

    direct_chunks = set(plan.direct_chunk_ids)
    if not direct_chunks:
        return frozenset()
    return frozenset(
        bundle.bundle_id
        for bundle in plan.bundles
        if not bundle.obligation_ids
        and not bundle.relation_ids
        and bundle.atom_ids
        and all(
            atom_id in atom_by_id
            and atom_by_id[atom_id].span.chunk_id in direct_chunks
            and atom_by_id[atom_id].label
            == f"direct:{atom_by_id[atom_id].span.chunk_id}"
            for atom_id in bundle.atom_ids
        )
    )


def normalize_evidence_prompt_budget(
    *,
    base_messages: Sequence[Mapping[str, str]] | None,
    evidence_message_role: str,
    evidence_prefix: str,
    evidence_suffix: str,
    max_prompt_tokens: int | None,
    output_token_reserve: int,
) -> EvidencePromptBudget | None:
    reserve = exact_int(output_token_reserve, "output_token_reserve", minimum=0)
    if not isinstance(evidence_message_role, str):
        raise TypeError("evidence_message_role must be a string")
    role = evidence_message_role.strip()
    if not role:
        raise ValueError("evidence_message_role must be non-empty")
    if role != evidence_message_role:
        raise ValueError("evidence_message_role cannot have surrounding whitespace")
    if not isinstance(evidence_prefix, str) or not isinstance(evidence_suffix, str):
        raise TypeError("evidence_prefix and evidence_suffix must be strings")

    raw_messages = tuple(base_messages or ())
    if max_prompt_tokens is None:
        if raw_messages or evidence_prefix or evidence_suffix or reserve or role != "user":
            raise ValueError(
                "max_prompt_tokens is required when chat-prompt framing is supplied"
            )
        return None

    maximum = exact_int(max_prompt_tokens, "max_prompt_tokens", minimum=0)
    normalized_messages: list[tuple[str, str]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"base message {index} must be a mapping")
        if set(message) - {"role", "content"}:
            raise ValueError(
                "base messages may contain only exact role and content fields"
            )
        if "role" not in message or "content" not in message:
            raise ValueError("base messages require role and content")
        message_role = message["role"]
        content = message["content"]
        if not isinstance(message_role, str) or not message_role.strip():
            raise ValueError(f"base message {index} role must be a non-empty string")
        if message_role != message_role.strip():
            raise ValueError(
                f"base message {index} role cannot have surrounding whitespace"
            )
        if not isinstance(content, str):
            raise TypeError(f"base message {index} content must be a string")
        normalized_messages.append((message_role, content))
    return EvidencePromptBudget(
        base_messages=tuple(normalized_messages),
        evidence_message_role=role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
        max_prompt_tokens=maximum,
        output_token_reserve=reserve,
    )


def pack_evidence_plan(
    plan: ClosurePlan,
    *,
    max_context_tokens: int,
    encoding: str = "cl100k_base",
    base_messages: Sequence[Mapping[str, str]] | None = None,
    evidence_message_role: str = "user",
    evidence_prefix: str = "",
    evidence_suffix: str = "",
    max_prompt_tokens: int | None = None,
    output_token_reserve: int = 0,
) -> EvidencePacket:
    """Select whole bundles under context and optional full-chat ceilings.

    Bundle atoms are all-or-nothing.  Shared atoms are rendered once and their
    exact union cost is recomputed for every admitted beam state.  The beam's
    lexicographic objective prioritizes complete required proof bound to the
    exact closure-result owners, then required weight, desired weight, utility,
    fail-open direct evidence, and finally lower token cost.

    When ``max_prompt_tokens`` is supplied, it is the total local request
    workspace ceiling: the exact chat-prompt proxy plus
    ``output_token_reserve`` must fit.  Evidence prefix/context/suffix are
    concatenated before BPE counting, so cross-boundary token merges and the
    chat framing allowance are both included during every beam admission.
    """
    max_context_tokens = exact_int(
        max_context_tokens,
        "max_context_tokens",
        minimum=0,
    )
    prompt_budget = normalize_evidence_prompt_budget(
        base_messages=base_messages,
        evidence_message_role=evidence_message_role,
        evidence_prefix=evidence_prefix,
        evidence_suffix=evidence_suffix,
        max_prompt_tokens=max_prompt_tokens,
        output_token_reserve=output_token_reserve,
    )

    atom_by_id = {atom.atom_id: atom for atom in plan.atoms}
    obligation_by_id = {
        obligation.obligation_id: obligation
        for obligation in plan.query_program.obligations
    }
    required_total = _required_proof_ids(plan.query_program)
    bundle_by_id = {bundle.bundle_id: bundle for bundle in plan.bundles}
    direct_raw_bundle_ids = _direct_raw_bundle_ids(plan, atom_by_id)
    ranked_candidates = tuple(
        sorted(
            plan.bundles,
            key=lambda item: (
                not item.required,
                -item.utility,
                item.bundle_id,
            ),
        )
    )
    candidates = ranked_candidates[: plan.policy.max_bundles]
    candidate_cap_drops = ranked_candidates[plan.policy.max_bundles :]

    _, empty_tokens, empty_prompt_tokens = _measure(
        (),
        (),
        encoding=encoding,
        prompt_budget=prompt_budget,
        max_context_tokens=max_context_tokens,
    )
    if _prompt_overflow(empty_prompt_tokens, prompt_budget):
        raise ValueError(
            "base chat prompt, evidence framing, and output reserve exceed "
            "max_prompt_tokens before evidence admission"
        )
    beam = [
        _BeamState(
            bundle_ids=(),
            atom_ids=frozenset(),
            obligation_ids=frozenset(),
            required_weight=0.0,
            desired_weight=0.0,
            utility=0.0,
            direct_raw_count=0,
            token_count=empty_tokens,
            prompt_token_count=empty_prompt_tokens,
        )
    ]

    for bundle in candidates:
        expanded = list(beam)
        for state in beam:
            next_bundle_ids = state.bundle_ids + (bundle.bundle_id,)
            next_atom_ids = state.atom_ids | frozenset(bundle.atom_ids)
            next_obligation_ids = _proved_obligation_ids(
                next_bundle_ids,
                plan=plan,
                bundle_by_id=bundle_by_id,
            )
            selected_bundles = tuple(
                candidate
                for candidate in candidates
                if candidate.bundle_id in next_bundle_ids
            )
            selected_atoms = tuple(
                atom_by_id[atom_id]
                for atom_id in next_atom_ids
            )
            _, tokens, prompt_tokens = _measure(
                selected_atoms,
                selected_bundles,
                encoding=encoding,
                prompt_budget=prompt_budget,
                max_context_tokens=max_context_tokens,
            )
            if tokens > max_context_tokens:
                continue
            if _prompt_overflow(prompt_tokens, prompt_budget):
                continue
            new_obligations = next_obligation_ids - state.obligation_ids
            required_gain = sum(
                obligation_by_id[item].weight
                for item in new_obligations
                if item in required_total
            )
            desired_gain = sum(
                obligation_by_id[item].weight
                for item in new_obligations
                if item not in required_total
            )
            expanded.append(
                _BeamState(
                    bundle_ids=next_bundle_ids,
                    atom_ids=next_atom_ids,
                    obligation_ids=next_obligation_ids,
                    required_weight=state.required_weight + required_gain,
                    desired_weight=state.desired_weight + desired_gain,
                    utility=state.utility + bundle.utility,
                    direct_raw_count=len(
                        set(next_bundle_ids) & direct_raw_bundle_ids
                    ),
                    token_count=tokens,
                    prompt_token_count=prompt_tokens,
                )
            )
        beam = _prune_beam(
            expanded,
            required_total=required_total,
            beam_width=plan.policy.beam_width,
        )

    best = _prune_beam(
        beam,
        required_total=required_total,
        beam_width=1,
    )[0]
    selected_id_set = set(best.bundle_ids)
    selected_bundles = tuple(
        bundle for bundle in candidates if bundle.bundle_id in selected_id_set
    )
    selected_atoms = tuple(
        sorted(
            (atom_by_id[atom_id] for atom_id in best.atom_ids),
            key=_atom_sort_key,
        )
    )
    context, exact_tokens, exact_prompt_tokens = _measure(
        selected_atoms,
        selected_bundles,
        encoding=encoding,
        prompt_budget=prompt_budget,
        max_context_tokens=max_context_tokens,
    )
    if exact_tokens > max_context_tokens:  # pragma: no cover - defensive
        raise AssertionError("atomic evidence packer exceeded its hard budget")
    if _prompt_overflow(
        exact_prompt_tokens, prompt_budget
    ):  # pragma: no cover - defensive
        raise AssertionError("atomic evidence packer exceeded its hard prompt budget")

    dropped: dict[str, str] = {
        bundle.bundle_id: "candidate_cap" for bundle in candidate_cap_drops
    }
    for bundle in candidates:
        if bundle.bundle_id in selected_id_set:
            continue
        trial_bundle_ids = selected_id_set | {bundle.bundle_id}
        trial_bundles = tuple(
            candidate
            for candidate in candidates
            if candidate.bundle_id in trial_bundle_ids
        )
        trial_atom_ids = set(best.atom_ids) | set(bundle.atom_ids)
        trial_atoms = tuple(atom_by_id[item] for item in trial_atom_ids)
        _, trial_tokens, trial_prompt_tokens = _measure(
            trial_atoms,
            trial_bundles,
            encoding=encoding,
            prompt_budget=prompt_budget,
            max_context_tokens=max_context_tokens,
        )
        if trial_tokens > max_context_tokens:
            dropped[bundle.bundle_id] = "hard_budget"
            continue
        if _prompt_overflow(trial_prompt_tokens, prompt_budget):
            dropped[bundle.bundle_id] = "hard_prompt_budget"
            continue
        dropped[bundle.bundle_id] = "lower_utility"

    required_selected = required_total <= best.obligation_ids
    complete = plan.complete_claimed and required_selected
    if not required_selected:
        stopping_reason = "budget_impossible"
    elif plan.stopping_reason == "complete":
        stopping_reason = "complete"
    else:
        stopping_reason = plan.stopping_reason
    tokenizer_body = tokenizer_proxy_identity(encoding)
    tokenizer_identity = identity_sha256(tokenizer_body)
    receipt = ClosureReceipt(
        plan_sha256=plan.plan_sha256,
        context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
        selected_bundle_ids=tuple(item.bundle_id for item in selected_bundles),
        selected_atom_ids=tuple(item.atom_id for item in selected_atoms),
        dropped_bundle_reasons=dropped,
        context_token_proxy=exact_tokens,
        max_context_token_proxy=max_context_tokens,
        tokenizer_identity=f"{tokenizer_body['encoding']}:{tokenizer_identity}",
        stopping_reason=stopping_reason,
        complete_claimed=complete,
        retained_request_token_state_bytes=0,
        prompt_token_proxy=exact_prompt_tokens,
        max_prompt_token_proxy=(
            None if prompt_budget is None else prompt_budget.max_prompt_tokens
        ),
        responder_output_token_reserve=(
            0 if prompt_budget is None else prompt_budget.output_token_reserve
        ),
        prompt_workspace_token_proxy=(
            None
            if prompt_budget is None or exact_prompt_tokens is None
            else exact_prompt_tokens + prompt_budget.output_token_reserve
        ),
        base_messages_sha256=(
            None if prompt_budget is None else prompt_budget.base_messages_sha256
        ),
        evidence_message_role=(
            None if prompt_budget is None else prompt_budget.evidence_message_role
        ),
        evidence_prefix_sha256=(
            None if prompt_budget is None else prompt_budget.evidence_prefix_sha256
        ),
        evidence_suffix_sha256=(
            None if prompt_budget is None else prompt_budget.evidence_suffix_sha256
        ),
        prompt_messages_sha256=(
            None
            if prompt_budget is None
            else prompt_budget.prompt_messages_sha256(context)
        ),
    )
    return EvidencePacket(
        context=context,
        atoms=selected_atoms,
        bundles=selected_bundles,
        receipt=receipt,
    )


def packet_identity(packet: EvidencePacket) -> str:
    """Return a text-free identity for a fully packed packet."""
    return identity_sha256(
        {
            "receipt_sha256": packet.receipt.receipt_sha256,
            "context_sha256": hashlib.sha256(
                packet.context.encode("utf-8")
            ).hexdigest(),
            "atom_ids": [item.atom_id for item in packet.atoms],
            "bundle_ids": [item.bundle_id for item in packet.bundles],
            "tokenizer": packet.receipt.tokenizer_identity,
        }
    )


__all__ = [
    "EvidencePromptBudget",
    "normalize_evidence_prompt_budget",
    "pack_evidence_plan",
    "packet_identity",
    "render_evidence_context",
    "render_grouped_evidence_context",
]
