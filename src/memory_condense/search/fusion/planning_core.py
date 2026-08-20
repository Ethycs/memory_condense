"""Cold-safe topology and bounded route-reduction helpers.

The functions in this module are shared by the CPU-hashed reference planner
and the resident matched-pair builder.  Their behavior is intentionally the
same as the original reference implementation.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from itertools import combinations
from typing import Sequence

from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    ExtractiveGroup,
    FusionCaps,
    LatentMembership,
)
from memory_condense.search.fusion.tensor_identity import CanonicalTensor


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _adjacency(
    atom_ids: Sequence[str],
    hyperedges: Sequence[AuthoritativeHyperedge],
) -> dict[str, set[str]]:
    adjacency = {atom_id: set() for atom_id in atom_ids}
    for edge in hyperedges:
        for left, right in combinations(edge.atom_ids, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def _topology_atom_groups(
    atom_ids: tuple[str, ...],
    hyperedges: tuple[AuthoritativeHyperedge, ...],
    caps: FusionCaps,
) -> tuple[tuple[ExtractiveGroup, ...], tuple[str, ...], dict[str, int]]:
    if len(hyperedges) > caps.max_hyperedges:
        raise MemoryError("selected bundle hyperedges exceed max_hyperedges")
    topology_links = sum(
        len(edge.atom_ids) * (len(edge.atom_ids) - 1) // 2
        for edge in hyperedges
    )
    if topology_links > caps.max_topology_links:
        raise MemoryError("selected bundle co-memberships exceed max_topology_links")
    adjacency = _adjacency(atom_ids, hyperedges)
    source_index = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    degree = {atom_id: len(neighbors) for atom_id, neighbors in adjacency.items()}
    unseen = set(atom_ids)
    ordered_components: list[tuple[str, ...]] = []
    while unseen:
        component_start = min(unseen, key=source_index.__getitem__)
        component: set[str] = set()
        pending = [component_start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        unseen -= component
        medoid = min(
            component,
            key=lambda atom_id: (-degree[atom_id], source_index[atom_id]),
        )
        traversal: list[str] = []
        visited = {medoid}
        frontier: deque[str] = deque((medoid,))
        while frontier:
            current = frontier.popleft()
            traversal.append(current)
            neighbors = sorted(
                adjacency[current] - visited,
                key=lambda atom_id: (-degree[atom_id], source_index[atom_id]),
            )
            visited.update(neighbors)
            frontier.extend(neighbors)
        # Defensive fallback for a malformed traversal; components are derived
        # from the same adjacency, so this should normally be empty.
        traversal.extend(
            sorted(component - visited, key=source_index.__getitem__)
        )
        ordered_components.append(tuple(traversal))

    ordered_components.sort(
        key=lambda component: min(source_index[item] for item in component)
    )
    group_count = sum(
        (len(component) + caps.max_group_atoms - 1) // caps.max_group_atoms
        for component in ordered_components
    )
    if group_count > caps.max_groups:
        raise MemoryError("extractive topology groups exceed max_groups")
    chunks = [
        component[start : start + caps.max_group_atoms]
        for component in ordered_components
        for start in range(0, len(component), caps.max_group_atoms)
    ]
    groups = tuple(
        ExtractiveGroup(group_index=index, atom_ids=chunk)
        for index, chunk in enumerate(chunks)
    )
    atom_order = tuple(atom_id for chunk in chunks for atom_id in chunk)
    return groups, atom_order, degree


def _preflight_topology(
    hyperedges: Sequence[AuthoritativeHyperedge],
    groups: Sequence[ExtractiveGroup],
    caps: FusionCaps,
) -> None:
    if len(hyperedges) > caps.max_hyperedges:
        raise MemoryError("selected bundle hyperedges exceed max_hyperedges")
    topology_links = sum(
        len(edge.atom_ids) * (len(edge.atom_ids) - 1) // 2
        for edge in hyperedges
    )
    if topology_links > caps.max_topology_links:
        raise MemoryError("selected bundle co-memberships exceed max_topology_links")
    if len(groups) > caps.max_groups:
        raise MemoryError("extractive topology groups exceed max_groups")


def _matrix_rows(tensor: CanonicalTensor) -> tuple[tuple[float, ...], ...]:
    if len(tensor.shape) != 2:
        raise ValueError("route attention must be a two-dimensional matrix")
    rows, columns = tensor.shape
    return tuple(
        tensor.flat_values[start : start + columns]
        for start in range(0, rows * columns, columns)
    )


def _validate_attention_rows(
    rows: Sequence[Sequence[float]],
    *,
    label: str,
    source_dtype: str,
) -> None:
    # Canonical values are float32, but the attention weights may have been
    # rounded in a lower-precision resident execution before the bounded D2H
    # copy.  Keep float32 as strict as the reference planner while admitting
    # the normal unit-sum error of IEEE fp16/bf16 softmax outputs.
    if source_dtype == "torch.bfloat16":
        absolute_tolerance = 8e-3
    elif source_dtype == "torch.float16":
        absolute_tolerance = 2e-3
    elif source_dtype in {"torch.float32", "torch.float64"}:
        absolute_tolerance = 1e-4
    else:
        raise ValueError("route attention source dtype is unsupported")
    for row in rows:
        if any(value < 0.0 or value > 1.0 for value in row):
            raise ValueError(f"{label} weights must lie in [0, 1]")
        if not math.isclose(
            sum(row),
            1.0,
            rel_tol=1e-3,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(f"{label} rows must be softmax-normalized")


def _latent_memberships_and_groups(
    atom_ids: tuple[str, ...],
    extraction: CanonicalTensor,
    reinjection: CanonicalTensor,
    degree: dict[str, int],
    caps: FusionCaps,
    *,
    source_dtype: str,
) -> tuple[tuple[LatentMembership, ...], tuple[ExtractiveGroup, ...], tuple[str, ...]]:
    latent_count, atom_count = extraction.shape
    extraction_rows = _matrix_rows(extraction)
    reinjection_rows = _matrix_rows(reinjection)
    _validate_attention_rows(
        extraction_rows,
        label="extraction attention",
        source_dtype=source_dtype,
    )
    _validate_attention_rows(
        reinjection_rows,
        label="reinjection attention",
        source_dtype=source_dtype,
    )
    keep = min(caps.max_latent_memberships_per_atom, latent_count)
    source_index = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    memberships: list[LatentMembership] = []
    primary_by_atom: dict[str, LatentMembership] = {}
    for atom_index, atom_id in enumerate(atom_ids):
        routes = []
        for latent_index in range(latent_count):
            extract_weight = extraction_rows[latent_index][atom_index]
            reinject_weight = reinjection_rows[atom_index][latent_index]
            joint = _float32(math.sqrt(extract_weight * reinject_weight))
            routes.append((joint, latent_index, extract_weight, reinject_weight))
        routes.sort(key=lambda item: (-item[0], item[1]))
        retained = tuple(
            LatentMembership(
                atom_id=atom_id,
                latent_index=latent_index,
                extraction_weight=extract_weight,
                reinjection_weight=reinject_weight,
                joint_weight=joint,
            )
            for joint, latent_index, extract_weight, reinject_weight in routes[:keep]
        )
        memberships.extend(retained)
        primary_by_atom[atom_id] = retained[0]

    atoms_by_latent: dict[int, list[str]] = {}
    for atom_id, membership in primary_by_atom.items():
        atoms_by_latent.setdefault(membership.latent_index, []).append(atom_id)
    latent_order = sorted(
        atoms_by_latent,
        key=lambda latent_index: (
            -sum(
                primary_by_atom[item].joint_weight
                for item in atoms_by_latent[latent_index]
            ),
            latent_index,
        ),
    )
    group_count = sum(
        (len(atoms_by_latent[latent_index]) + caps.max_group_atoms - 1)
        // caps.max_group_atoms
        for latent_index in latent_order
    )
    if group_count > caps.max_groups:
        raise MemoryError("extractive latent groups exceed max_groups")
    grouped_rows: list[tuple[int, tuple[str, ...]]] = []
    for latent_index in latent_order:
        ordered = tuple(
            sorted(
                atoms_by_latent[latent_index],
                key=lambda atom_id: (
                    -primary_by_atom[atom_id].joint_weight,
                    -degree[atom_id],
                    source_index[atom_id],
                ),
            )
        )
        grouped_rows.extend(
            (latent_index, ordered[start : start + caps.max_group_atoms])
            for start in range(0, len(ordered), caps.max_group_atoms)
        )
    groups = tuple(
        ExtractiveGroup(
            group_index=index,
            atom_ids=group_atoms,
            latent_index=latent_index,
        )
        for index, (latent_index, group_atoms) in enumerate(grouped_rows)
    )
    atom_order = tuple(atom_id for group in groups for atom_id in group.atom_ids)
    return tuple(memberships), groups, atom_order


__all__ = [
    "_latent_memberships_and_groups",
    "_preflight_topology",
    "_topology_atom_groups",
]
