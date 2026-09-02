from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval import semantic_residual_search as residual_module
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_residual_search import (
    SemanticResidualPolicy,
    TYPED_ADAPTER_MECHANISM_ID,
    adapt_semantic_residual_to_typed_contribution,
    build_semantic_residual_index,
    compile_semantic_residual_query,
    load_stored_chunk_vectors,
    replay_semantic_residual_search,
    search_semantic_residual,
    semantic_residual_query_facets,
    validate_semantic_residual_search_projection,
)
from tools.matched_eval.typed_operator_adapter import (
    FrontierMode,
    merge_typed_evidence_contributions,
)


BASE = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_cache(
    path: Path,
    rows: list[tuple[str, str, datetime, str]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for ordinal, (source_id, text, created_at, role) in enumerate(rows):
        turn = transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{ordinal}",
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha(f"store:{path.name}")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"snapshot:{path.name}"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        return cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"database:{path.name}"),
            source_store_receipt_sha256=store_receipt,
        )


def _build(
    tmp_path: Path,
    name: str,
    rows: list[tuple[str, str, datetime, str]],
    vectors: dict[str, list[float] | None],
    *,
    max_cell_tokens: int = 256,
    payload_token_cap: int = 500,
    floor: float = 0.4,
):
    cache = _write_cache(tmp_path / f"{name}.db", rows)
    window_index = build_full_store_window_index(cache)
    policy = SemanticResidualPolicy(
        max_cell_tokens=max_cell_tokens,
        payload_token_cap=payload_token_cap,
        cosine_upper_bound_floor=floor,
    )
    return build_semantic_residual_index(window_index, vectors, policy=policy)


def _build_stored(
    tmp_path: Path,
    name: str,
    rows: list[tuple[str, str, datetime, str]],
    chunk_vectors: dict[int, list[float] | None],
    *,
    max_cell_tokens: int = 256,
    payload_token_cap: int = 500,
    floor: float = 0.4,
):
    path = tmp_path / f"{name}.db"
    cache = _write_cache(path, rows)
    with Database(path) as writable:
        for ordinal, vector in chunk_vectors.items():
            if vector is None:
                continue
            writable.execute(
                "UPDATE chunks SET embedding = ?, hnsw_label = ? WHERE chunk_id = ?",
                (
                    np.asarray(vector, dtype=np.float32).tobytes(),
                    ordinal + 1,
                    f"chunk-{ordinal}",
                ),
            )
        writable.commit()
    window_index = build_full_store_window_index(cache)
    with Database(path, read_only=True) as readonly:
        stored = load_stored_chunk_vectors(readonly, window_index)
    policy = SemanticResidualPolicy(
        max_cell_tokens=max_cell_tokens,
        payload_token_cap=payload_token_cap,
        cosine_upper_bound_floor=floor,
    )
    return build_semantic_residual_index(window_index, stored, policy=policy)


def _compile(index, question: str, directions: list[list[float]]):
    facets = semantic_residual_query_facets(question)
    vectors = [directions[position % len(directions)] for position in range(len(facets))]
    return compile_semantic_residual_query(index, question, query_vectors=vectors)


def _source_ids(result) -> set[str]:
    return {row.source_id for row in result.local_bindings}


def _filler(word: str, count: int = 180) -> str:
    return " ".join(f"{word}{index}" for index in range(count)) + "."


def test_classifier_materializes_manifest_index_once_without_one_use_term_caches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [
        ("p0::alpha", "I bought a blue touring bicycle.", BASE, "user"),
        (
            "p1::beta",
            "I replaced the bicycle tires before a long ride.",
            BASE + timedelta(days=1),
            "user",
        ),
        (
            "p2::gamma",
            "I stored the bicycle beside my red helmet.",
            BASE + timedelta(days=2),
            "user",
        ),
    ]
    index = _build(
        tmp_path,
        "manifest-once",
        rows,
        {
            "p0::alpha": [1.0, 0.0],
            "p1::beta": [0.9, 0.1],
            "p2::gamma": [0.8, 0.2],
        },
    )
    descriptor = residual_module.SemanticResidualIndex.manifest_by_node_receipt
    original_getter = descriptor.fget
    assert original_getter is not None
    calls = 0

    def counted_getter(self):
        nonlocal calls
        calls += 1
        return original_getter(self)

    monkeypatch.setattr(
        residual_module.SemanticResidualIndex,
        "manifest_by_node_receipt",
        property(counted_getter),
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What color was my touring bicycle?"
    )
    query = _compile(index, question, [[1.0, 0.0]])
    classifier = residual_module._ConservativeResidualClassifier(index, query)
    assert not hasattr(classifier, "_manifest_surface_terms")
    assert not hasattr(classifier, "_manifest_action_concepts")
    classifier.classify(
        question=question,
        node=index.core_tree.root,
        call_ordinal=0,
    )
    assert calls == 1
    calls = 0

    result = search_semantic_residual(
        index,
        query,
    )

    assert result.core_result.classifier_calls > 1
    assert calls == 1


def test_q42_like_absence_keeps_related_sources_and_closes_full_leaf_partition(
    tmp_path: Path,
) -> None:
    rows = [
        (
            "p0::education",
            "I am considering doctoral programs, but I have not selected a university.",
            BASE,
            "user",
        ),
        (
            "p1::conference",
            "I attended the Graph Memory conference and presented a poster there.",
            BASE + timedelta(days=1),
            "user",
        ),
        ("p2::noise", _filler("quartz"), BASE, "user"),
    ]
    index = _build(
        tmp_path,
        "q42",
        rows,
        {
            "p0::education": [1.0, 0.0],
            "p1::conference": [0.95, 0.05],
            "p2::noise": [-1.0, 0.0],
        },
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "Which university did I attend for the Graph Memory conference?"
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))

    assert {"p0::education", "p1::conference"} <= _source_ids(result)
    assert result.classified_frontier.retained_leaf_cell_ids
    assert result.evidence
    assert result.classified_frontier.all_novel_survivors_protected is False
    assert result.provider_projection()["residual_frontier"][
        "all_novel_survivors_protected"
    ] is False
    assert result.classified_frontier.complete_leaf_partition is True
    assert result.classified_frontier.classified_leaf_count == len(index.cells)
    assert (
        set(result.core_result.retained_leaf_cell_ids)
        | set(result.core_result.pruned_leaf_cell_ids)
        == {row.cell_id for row in index.cells}
    )
    assert result.classified_frontier.unresolved_segment_receipt_sha256s

    contribution = adapt_semantic_residual_to_typed_contribution(
        result,
        handle_start=420,
        group_start=420,
    )
    packet = merge_typed_evidence_contributions(
        result.query.operator_spec,
        (contribution,),
    )

    # Search closure proves only that every retained MAY segment was packed.
    # It cannot certify that semantic pruning found every supporting memory,
    # authorize an absence conclusion, or close the common typed frontier.
    assert result.packing_frontier_closed is False
    assert contribution.mechanism_id == TYPED_ADAPTER_MECHANISM_ID
    assert contribution.frontier_mode is FrontierMode.OPEN
    assert contribution.truncated is True
    assert result.query.operator_spec.absence_decision_requires_closed_frontier is True
    assert packet.frontier.mode is FrontierMode.OPEN
    assert packet.frontier.closed is False


def test_q65_multifacet_search_retains_opposite_semantic_root_branches(
    tmp_path: Path,
) -> None:
    rows = [
        (
            "p0::photo",
            "I joined the Shutter Circle online community for photography.",
            BASE,
            "user",
        ),
        (
            "p1::cook",
            "I joined the Pan and Ladle online community for cooking.",
            BASE,
            "user",
        ),
        ("p2::north", _filler("zircon"), BASE, "user"),
        ("p3::south", _filler("xenon"), BASE, "user"),
    ]
    index = _build(
        tmp_path,
        "q65",
        rows,
        {
            "p0::photo": [1.0, 0.0],
            "p1::cook": [-1.0, 0.0],
            "p2::north": [0.0, 1.0],
            "p3::south": [0.0, -1.0],
        },
        payload_token_cap=650,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "Which online communities did I join for photography and cooking?"
    )
    query = _compile(index, question, [[1.0, 0.0], [-1.0, 0.0]])
    result = search_semantic_residual(index, query)
    cell_source = {row.cell_id: row.source_id for row in index.cells}
    child_sources = [
        {cell_source[cell.cell_id] for cell in child.cells}
        for child in index.core_tree.root.children
    ]

    assert any("p0::photo" in sources for sources in child_sources)
    assert any("p1::cook" in sources for sources in child_sources)
    assert next(i for i, sources in enumerate(child_sources) if "p0::photo" in sources) != next(
        i for i, sources in enumerate(child_sources) if "p1::cook" in sources
    )
    assert {"p0::photo", "p1::cook"} <= _source_ids(result)
    assert result.classified_frontier.retained_leaf_cell_ids


def test_q74_needle_audits_low_bound_noise_but_fails_open(
    tmp_path: Path,
) -> None:
    target = "I saved the Mayo Clinic posture reset video on YouTube."
    rows = [
        ("p0::needle", target, BASE, "user"),
        ("p1::noise", _filler("amber"), BASE, "user"),
        ("p2::noise", _filler("cobalt"), BASE, "user"),
        ("p3::noise", _filler("indigo"), BASE, "user"),
    ]
    index = _build(
        tmp_path,
        "q74",
        rows,
        {
            "p0::needle": [1.0, 0.0],
            "p1::noise": [-1.0, 0.0],
            "p2::noise": [-1.0, 0.0],
            "p3::noise": [-1.0, 0.0],
        },
        payload_token_cap=400,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "Which Mayo Clinic posture video did I save on YouTube?"
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))

    assert any(row.quote == target for row in result.evidence)
    assert any(row.reason == "dual_gate" for row in result.decision_audits)
    assert result.core_result.pruned_leaf_cell_ids == ()
    assert result.classified_frontier.closed is False


def test_q79_conflicting_prices_keep_exact_whitespace_and_created_at_chronology(
    tmp_path: Path,
) -> None:
    first = "I said the  handbag cost $800."
    second = "Later I said the handbag budget was $2,000."
    rows = [
        ("p0::price-old", first, BASE, "user"),
        ("p1::price-new", second, BASE + timedelta(days=14), "user"),
        ("p2::noise", _filler("topaz", 260), BASE, "user"),
    ]
    index = _build(
        tmp_path,
        "q79",
        rows,
        {
            "p0::price-old": [1.0, 0.0],
            "p1::price-new": [1.0, 0.0],
            "p2::noise": [-1.0, 0.0],
        },
        payload_token_cap=650,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What is the latest handbag price I mentioned?"
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))
    contribution = adapt_semantic_residual_to_typed_contribution(
        result,
        handle_start=700,
        group_start=700,
    )
    items = contribution.parsed.accepted_items

    assert first in [row.quote for row in result.evidence]
    assert first in [row.summary for row in items]
    assert {800.0, 2000.0} <= {
        row.numeric_value for row in items if row.numeric_value is not None
    }
    assert all(row.date is not None for row in items)
    assert all(row.value_authority.value == "derived" for row in items)
    assert all("date_basis=source_created_at" in (row.relation or "") for row in items)
    assert contribution.frontier_mode is FrontierMode.OPEN
    assert contribution.truncated is True


def test_missing_vector_is_may_answer_and_unknown_overbudget_packs_partial(
    tmp_path: Path,
) -> None:
    rows = [
        ("p0::missing", "The rare heliotrope passphrase was ALPHA-9.", BASE, "user"),
        ("p1::known", "A completely unrelated memo about weather.", BASE, "user"),
    ]
    index = _build(
        tmp_path,
        "missing",
        rows,
        {"p0::missing": None, "p1::known": [-1.0, 0.0]},
        payload_token_cap=800,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What was the heliotrope passphrase?"
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))
    assert "p0::missing" in _source_ids(result)
    assert any(
        audit.reason == "may_answer" and not audit.vector_gate_available
        for audit in result.decision_audits
    )

    over_rows = [
        (f"p{ordinal}::unknown", _filler(f"unknown{ordinal}", 100), BASE, "user")
        for ordinal in range(4)
    ]
    over = _build(
        tmp_path,
        "over",
        over_rows,
        {source_id: None for source_id, _text, _date, _role in over_rows},
        max_cell_tokens=80,
        payload_token_cap=120,
    )
    over_question = (
        "[Question asked at 2026/08/27 12:00] What was the unknown detail?"
    )
    over_result = search_semantic_residual(
        over,
        compile_semantic_residual_query(over, over_question),
    )
    assert over_result.fallback_required is False
    assert over_result.fallback_reason == "none"
    assert over_result.evidence
    assert over_result.attempted_evidence_count > 1
    assert len(over_result.evidence) < over_result.attempted_evidence_count
    assert (
        over_result.packed_residual_evidence_tokens
        <= over_result.residual_evidence_token_cap
        == 120
    )
    assert set(over_result.classified_frontier.unresolved_segment_receipt_sha256s)
    assert over_result.classified_frontier.closed is False

    over_contribution = adapt_semantic_residual_to_typed_contribution(
        over_result,
        handle_start=800,
        group_start=800,
    )
    over_packet = merge_typed_evidence_contributions(
        over_result.query.operator_spec,
        (over_contribution,),
    )
    assert over_contribution.mechanism_id == TYPED_ADAPTER_MECHANISM_ID
    assert over_contribution.frontier_mode is FrontierMode.OPEN
    assert over_contribution.truncated is True
    assert over_packet.frontier.mode is FrontierMode.OPEN
    assert over_packet.frontier.closed is False


def test_stored_chunk_centroids_prevent_source_mean_dilution_between_cells(
    tmp_path: Path,
) -> None:
    source = "p0::multi-cell-history"
    target = "The heliotrope access code is LIME-42."
    rows = [
        (source, target, BASE, "user"),
        (source, "I want to follow up on routine garden watering.", BASE, "user"),
        (source, "I want to follow up on routine pantry shelving.", BASE, "user"),
        (source, "I want to follow up on routine bicycle cleaning.", BASE, "user"),
    ]
    index = _build_stored(
        tmp_path,
        "cell-local-dilution",
        rows,
        {
            0: [1.0, 0.0],
            1: [-1.0, 0.0],
            2: [-1.0, 0.0],
            3: None,
        },
        max_cell_tokens=12,
        payload_token_cap=900,
        floor=0.4,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What heliotrope access code did I want to follow up on?"
    )
    result = search_semantic_residual(
        index, _compile(index, question, [[1.0, 0.0]])
    )
    target_cell = next(
        row for row in index.cells if "heliotrope" in row.core_cell.text
    )
    negative_cells = [
        row
        for row in index.cells
        if row.normalized_source_centroid == (-1.0, 0.0)
    ]
    missing_cell = next(
        row for row in index.cells if "bicycle cleaning" in row.core_cell.text
    )

    assert target_cell.normalized_source_centroid == (1.0, 0.0)
    assert target_cell.cell_id in result.core_result.retained_leaf_cell_ids
    assert all(
        row.cell_id in result.core_result.retained_leaf_cell_ids
        for row in negative_cells
    )
    assert missing_cell.normalized_source_centroid is None
    assert missing_cell.cell_id in result.core_result.retained_leaf_cell_ids
    assert any("LIME-42" in row.quote for row in result.evidence)


def test_idf_specificity_audits_generic_overlap_without_pruning(
    tmp_path: Path,
) -> None:
    rows = [
        (
            "p0::target",
            "I want to follow up on the heliotrope access code LIME-42.",
            BASE,
            "user",
        ),
        (
            "p1::generic",
            "I want to follow up on routine garden watering.",
            BASE,
            "user",
        ),
        (
            "p2::generic",
            "I want to follow up on routine pantry shelving.",
            BASE,
            "user",
        ),
        (
            "p3::generic",
            "I want to follow up on routine bicycle cleaning.",
            BASE,
            "user",
        ),
    ]
    index = _build(
        tmp_path,
        "idf-specificity",
        rows,
        {
            "p0::target": [1.0, 0.0],
            "p1::generic": [-1.0, 0.0],
            "p2::generic": [-1.0, 0.0],
            "p3::generic": [-1.0, 0.0],
        },
        floor=0.4,
        payload_token_cap=800,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What heliotrope access code did I want to follow up on?"
    )
    result = search_semantic_residual(
        index, _compile(index, question, [[1.0, 0.0]])
    )

    assert any("LIME-42" in row.quote for row in result.evidence)
    assert result.core_result.pruned_leaf_cell_ids == ()
    assert any(
        audit.reason == "dual_gate"
        and bool(audit.intersecting_surface_terms)
        and audit.node_specificity_upper_bound is not None
        and audit.node_specificity_upper_bound < audit.specificity_threshold
        for audit in result.decision_audits
    )


def test_post_selection_dedup_has_visible_owner_and_deterministic_projection_replay(
    tmp_path: Path,
) -> None:
    rows = [
        ("p0::one", "The cedar token is K-19.", BASE, "user"),
        ("p1::two", "The cedar token was copied into the travel note.", BASE, "user"),
    ]
    index = _build(
        tmp_path,
        "replay",
        rows,
        {"p0::one": [1.0, 0.0], "p1::two": [1.0, 0.0]},
        payload_token_cap=800,
    )
    question = "[Question asked at 2026/08/27 12:00] What was the cedar token?"
    query = _compile(index, question, [[1.0, 0.0]])
    first = search_semantic_residual(index, query)
    protected = (first.local_bindings[0],)
    deduped = search_semantic_residual(index, query, protected_evidence=protected)

    assert deduped.core_result.retained_leaf_cell_ids == first.core_result.retained_leaf_cell_ids
    assert len(deduped.protected_duplicates) == 1
    assert deduped.protected_duplicates[0].protected_binding_receipt_sha256 == protected[0].receipt_sha256
    assert deduped.classified_frontier.closed is True
    assert deduped.classified_frontier.protected_duplicate_audit_receipt_sha256s
    replayed = replay_semantic_residual_search(
        index,
        query,
        deduped,
        protected_evidence=protected,
    )
    assert replayed.projection() == deduped.projection()
    loaded = validate_semantic_residual_search_projection(
        index,
        query,
        json.loads(json.dumps(deduped.projection())),
        protected_evidence=protected,
    )
    assert loaded.receipt_sha256 == deduped.receipt_sha256
    assert residual_module._farthest_seed_pair(index.cells) == (
        residual_module._farthest_seed_pair_scalar_reference(index.cells)
    )


def test_protected_dedup_occurs_after_tree_selection_before_payload_cap(
    tmp_path: Path,
) -> None:
    protected_detail = " ".join(
        f"protected-checkpoint-{ordinal:03d}" for ordinal in range(320)
    )
    rows = [
        (
            "p0::one",
            (
                "The protected cobalt itinerary contains the cedar token K-19 and "
                f"arrival notes. {protected_detail}"
            ),
            BASE,
            "user",
        ),
        (
            "p1::two",
            "The second cobalt itinerary contains the cedar token Q-27 and departure notes.",
            BASE,
            "user",
        ),
    ]
    cache = _write_cache(tmp_path / "protected-cap.db", rows)
    window = build_full_store_window_index(cache)
    vectors = {"p0::one": [1.0, 0.0], "p1::two": [1.0, 0.0]}
    high = build_semantic_residual_index(
        window,
        vectors,
        policy=SemanticResidualPolicy(payload_token_cap=8_000),
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What cedar tokens were in the cobalt itineraries?"
    )
    high_query = _compile(high, question, [[1.0, 0.0]])
    full = search_semantic_residual(high, high_query)
    protected = (
        next(
            binding
            for binding in full.local_bindings
            if binding.source_id == "p0::one"
        ),
    )
    high_deduped = search_semantic_residual(
        high, high_query, protected_evidence=protected
    )
    deduped_tokens = high_deduped.attempted_provider_payload_tokens
    full_tokens = full.attempted_provider_payload_tokens
    assert full_tokens - deduped_tokens >= 200
    low_cap = deduped_tokens + ((full_tokens - deduped_tokens) // 2)

    low = build_semantic_residual_index(
        window,
        vectors,
        policy=SemanticResidualPolicy(
            payload_token_cap=low_cap
        ),
    )
    low_query = _compile(low, question, [[1.0, 0.0]])
    without_owner = search_semantic_residual(low, low_query)
    with_owner = search_semantic_residual(
        low, low_query, protected_evidence=protected
    )

    assert with_owner.attempted_provider_payload_tokens < low_cap
    assert without_owner.attempted_provider_payload_tokens > low_cap
    assert without_owner.fallback_required is False
    assert with_owner.fallback_required is False
    assert without_owner.evidence
    assert without_owner.classified_frontier.closed is False
    assert without_owner.classified_frontier.unresolved_segment_receipt_sha256s
    assert without_owner.packed_residual_evidence_tokens <= low_cap
    assert with_owner.core_result.retained_leaf_cell_ids == (
        without_owner.core_result.retained_leaf_cell_ids
    )
    assert len(with_owner.protected_duplicates) == 1
    assert with_owner.protected_duplicates[0].protected_binding_receipt_sha256 == (
        protected[0].receipt_sha256
    )
    assert tuple(
        row.segment_receipt_sha256 for row in with_owner.attempted_selection
    ) == with_owner.classified_frontier.retained_segment_receipt_sha256s
    assert tuple(
        row.segment_receipt_sha256 for row in full.attempted_selection
    ) == tuple(
        row.segment_receipt_sha256 for row in high_deduped.attempted_selection
    )
    assert with_owner.classified_frontier.closed is True


def test_compact_provider_plane_not_audit_hashes_controls_payload_cap(
    tmp_path: Path,
) -> None:
    rows: list[tuple[str, str, datetime, str]] = []
    for source_ordinal in range(2):
        source_id = f"p{source_ordinal}::history"
        for turn_ordinal in range(5):
            rows.append(
                (
                    source_id,
                    f"Community memory {source_ordinal}-{turn_ordinal} contains the cobalt clue.",
                    BASE + timedelta(minutes=turn_ordinal),
                    "user",
                )
            )
    cap = 1_600
    index = _build(
        tmp_path,
        "compact",
        rows,
        {"p0::history": [1.0, 0.0], "p1::history": [1.0, 0.0]},
        max_cell_tokens=512,
        payload_token_cap=cap,
    )
    question = (
        "[Question asked at 2026/08/27 12:00] What was the cobalt community clue?"
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))
    audit_tokens = count_tokens(
        json.dumps([row.projection() for row in result.evidence], sort_keys=True)
    )

    assert len(result.evidence) == 10
    assert audit_tokens > result.attempted_provider_payload_tokens
    assert result.attempted_provider_payload_tokens <= cap
    assert result.fallback_required is False
    assert all(row.packing_protection == "must_include" for row in result.evidence)


def test_oversized_row_split_preserves_bytes_without_boundary_whitespace() -> None:
    text = " ".join(f"token{ordinal}" for ordinal in range(180)) + "."
    slices = residual_module._bounded_text_slices(text, 60)
    quotes = tuple(text[start:end] for start, end in slices)

    assert "".join(quotes) == text
    assert all(quote == quote.strip() for quote in quotes)
    assert all(count_tokens(quote) <= 60 for quote in quotes)


def test_greedy_packer_skips_oversized_ranked_row_and_keeps_later_short_row(
    tmp_path: Path,
) -> None:
    long_quote = (
        "The exact heliotrope passphrase is ALPHA-9. "
        + _filler("heliotrope-detail", 260)
    )
    short_quote = "The backup heliotrope passphrase is BETA-4."
    rows = [
        ("p0::long", long_quote, BASE, "user"),
        ("p1::short", short_quote, BASE, "user"),
    ]
    cache = _write_cache(tmp_path / "greedy-skip.db", rows)
    window = build_full_store_window_index(cache)
    index = build_semantic_residual_index(
        window,
        {"p0::long": [1.0, 0.0], "p1::short": [0.8, 0.2]},
        policy=SemanticResidualPolicy(
            max_cell_tokens=2_048,
            payload_token_cap=180,
            dual_gate_enabled=False,
        ),
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What was the exact heliotrope passphrase?"
    )
    result = search_semantic_residual(
        index,
        _compile(index, question, [[1.0, 0.0]]),
    )

    assert result.attempted_selection[0].source_id == "p0::long"
    assert [row.source_id for row in result.local_bindings] == ["p1::short"]
    assert [row.quote for row in result.evidence] == [short_quote]
    assert result.fallback_required is False
    assert result.classified_frontier.closed is False
    assert result.classified_frontier.unresolved_segment_receipt_sha256s == (
        result.attempted_selection[0].segment_receipt_sha256,
    )
    assert result.packed_residual_evidence_tokens <= 180


def test_source_group_allocator_resolves_collisions_deterministically(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        residual_module,
        "semantic_residual_source_identity_receipt",
        lambda _source_id: "0" * 64,
    )

    forward = dict(
        residual_module.semantic_residual_source_group_map(
            ("p0::alpha", "p1::beta", "p2::gamma")
        )
    )
    reverse = dict(
        residual_module.semantic_residual_source_group_map(
            ("p2::gamma", "p1::beta", "p0::alpha")
        )
    )

    assert forward == reverse
    assert forward == {
        "p0::alpha": "G000000",
        "p1::beta": "G000001",
        "p2::gamma": "G000002",
    }
    assert len(set(forward.values())) == len(forward)


def test_zero_packable_novel_population_uses_explicit_fallback(
    tmp_path: Path,
) -> None:
    quote = "The heliotrope vault code is K-19. " + _filler("oversized", 120)
    rows = [("p0::only", quote, BASE, "user")]
    cache = _write_cache(tmp_path / "zero-packable.db", rows)
    window = build_full_store_window_index(cache)
    index = build_semantic_residual_index(
        window,
        {"p0::only": [1.0, 0.0]},
        policy=SemanticResidualPolicy(
            max_cell_tokens=2_048,
            payload_token_cap=10,
            dual_gate_enabled=False,
        ),
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What was the heliotrope vault code?"
    )
    result = search_semantic_residual(
        index,
        _compile(index, question, [[1.0, 0.0]]),
    )

    assert result.attempted_evidence_count == 1
    assert not result.evidence
    assert result.fallback_required is True
    assert result.fallback_reason == "zero_packable_novel_evidence"
    assert result.classified_frontier.closed is False
    assert result.classified_frontier.all_novel_survivors_protected is False
    assert result.classified_frontier.unresolved_segment_receipt_sha256s


def test_split_exact_literals_fail_open_across_separate_leaves(tmp_path: Path) -> None:
    alpha = "Alpha Key identifies the first record."
    beta = "Beta Key identifies the second record."
    index = _build(
        tmp_path,
        "split-exact-fail-open",
        [("alpha", alpha, BASE, "user"), ("beta", beta, BASE, "user")],
        {"alpha": [1.0, 0.0], "beta": [1.0, 0.0]},
    )
    question = (
        '[Question asked at 2026/08/27 12:00] Compare exact phrase "Alpha Key" '
        'with exact phrase "Beta Key".'
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))

    assert result.core_result.pruned_leaf_cell_ids == ()
    assert {alpha, beta} <= {row.quote for row in result.evidence}
    assert sum(row.reason == "exact_literal_absent" for row in result.decision_audits) == 2


def test_required_user_role_does_not_prune_assistant_answer_bridge(
    tmp_path: Path,
) -> None:
    user = "I visited the camera shop on Saturday."
    answer = "You bought the Lumina camera during that visit."
    index = _build(
        tmp_path,
        "role-bridge-fail-open",
        [("user", user, BASE, "user"), ("assistant", answer, BASE, "assistant")],
        {"user": [1.0, 0.0], "assistant": [1.0, 0.0]},
    )
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What did I buy when I visited the camera shop?"
    )
    result = search_semantic_residual(index, _compile(index, question, [[1.0, 0.0]]))

    assert result.query.operator_spec.required_evidence_role == "user"
    assert result.core_result.pruned_leaf_cell_ids == ()
    assert answer in {row.quote for row in result.evidence}
    assert any(row.reason == "required_role_absent" for row in result.decision_audits)
