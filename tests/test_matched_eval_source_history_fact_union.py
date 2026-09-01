from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools.matched_eval.query_expansion import FrozenSourceMembership
from tools.matched_eval.source_history_fact_union import (
    EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE,
    FINAL_PROMPT_TOKEN_CAP,
    LANE_ORDER,
    LANE_TOKEN_BUDGETS,
    MAX_PARENT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    DirectEvidenceRef,
    FactLane,
    ParentIdentity,
    SourceHistoryFactUnionError,
    SourceSelection,
    build_post_map_fact_union,
    direct_evidence_projection_sha256,
    hydrate_source_histories,
    hydrate_source_history,
    pack_fact_union_envelope,
    plan_source_history_hydration,
    validate_mapped_facts,
    validate_mapper_completion,
)


def _sha(character: str) -> str:
    return character * 64


_NAMESPACE_ID = _sha("0")


def _write_store(
    path: Path,
    sources: dict[str, list[str]],
) -> dict[str, FrozenSourceMembership]:
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    offset = 0
    for source_index, (source_id, texts) in enumerate(sources.items()):
        for text_index, text in enumerate(texts):
            turn = transcript.append(
                "user" if text_index % 2 == 0 else "assistant",
                text,
                source_id=source_id,
                turn_id=f"turn-{source_index}-{text_index}",
                created_at=datetime(2026, 8, 1, tzinfo=timezone.utc)
                + timedelta(days=offset),
            )
            chunk = Chunk(
                chunk_id=f"chunk-{source_index}-{text_index}",
                turn_id=turn.turn_id,
                text=text,
                start_char=0,
                end_char=len(text),
                token_count=count_tokens(text),
            )
            lexical.add_chunks([chunk])
            offset += 1
    streams = scan_discourse_source_chunks(database)
    memberships = {
        stream.source_id: FrozenSourceMembership(
            source_id=stream.source_id,
            content_chunk_ids=stream.content_chunk_ids,
            metadata_chunk_ids=stream.metadata_chunk_ids,
            stream_sha256=stream.stream_sha256,
        )
        for stream in streams
    }
    database.close()
    return memberships


def _hydrate_all(
    path: Path,
    memberships: dict[str, FrozenSourceMembership],
):
    database = Database(path, read_only=True)
    try:
        histories = hydrate_source_histories(
            database,
            tuple(memberships.values()),
            namespace_id=_NAMESPACE_ID,
            revalidate_store_bytes=lambda: None,
        )
        return {row.source_id: row for row in histories}
    finally:
        database.close()


def _parent(
    direct: tuple[DirectEvidenceRef, ...] = (),
) -> ParentIdentity:
    return ParentIdentity(
        population_identity_sha256=_sha("a"),
        question_order_sha256=_sha("b"),
        snapshot_id=_sha("c"),
        namespace_id=_NAMESPACE_ID,
        parent_packet_id=_sha("d"),
        parent_stage_receipt_sha256=_sha("e"),
        direct_evidence_projection_sha256=direct_evidence_projection_sha256(direct),
    )


def _selection(
    selection_id: str,
    lane: FactLane,
    source_id: str,
    rank: int = 0,
) -> SourceSelection:
    return SourceSelection(
        selection_id=selection_id,
        lane=lane,
        namespace_id=_NAMESPACE_ID,
        source_id=source_id,
        rank=rank,
        selector_receipt_sha256=quote_sha256(
            f"selector:{lane.value}:{rank}:{source_id}"
        ),
    )


def _mapped_item(
    window,
    *,
    mapper_item_id: str,
    fact: str,
    quote: str,
    event_tuple: dict[str, str] | None = None,
    chunk_id: str | None = None,
) -> dict[str, object]:
    chunk = next(
        row
        for row in window.chunks
        if quote in row.text and (chunk_id is None or row.chunk_id == chunk_id)
    )
    start = chunk.text.index(quote)
    return {
        "chunk_id": chunk.chunk_id,
        "event_tuple": event_tuple,
        "fact": fact,
        "mapper_item_id": mapper_item_id,
        "quote": quote,
        "quote_end_char": start + len(quote),
        "quote_sha256": quote_sha256(quote),
        "quote_start_char": start,
        "source_id": window.selection.source_id,
    }


def _event() -> dict[str, str]:
    return {
        "event_time": "2026-08-02",
        "object": "Rome",
        "polarity": "positive",
        "predicate": "visited",
        "status": "current",
        "subject": "Beta",
    }


def test_hydration_rechecks_store_scan_and_preserves_duplicate_source_selections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    memberships = _write_store(
        path,
        {
            "history-a": [
                "[history-a took place at 2026-08-01T00:00:00+00:00]",
                "Alpha kept 3 red bikes.",
                "Alpha later donated one red bike.",
            ]
        },
    )
    histories = _hydrate_all(path, memberships)
    history = histories["history-a"]

    assert tuple(row.turn_ordinal for row in history.chunks) == (1, 2, 3)
    assert sum(row.metadata_chunk for row in history.chunks) == 1
    assert all(row.start_char == 0 and row.end_char == len(row.text) for row in history.chunks)
    assert all(row.turn_text_sha256 == quote_sha256(row.text) for row in history.chunks)

    read_only = Database(path, read_only=True)
    try:
        revalidation_calls: list[int] = []
        repeated = hydrate_source_history(
            read_only,
            memberships["history-a"],
            namespace_id=_NAMESPACE_ID,
            revalidate_store_bytes=lambda: revalidation_calls.append(1),
        )
        tampered = FrozenSourceMembership(
            source_id="history-a",
            content_chunk_ids=memberships["history-a"].content_chunk_ids,
            metadata_chunk_ids=(),
            stream_sha256=_sha("f"),
        )
        with pytest.raises(
            SourceHistoryFactUnionError,
            match="scan_discourse_source_chunks",
        ):
            hydrate_source_history(
                read_only,
                tampered,
                namespace_id=_NAMESPACE_ID,
                revalidate_store_bytes=lambda: None,
            )
    finally:
        read_only.close()
    assert repeated.receipt_sha256 == history.receipt_sha256
    assert revalidation_calls == [1, 1]
    assert repeated.store_bytes_revalidated is True

    selections = (
        _selection("partition-a", FactLane.PARTITION, "history-a"),
        _selection("guided-a", FactLane.GUIDED, "history-a"),
    )
    cap = max(row.token_count for row in history.chunks if not row.metadata_chunk)
    plan = plan_source_history_hydration(
        _parent(),
        selections=selections,
        histories=(history,),
        max_window_tokens=cap,
    )

    assert len(plan.windows) == 4
    assert tuple(row.selection.selection_id for row in plan.windows) == (
        "partition-a",
        "partition-a",
        "guided-a",
        "guided-a",
    )
    assert all(len(row.chunks) == 1 for row in plan.windows)
    assert all(not chunk.metadata_chunk for row in plan.windows for chunk in row.chunks)
    assert all(row.mapping_payload()["frozen_chunk_boundaries"] is True for row in plan.windows)
    assert plan.receipt_sha256 == plan_source_history_hydration(
        _parent(),
        selections=selections,
        histories=(history,),
        max_window_tokens=cap,
    ).receipt_sha256
    with pytest.raises(FrozenInstanceError):
        plan.parent.snapshot_id = _sha("9")  # type: ignore[misc]


def test_batch_hydration_scans_and_reads_namespace_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "memory.db"
    memberships = _write_store(
        path,
        {"history-a": ["Alpha fact."], "history-b": ["Beta fact."]},
    )
    database = Database(path, read_only=True)
    original_execute = database.execute
    queries: list[str] = []
    revalidations: list[int] = []

    def counted_execute(sql: str, params: tuple = ()):
        queries.append(sql)
        return original_execute(sql, params)

    monkeypatch.setattr(database, "execute", counted_execute)
    try:
        histories = hydrate_source_histories(
            database,
            tuple(memberships.values()),
            namespace_id=_NAMESPACE_ID,
            revalidate_store_bytes=lambda: revalidations.append(1),
        )
    finally:
        database.close()

    assert tuple(row.source_id for row in histories) == tuple(memberships)
    assert revalidations == [1, 1]
    assert len(queries) == 2  # one authoritative scan + one ordered text hydration


def test_mapper_validation_is_exact_and_salvages_valid_items_individually(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    memberships = _write_store(
        path,
        {"history-a": ["Alpha kept 3 red bikes for the summer."]},
    )
    history = _hydrate_all(path, memberships)["history-a"]
    plan = plan_source_history_hydration(
        _parent(),
        selections=(
            _selection("partition-a", FactLane.PARTITION, "history-a"),
        ),
        histories=(history,),
    )
    window = plan.windows[0]
    valid = _mapped_item(
        window,
        mapper_item_id="m-good",
        fact="Alpha had 3 red bikes.",
        quote="3 red bikes",
    )
    inexact = dict(valid)
    inexact["mapper_item_id"] = "m-inexact"
    inexact["quote"] = "3  red bikes"
    inexact["quote_sha256"] = quote_sha256("3  red bikes")
    bad_schema = {"mapper_item_id": "m-schema", "fact": "unsupported"}

    batch = validate_mapped_facts(plan, window, (valid, inexact, bad_schema))

    assert len(batch.accepted) == 1
    assert batch.accepted[0].quote == "3 red bikes"
    assert len(batch.rejected) == 2
    assert batch.source_item_count == 3
    assert batch.accepted[0].source_index == 0
    assert any("quote_not_exact" in row.reason for row in batch.rejected)
    assert any("item_schema" in row.reason for row in batch.rejected)

    completion = '{"facts":[' + __import__("json").dumps(valid) + "," + __import__("json").dumps(inexact) + "]}"
    replay = validate_mapper_completion(plan, window, completion)
    assert len(replay.accepted) == 1 and len(replay.rejected) == 1
    assert replay.receipt_sha256 == validate_mapper_completion(
        plan, window, completion
    ).receipt_sha256


def test_post_map_union_dedups_only_after_mapping_merges_provenance_and_then_excludes_direct(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    memberships = _write_store(
        path,
        {
            "history-a": [
                "Alpha kept 3 red bikes.",
                "On Tuesday, Beta visited Rome.",
            ],
            "history-b": ["Beta arrived in Rome on Tuesday."],
        },
    )
    histories = _hydrate_all(path, memberships)
    direct_quote = "Beta visited Rome"
    direct = (
        DirectEvidenceRef(
            evidence_id="S0-1",
            namespace_id=_NAMESPACE_ID,
            source_id="history-a",
            quote_sha256=quote_sha256(direct_quote),
            evidence_receipt_sha256=_sha("1"),
        ),
    )
    selections = (
        _selection("partition-a", FactLane.PARTITION, "history-a"),
        _selection("guided-a", FactLane.GUIDED, "history-a"),
        _selection("em-a", FactLane.EM, "history-a"),
        _selection("direct-b", FactLane.DIRECT, "history-b"),
    )
    plan = plan_source_history_hydration(
        _parent(direct),
        selections=selections,
        histories=(histories["history-a"], histories["history-b"]),
    )
    windows = {row.selection.selection_id: row for row in plan.windows}
    partition_items = (
        _mapped_item(
            windows["partition-a"],
            mapper_item_id="p-coordinate",
            fact="Alpha possessed three red bikes.",
            quote="3 red bikes",
        ),
        _mapped_item(
            windows["partition-a"],
            mapper_item_id="p-direct",
            fact="Beta visited Rome.",
            quote=direct_quote,
        ),
    )
    guided_items = (
        _mapped_item(
            windows["guided-a"],
            mapper_item_id="g-coordinate",
            fact="The red-bike count was 3.",
            quote="3 red bikes",
        ),
    )
    em_items = (
        _mapped_item(
            windows["em-a"],
            mapper_item_id="e-event",
            fact="Beta visited Rome on Tuesday.",
            quote=direct_quote,
            event_tuple=_event(),
        ),
    )
    direct_items = (
        _mapped_item(
            windows["direct-b"],
            mapper_item_id="c-event",
            fact="Tuesday's destination was Rome.",
            quote="Beta arrived in Rome on Tuesday",
            event_tuple=_event(),
        ),
    )
    batches = (
        validate_mapped_facts(plan, windows["partition-a"], partition_items),
        validate_mapped_facts(plan, windows["guided-a"], guided_items),
        validate_mapped_facts(plan, windows["em-a"], em_items),
        validate_mapped_facts(plan, windows["direct-b"], direct_items),
    )

    pending = build_post_map_fact_union(plan, direct_evidence=direct)
    assert pending.completed_window_ids == ()
    assert pending.pending_window_ids == tuple(row.window_id for row in plan.windows)

    union = build_post_map_fact_union(
        plan,
        batches=tuple(reversed(batches)),
        direct_evidence=direct,
    )
    replay = build_post_map_fact_union(
        plan,
        batches=batches,
        direct_evidence=direct,
    )

    assert union.receipt_sha256 == replay.receipt_sha256
    assert union.accepted_before_dedup_count == 5
    assert len(union.union_facts_before_direct_exclusion) == 3
    assert len(union.direct_exclusions) == 1
    assert len(union.retained_facts) == 2
    coordinate = next(
        row
        for row in union.retained_facts
        if row.dedup_projection["kind"] == "source_chunk_quote"
    )
    event = next(
        row
        for row in union.retained_facts
        if row.dedup_projection["kind"] == "full_event_tuple"
    )
    assert coordinate.fact_variants == (
        "Alpha possessed three red bikes.",
        "The red-bike count was 3.",
    )
    assert tuple(row.lane for row in coordinate.origins) == (
        FactLane.PARTITION,
        FactLane.GUIDED,
    )
    assert {row.source_id for row in event.origins} == {"history-a", "history-b"}
    assert union.direct_exclusions[0].matching_direct_evidence_ids == ("S0-1",)
    assert union.direct_exclusions[0].match_modes == ("legacy_exact_quote_hash",)


def test_direct_chunk_strict_substring_is_excluded_only_for_its_exact_origin_chunk(
    tmp_path: Path,
) -> None:
    texts = [
        "First observation says the shared blue token remained locked overnight.",
        "Second observation says the shared blue token was moved at noon.",
    ]
    path = tmp_path / "memory.db"
    memberships = _write_store(path, {"history-a": texts})
    history = _hydrate_all(path, memberships)["history-a"]
    direct = (
        DirectEvidenceRef(
            evidence_id="S0-full-first",
            namespace_id=_NAMESPACE_ID,
            source_id="history-a",
            quote_sha256=quote_sha256(texts[0]),
            evidence_receipt_sha256=_sha("2"),
            text=texts[0],
        ),
    )
    plan = plan_source_history_hydration(
        _parent(direct),
        selections=(_selection("partition-a", FactLane.PARTITION, "history-a"),),
        histories=(history,),
    )
    window = plan.windows[0]
    quote = "shared blue token"
    items = tuple(
        _mapped_item(
            window,
            mapper_item_id=f"m-{index}",
            fact=f"Mapped fact from observation {index + 1}.",
            quote=quote,
            chunk_id=f"chunk-0-{index}",
        )
        for index in range(2)
    )
    batch = validate_mapped_facts(plan, window, items)
    union = build_post_map_fact_union(
        plan,
        batches=(batch,),
        direct_evidence=direct,
    )

    assert union.accepted_before_dedup_count == 2
    assert len(union.union_facts_before_direct_exclusion) == 2
    assert len(union.direct_exclusions) == 1
    assert union.direct_exclusions[0].matching_direct_evidence_ids == (
        "S0-full-first",
    )
    assert union.direct_exclusions[0].match_modes == (
        "same_chunk_strict_substring",
    )
    assert len(union.retained_facts) == 1
    assert union.retained_facts[0].origins[0].chunk_id == "chunk-0-1"
    assert direct[0].text not in json.dumps(direct[0].projection(), sort_keys=True)


def test_compact_fact_aliases_admit_more_than_full_prompt_provenance(
    tmp_path: Path,
) -> None:
    texts = [
        (
            f"Observation {index}: "
            + "long exact provenance wording " * 18
            + f"unique marker {index}."
        )
        for index in range(8)
    ]
    path = tmp_path / "memory.db"
    memberships = _write_store(path, {"history-a": texts})
    history = _hydrate_all(path, memberships)["history-a"]
    plan = plan_source_history_hydration(
        _parent(),
        selections=(_selection("partition-a", FactLane.PARTITION, "history-a"),),
        histories=(history,),
    )
    window = plan.windows[0]
    batch = validate_mapped_facts(
        plan,
        window,
        tuple(
            _mapped_item(
                window,
                mapper_item_id=f"compact-{index}",
                fact=f"Marker {index} was observed.",
                quote=text,
                chunk_id=f"chunk-0-{index}",
            )
            for index, text in enumerate(texts)
        ),
    )
    union = build_post_map_fact_union(plan, batches=(batch,))
    envelope = pack_fact_union_envelope(union, parent_prompt_token_proxy=0)
    partition = next(row for row in envelope.lane_packs if row.lane is FactLane.PARTITION)

    header = "[PARTITION_FACTS]"
    legacy_lines: list[str] = []
    legacy_admitted = 0
    for fact in union.retained_facts:
        alias = f"P{legacy_admitted + 1:03d}"
        legacy = json.dumps(
            {
                "citations": [
                    {
                        "chunk": origin.chunk_id,
                        "quote": origin.quote,
                        "source": origin.source_id,
                    }
                    for origin in fact.origins
                ],
                "facts": list(fact.fact_variants),
                "id": alias,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if count_tokens(header + "\n" + "\n".join((*legacy_lines, legacy))) <= 192:
            legacy_lines.append(legacy)
            legacy_admitted += 1

    assert len(partition.admissions) > legacy_admitted
    assert all('"quote"' not in row.rendered_line for row in partition.admissions)
    assert all('"evidence_id":"P' in row.rendered_line for row in partition.admissions)
    assert partition.tokens_used <= LANE_TOKEN_BUDGETS[FactLane.PARTITION]


def test_non_borrowing_lanes_and_hard_final_envelope(tmp_path: Path) -> None:
    text = " ".join(f"unique{i:02d}" for i in range(20))
    path = tmp_path / "memory.db"
    memberships = _write_store(path, {"history-a": [text]})
    history = _hydrate_all(path, memberships)["history-a"]
    plan = plan_source_history_hydration(
        _parent(),
        selections=(
            _selection("partition-a", FactLane.PARTITION, "history-a"),
        ),
        histories=(history,),
    )
    window = plan.windows[0]
    items = tuple(
        _mapped_item(
            window,
            mapper_item_id=f"m-{index}",
            fact=(f"Fact {index}: " + "long supporting detail " * 30).strip(),
            quote=f"unique{index:02d}",
        )
        for index in range(20)
    )
    batch = validate_mapped_facts(plan, window, items)
    union = build_post_map_fact_union(plan, batches=(batch,))
    envelope = pack_fact_union_envelope(
        union,
        parent_prompt_token_proxy=MAX_PARENT_PROMPT_TOKENS,
    )

    assert {lane.value: LANE_TOKEN_BUDGETS[lane] for lane in LANE_ORDER} == {
        "direct": 384,
        "partition": 192,
        "guided": 192,
        "em": 256,
    }
    direct, partition, guided, em = envelope.lane_packs
    assert partition.tokens_used <= 192
    assert partition.not_admitted_union_fact_ids
    assert direct.tokens_used == guided.tokens_used == em.tokens_used == 0
    assert all(not row.admissions for row in (direct, guided, em))
    assert (
        envelope.external_link_overlay_token_reserve
        == EXTERNAL_LINK_OVERLAY_TOKEN_RESERVE
        == 256
    )
    assert envelope.output_token_reserve == OUTPUT_TOKEN_RESERVE == 768
    assert envelope.final_envelope_token_proxy <= FINAL_PROMPT_TOKEN_CAP == 8_000
    assert envelope.retained_transformer_token_state_bytes == 0
    with pytest.raises(SourceHistoryFactUnionError, match="transformer token state"):
        replace(envelope, retained_transformer_token_state_bytes=False)
    assert envelope.receipt_sha256 == pack_fact_union_envelope(
        union,
        parent_prompt_token_proxy=MAX_PARENT_PROMPT_TOKENS,
    ).receipt_sha256
    with pytest.raises(SourceHistoryFactUnionError, match="reserved"):
        pack_fact_union_envelope(
            union,
            parent_prompt_token_proxy=MAX_PARENT_PROMPT_TOKENS + 1,
        )
