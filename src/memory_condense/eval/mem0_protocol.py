"""Mem0 response normalization and LongMemEval corpus preparation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_condense.eval.mem0_models import (
    Mem0ProtocolError,
    SourceRef,
    _PreparedBatch,
    _PreparedCorpus,
)
from memory_condense.ingest.loader import (
    BenchmarkSample,
    _parse_longmemeval_date,
)


_SESSION_DATE_RE = re.compile(
    r"^\[(?P<session>.+?) took place at (?P<date>.+?)\]$",
    re.IGNORECASE,
)

def _response_rows(response: Any, *, operation: str) -> list[Mapping[str, Any]]:
    """Strictly normalize documented list/``{"results": [...]}`` variants."""

    rows: Any
    if isinstance(response, Mapping) and "results" in response:
        rows = response["results"]
    elif isinstance(response, Sequence) and not isinstance(
        response, (str, bytes, bytearray)
    ):
        rows = response
    elif isinstance(response, Mapping) and (
        "id" in response or "memory_id" in response
    ):
        rows = [response]
    else:
        raise Mem0ProtocolError(
            f"Mem0 {operation} returned an unsupported response shape."
        )
    if not isinstance(rows, Sequence) or isinstance(
        rows, (str, bytes, bytearray)
    ):
        raise Mem0ProtocolError(
            f"Mem0 {operation} response 'results' must be a sequence."
        )
    normalized: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Mem0ProtocolError(
                f"Mem0 {operation} result {index} is not a mapping."
            )
        normalized.append(row)
    return normalized


def _memory_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("id", row.get("memory_id"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _memory_text(row: Mapping[str, Any]) -> str:
    value = row.get("memory", row.get("text", ""))
    return value.strip() if isinstance(value, str) else ""


def _memory_created_at(row: Mapping[str, Any]) -> str | None:
    value = row.get("created_at")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _official_date_label(value: str) -> str:
    """Render Mem0's returned timestamp as the official benchmark date label."""

    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%A, %B %d, %Y")
    except ValueError:
        # The official runner tolerates provider timestamp variants by using
        # their date prefix.  Refuse values that cannot supply even that.
        prefix = candidate[:10]
        try:
            parsed = datetime.strptime(prefix, "%Y-%m-%d")
        except ValueError as exc:
            raise Mem0ProtocolError(
                f"Mem0 search returned an invalid created_at value: {value!r}."
            ) from exc
        return parsed.strftime("%A, %B %d, %Y")


def _memory_score(row: Mapping[str, Any]) -> float | None:
    value = row.get("score")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _safe_label_value(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _parse_session_date(value: str) -> datetime:
    """Parse a session date via the canonical loader parser (UTC-aware).

    Certified chronology cannot tolerate an unparseable date, so the loader's
    ``None`` failure signal is promoted to :class:`Mem0ProtocolError` here.
    """
    parsed = _parse_longmemeval_date(value)
    if parsed is None:
        raise Mem0ProtocolError(
            f"Unsupported haystack session date {value!r}; chronology cannot be certified."
        )
    return parsed


@dataclass(frozen=True, slots=True)
class _SessionBlock:
    source: str
    date: str
    turns: tuple[tuple[str, str], ...]
    original_index: int


def _session_blocks(sample: BenchmarkSample) -> tuple[_SessionBlock, ...]:
    source_ids = sample.turn_source_ids
    if source_ids and len(source_ids) != len(sample.turns):
        raise Mem0ProtocolError(
            "turn_source_ids must be empty or parallel to every benchmark turn."
        )

    blocks: list[_SessionBlock] = []
    current_source = ""
    current_date = ""
    current_turns: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal current_turns
        if not current_turns:
            return
        blocks.append(
            _SessionBlock(
                source=current_source or f"{sample.sample_id}:session_1",
                date=current_date,
                turns=tuple(current_turns),
                original_index=len(blocks) + 1,
            )
        )
        current_turns = []

    for index, (raw_role, text) in enumerate(sample.turns):
        declared = ""
        if source_ids and source_ids[index] is not None:
            declared = str(source_ids[index]).strip()
        match = _SESSION_DATE_RE.fullmatch(text.strip())
        if match:
            marker_source = match.group("session").strip()
            if declared and declared != marker_source:
                raise Mem0ProtocolError(
                    "Session marker and turn_source_ids disagree at turn "
                    f"{index}."
                )
            if current_turns:
                flush()
            current_source = marker_source
            current_date = match.group("date").strip()
            continue

        source = declared or current_source or f"{sample.sample_id}:session_1"
        if current_source and source != current_source:
            flush()
            current_date = ""
        current_source = source
        role = str(raw_role).strip().lower()
        current_turns.append((role, text))

    flush()

    def sort_key(block: _SessionBlock) -> tuple[int, datetime, int]:
        if not block.date:
            return (
                1,
                datetime.max.replace(tzinfo=timezone.utc),
                block.original_index,
            )
        return (0, _parse_session_date(block.date), block.original_index)

    return tuple(sorted(blocks, key=sort_key))


def _prepared_batches(sample: BenchmarkSample) -> tuple[_PreparedBatch, ...]:
    batches: list[_PreparedBatch] = []
    for chronological_index, block in enumerate(_session_blocks(sample), start=1):
        for turn_start in range(0, len(block.turns), 2):
            messages = block.turns[turn_start : turn_start + 2]
            ref = SourceRef(
                sample_id=sample.sample_id,
                source=block.source,
                session=block.source,
                session_index=chronological_index,
                original_session_index=block.original_index,
                batch_index=(turn_start // 2) + 1,
                date=block.date,
                turn_start=turn_start,
                turn_count=len(messages),
                roles=tuple(role for role, _ in messages),
            )
            batches.append(_PreparedBatch(ref=ref, messages=messages))
    return tuple(batches)


def _prepared_sample(sample: BenchmarkSample) -> _PreparedCorpus:
    batches = _prepared_batches(sample)
    return _PreparedCorpus(
        sample_id=sample.sample_id,
        batches=batches,
        raw_pair_count=len(batches),
        skipped_empty_pair_count=0,
        official_longmemeval_protocol=False,
    )


def _prepared_longmemeval_record(record: Mapping[str, Any]) -> _PreparedCorpus:
    """Prepare the official lossless LongMemEval add sequence.

    The shared :class:`BenchmarkSample` loader intentionally drops empty text,
    so it cannot reproduce the official order for the handful of sessions that
    contain empty turns.  Certified Mem0 runs therefore consume the raw record:
    pair original consecutive turns first, then skip a whole pair if either
    message is empty.
    """

    sample_id_value = record.get("question_id")
    sample_id = str(sample_id_value).strip() if sample_id_value is not None else ""
    if not sample_id:
        raise Mem0ProtocolError(
            "Certified LongMemEval input requires a non-empty question_id."
        )
    sessions = record.get("haystack_sessions")
    session_ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    if not all(isinstance(value, list) for value in (sessions, session_ids, dates)):
        raise Mem0ProtocolError(
            "Certified LongMemEval input requires list-valued sessions, IDs, and dates."
        )
    assert isinstance(sessions, list)
    assert isinstance(session_ids, list)
    assert isinstance(dates, list)
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise Mem0ProtocolError(
            "LongMemEval sessions, session IDs, and dates must be parallel."
        )

    ordered: list[tuple[datetime, int, str, str, list[Any]]] = []
    for original_index, (source_value, date_value, session) in enumerate(
        zip(session_ids, dates, sessions),
        start=1,
    ):
        if not isinstance(source_value, str) or not isinstance(date_value, str):
            raise Mem0ProtocolError(
                "Every certified LongMemEval session ID and date must be a string."
            )
        source = source_value.strip()
        date = date_value.strip()
        if not source or not date or not isinstance(session, list):
            raise Mem0ProtocolError(
                "Every certified LongMemEval session needs an ID, date, and turn list."
            )
        parsed_date = _parse_session_date(date)
        ordered.append((parsed_date, original_index, source, date, session))
    ordered.sort(key=lambda item: (item[0], item[1]))

    batches: list[_PreparedBatch] = []
    raw_pair_count = 0
    skipped_empty_pair_count = 0
    for chronological_index, (
        _parsed_date,
        original_index,
        source,
        date,
        session,
    ) in enumerate(ordered, start=1):
        for turn_start in range(0, len(session), 2):
            raw_pair_count += 1
            raw_pair = session[turn_start : turn_start + 2]
            messages: list[tuple[str, str]] = []
            for turn in raw_pair:
                if not isinstance(turn, Mapping):
                    raise Mem0ProtocolError(
                        "Every certified LongMemEval turn must be a mapping."
                    )
                role_value = turn.get("role")
                content_value = turn.get("content")
                if not isinstance(role_value, str) or not isinstance(
                    content_value, str
                ):
                    raise Mem0ProtocolError(
                        "Every certified LongMemEval turn needs string role/content."
                    )
                role = role_value.strip().lower()
                if role not in {"user", "assistant"}:
                    raise Mem0ProtocolError(
                        f"Unsupported certified LongMemEval role: {role_value!r}."
                    )
                messages.append((role, content_value))
            # Match the official runner's truthiness check exactly: an empty
            # string suppresses the entire original pair; whitespace is still
            # a non-empty message and must not shift later pair boundaries.
            if any(not content for _role, content in messages):
                skipped_empty_pair_count += 1
                continue
            ref = SourceRef(
                sample_id=sample_id,
                source=source,
                session=source,
                session_index=chronological_index,
                original_session_index=original_index,
                batch_index=(turn_start // 2) + 1,
                date=date,
                turn_start=turn_start,
                turn_count=len(messages),
                roles=tuple(role for role, _content in messages),
            )
            batches.append(_PreparedBatch(ref=ref, messages=tuple(messages)))

    return _PreparedCorpus(
        sample_id=sample_id,
        batches=tuple(batches),
        raw_pair_count=raw_pair_count,
        skipped_empty_pair_count=skipped_empty_pair_count,
        official_longmemeval_protocol=True,
    )


def _merge_refs(
    existing: Sequence[SourceRef], incoming: Sequence[SourceRef]
) -> list[SourceRef]:
    merged = list(existing)
    for ref in incoming:
        if ref not in merged:
            merged.append(ref)
    return merged


def _validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("threshold must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1].")
    return result
