"""Parse conversation corpora into (role, text) turn lists.

Two families of input are supported:

1. Claude conversation exports (``.txt`` / ``.md``) — the project's own
   self-replay eval corpus.
2. Public agent-memory benchmarks shipped as JSON — LongMemEval and LoCoMo.
   These drive the QA-probe eval in :mod:`memory_condense.eval.benchmark`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from memory_condense.domain.schemas import Turn

# .txt format: "User:\n<text>" / "Claude:\n <text>"
_TXT_TURN_RE = re.compile(
    r"^(User|Claude):\s*\n(.*?)(?=^(?:User|Claude):\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

# .md format: "**User:**\n<text>" / "**Assistant:**\n<text>"
# Note: colon is inside the bold markers: **User:** not **User**:
_MD_TURN_RE = re.compile(
    r"^\*\*(User|Assistant):\*\*\s*\n(.*?)(?=^\*\*(?:User|Assistant):\*\*\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

_ROLE_MAP = {
    "User": "user",
    "Claude": "assistant",
    "Assistant": "assistant",
}


def parse_txt(text: str) -> list[tuple[str, str]]:
    """Parse a .txt conversation export.

    Returns a list of (role, text) tuples.
    """
    turns: list[tuple[str, str]] = []
    for match in _TXT_TURN_RE.finditer(text):
        raw_role = match.group(1)
        body = match.group(2).strip()
        if body:
            role = _ROLE_MAP.get(raw_role, raw_role.lower())
            turns.append((role, body))
    return turns


def parse_md(text: str) -> list[tuple[str, str]]:
    """Parse a .md conversation export.

    Returns a list of (role, text) tuples.
    """
    turns: list[tuple[str, str]] = []
    for match in _MD_TURN_RE.finditer(text):
        raw_role = match.group(1)
        body = match.group(2).strip()
        if body:
            role = _ROLE_MAP.get(raw_role, raw_role.lower())
            turns.append((role, body))
    return turns


def load_conversation(path: str | Path) -> list[tuple[str, str]]:
    """Load a conversation from a .txt or .md file.

    Auto-detects format based on file extension.
    Returns a list of (role, text) tuples.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix == ".md":
        return parse_md(text)
    else:
        return parse_txt(text)


def load_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> dict[str, list[tuple[str, str]]]:
    """Load all conversation files from a directory.

    Returns a dict mapping filename -> list of (role, text) tuples.
    Skips files that yield no turns.
    """
    directory = Path(directory)
    conversations: dict[str, list[tuple[str, str]]] = {}

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in extensions:
            turns = load_conversation(path)
            if turns:
                conversations[path.name] = turns

    return conversations


# ---------------------------------------------------------------------------
# Public benchmark loaders (LongMemEval, LoCoMo)
# ---------------------------------------------------------------------------


class BenchmarkQuestion(BaseModel):
    """One QA probe asked against a sample's ingested haystack."""

    question_id: str
    question: str
    answer: str = Field(min_length=1)
    category: str | None = None
    evidence: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    question_date: str | None = None

    @property
    def dated_question(self) -> str:
        if not self.question_date:
            return self.question
        return f"[Question asked at {self.question_date}]\n{self.question}"


class BenchmarkSample(BaseModel):
    """One benchmark record: a haystack of turns plus the questions about it.

    ``turns`` is the haystack flattened into chronological order with roles
    normalized to ``"user"`` / ``"assistant"`` so it can be fed straight to
    :meth:`memory_condense.application.condenser.MemoryCondenser.ingest`.
    """

    sample_id: str
    turns: list[tuple[str, str]] = Field(default_factory=list)
    #: One source/session/document ID per flattened turn. Empty means legacy
    #: input without source boundaries.
    turn_source_ids: list[str | None] = Field(default_factory=list)
    questions: list[BenchmarkQuestion] = Field(default_factory=list)


#: Roles we accept verbatim from a benchmark record.
_BENCH_ROLES = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "system": "assistant",
}

#: Matches LoCoMo session keys: ``session_1``, ``session_12``, ...
#: (deliberately excludes ``session_1_date_time`` and ``session_summary``).
_LOCOMO_SESSION_RE = re.compile(r"^session_(\d+)$")


def _as_text(value: Any) -> str:
    """Coerce a benchmark ``content``/``text`` field to a plain string.

    Some records carry content as a list of parts (``[{"text": ...}, ...]``);
    anything unrecognized becomes an empty string so the turn is dropped.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                inner = part.get("text") or part.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
        return "\n".join(parts).strip()
    return ""


def _as_answer_text(value: Any) -> str:
    """Normalize a benchmark gold answer without discarding numeric labels.

    LongMemEval's current official cleaned file stores 32 counting answers as
    JSON integers. They are answer labels, not message content, so the stricter
    :func:`_as_text` contract is wrong for this field: it used to turn all 32
    into an empty gold string and silently corrupt both metrics and traces.
    """
    text = _as_text(value)
    if text:
        return text
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _as_str_list(value: Any) -> list[str]:
    """Coerce an evidence field (str | list | None) to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _normalize_role(raw: Any, index: int) -> str:
    """Map a benchmark role string to ``"user"`` / ``"assistant"``.

    Unknown or missing roles fall back to alternating by turn index, which
    matches the user-first convention both benchmarks use.
    """
    if isinstance(raw, str):
        mapped = _BENCH_ROLES.get(raw.strip().lower())
        if mapped:
            return mapped
    return "user" if index % 2 == 0 else "assistant"


def parse_longmemeval(data: Any) -> list[BenchmarkSample]:
    """Parse LongMemEval records into :class:`BenchmarkSample` objects.

    Expected record shape (optional keys tolerated)::

        {
          "question_id": "...",
          "question_type": "single-session-user",
          "question": "...",
          "answer": "...",
          "question_date": "2023/05/20 (Sat) 02:29",
          "haystack_dates": [...],
          "haystack_session_ids": [...],
          "haystack_sessions": [[{"role": "user", "content": "..."}, ...], ...],
          "answer_session_ids": [...]
        }

    Each record is one sample carrying exactly one question. Sessions are
    concatenated in the order given (the benchmark ships them chronologically,
    aligned with ``haystack_dates``). Malformed records are skipped.
    """
    samples: list[BenchmarkSample] = []

    for i, record in enumerate(_iter_records(data)):
        if not isinstance(record, dict):
            continue

        question_text = record.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            continue

        sample_id = str(record.get("question_id") or f"longmemeval_{i}")

        turns: list[tuple[str, str]] = []
        turn_source_ids: list[str | None] = []
        sessions = record.get("haystack_sessions") or []
        if not isinstance(sessions, list):
            sessions = []

        session_ids = _as_str_list(record.get("haystack_session_ids"))
        session_dates = _as_str_list(record.get("haystack_dates"))
        for session_index, session in enumerate(sessions):
            if not isinstance(session, list):
                continue
            source_id = (
                session_ids[session_index]
                if session_index < len(session_ids)
                else f"session_{session_index + 1}"
            )
            if session_index < len(session_dates) and session_dates[session_index]:
                turns.append(
                    (
                        "system",
                        f"[{source_id} took place at {session_dates[session_index]}]",
                    )
                )
                turn_source_ids.append(source_id)
            for j, turn in enumerate(session):
                if not isinstance(turn, dict):
                    continue
                text = _as_text(turn.get("content", turn.get("text")))
                if not text:
                    continue
                turns.append((_normalize_role(turn.get("role"), j), text))
                turn_source_ids.append(source_id)

        answer = _as_answer_text(record.get("answer"))
        if not answer:
            continue
        question = BenchmarkQuestion(
            question_id=sample_id,
            question=question_text.strip(),
            answer=answer,
            category=(
                str(record["question_type"])
                if record.get("question_type") is not None
                else None
            ),
            evidence=_as_str_list(record.get("answer_session_ids")),
            evidence_sources=_as_str_list(record.get("answer_session_ids")),
            question_date=_as_text(record.get("question_date")) or None,
        )

        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                turns=turns,
                turn_source_ids=turn_source_ids,
                questions=[question],
            )
        )

    return samples


def parse_locomo(data: Any) -> list[BenchmarkSample]:
    """Parse LoCoMo records into :class:`BenchmarkSample` objects.

    Expected record shape (optional keys tolerated)::

        {
          "sample_id": "conv-26",
          "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "Caroline", "dia_id": "D1:1",
                           "text": "..."}, ...],
            "session_2": [...]
          },
          "qa": [{"question": "...", "answer": "...",
                  "category": 1, "evidence": ["D1:1"]}]
        }

    Speaker normalization: LoCoMo dialogues are between two *named* humans,
    so there is no intrinsic user/assistant split. We map the **first speaker
    seen in the earliest session to "user"** and every other speaker to
    "assistant". This keeps turns alternating the way the ingest path expects
    and is stable across sessions within a sample.

    Sessions are ordered by their numeric suffix (so ``session_10`` follows
    ``session_9``, not ``session_1``). Malformed records are skipped.
    """
    samples: list[BenchmarkSample] = []

    for i, record in enumerate(_iter_records(data)):
        if not isinstance(record, dict):
            continue

        conversation = record.get("conversation")
        if not isinstance(conversation, dict):
            continue

        sample_id = str(
            record.get("sample_id") or record.get("id") or f"locomo_{i}"
        )

        # Order sessions numerically.
        session_keys: list[tuple[int, str]] = []
        for key in conversation:
            match = _LOCOMO_SESSION_RE.match(str(key))
            if match and isinstance(conversation[key], list):
                session_keys.append((int(match.group(1)), key))
        session_keys.sort()

        turns: list[tuple[str, str]] = []
        turn_source_ids: list[str | None] = []
        first_speaker: str | None = None

        for _, key in session_keys:
            # Ingest the session's timestamp as its own turn.
            #
            # LoCoMo's largest question category is temporal ("When did X…?"),
            # and the gold answers are these dates — which live in
            # `session_N_date_time`, *not* in any turn's text. Dropping them
            # made an entire category unanswerable from the ingested data:
            # measured on conv-26, only 2.7% of category-2 answers appeared
            # anywhere in the haystack, against 43-45% for the non-temporal
            # categories. A benchmark run on that corpus would have scored the
            # retrieval system for information it was never given.
            date_time = conversation.get(f"{key}_date_time")
            if isinstance(date_time, str) and date_time.strip():
                turns.append(("system", f"[{key} took place at {date_time.strip()}]"))
                turn_source_ids.append(key)

            for j, turn in enumerate(conversation[key]):
                if not isinstance(turn, dict):
                    continue
                text = _as_text(turn.get("text", turn.get("content")))
                if not text:
                    continue

                speaker = turn.get("speaker")
                if isinstance(speaker, str) and speaker.strip():
                    speaker = speaker.strip()
                    if first_speaker is None:
                        first_speaker = speaker
                    role = "user" if speaker == first_speaker else "assistant"
                else:
                    role = _normalize_role(None, j)

                turns.append((role, text))
                turn_source_ids.append(key)

        questions: list[BenchmarkQuestion] = []
        qa_list = record.get("qa") or []
        if not isinstance(qa_list, list):
            qa_list = []

        for q_index, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                continue
            q_text = qa.get("question")
            if not isinstance(q_text, str) or not q_text.strip():
                continue
            # Adversarial LoCoMo items use "adversarial_answer" instead.
            answer = _as_answer_text(
                qa.get("answer", qa.get("adversarial_answer"))
            )
            if not answer:
                continue
            questions.append(
                BenchmarkQuestion(
                    question_id=f"{sample_id}_q{q_index}",
                    question=q_text.strip(),
                    answer=answer,
                    category=(
                        str(qa["category"]) if qa.get("category") is not None else None
                    ),
                    evidence=_as_str_list(qa.get("evidence")),
                    evidence_sources=sorted(
                        {
                            f"session_{match.group(1)}"
                            for value in _as_str_list(qa.get("evidence"))
                            if (match := re.match(r"^D(\d+):", value))
                        }
                    ),
                )
            )

        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                turns=turns,
                turn_source_ids=turn_source_ids,
                questions=questions,
            )
        )

    return samples


def _iter_records(data: Any) -> list[Any]:
    """Normalize a parsed JSON payload to a list of records.

    Accepts a top-level list, a single record dict, or a wrapper dict with the
    records under a common key (``data`` / ``samples`` / ``records`` /
    ``questions`` / ``instances``).
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "records", "questions", "instances"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return []


def detect_benchmark_format(data: Any) -> str:
    """Sniff a parsed JSON payload and return ``"longmemeval"``/``"locomo"``.

    Raises ``ValueError`` if neither signature is present.
    """
    for record in _iter_records(data):
        if not isinstance(record, dict):
            continue
        if "haystack_sessions" in record:
            return "longmemeval"
        if "conversation" in record and "qa" in record:
            return "locomo"
        # Weaker fallbacks, checked only after the strong signatures.
        if "haystack_dates" in record or "answer_session_ids" in record:
            return "longmemeval"
        if isinstance(record.get("conversation"), dict) and any(
            _LOCOMO_SESSION_RE.match(str(k)) for k in record["conversation"]
        ):
            return "locomo"
    raise ValueError(
        "Could not auto-detect benchmark format: expected LongMemEval "
        "('haystack_sessions') or LoCoMo ('conversation' + 'qa') keys."
    )


def _read_json_payload(path: Path) -> Any:
    """Read a ``.json`` document or a ``.jsonl`` file (one record per line)."""
    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix.lower() in (".jsonl", ".ndjson"):
        records: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines rather than failing the file
        return records

    return json.loads(text)


def load_benchmark(
    path: str | Path,
    format: str = "auto",
) -> list[BenchmarkSample]:
    """Load a benchmark file into :class:`BenchmarkSample` objects.

    Args:
        path: A ``.json`` (single document) or ``.jsonl`` (one record per line)
            file.
        format: ``"auto"`` (sniff the keys), ``"longmemeval"`` or ``"locomo"``.

    Raises:
        ValueError: if ``format`` is unknown, or if ``"auto"`` cannot identify
            the payload.
    """
    path = Path(path)
    data = _read_json_payload(path)

    fmt = (format or "auto").strip().lower()
    if fmt == "auto":
        fmt = detect_benchmark_format(data)

    if fmt == "longmemeval":
        return parse_longmemeval(data)
    if fmt == "locomo":
        return parse_locomo(data)

    raise ValueError(
        f"Unknown benchmark format {format!r}; "
        'expected one of "auto", "longmemeval", "locomo".'
    )
