"""Vendor chat-transcript parsing, byte indexing, and incremental ingest."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from memory_condense.ingest.transcript_source import (
    TranscriptFile,
    detect_layout,
    map_file,
)
from memory_condense.ingest.transcripts import (
    ANTHROPIC_MESSAGES,
    CHATGPT_EXPORT,
    CLAUDE_EXPORT,
    TranscriptMessage,
    normalize_role,
    parse_chatgpt_conversation,
    parse_claude_conversation,
    parse_timestamp,
    parse_transcript_payload,
)


class FakeEmbedder:
    """Bag-of-words hashing embedder so ingest tests need no model download."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self._dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_chunks(self, chunks):
        return [
            chunk.model_copy(update={"embedding": self._vec(chunk.text).tolist()})
            for chunk in chunks
        ]


def claude_conversation(cid: str, messages: list[tuple[str, str]]) -> dict:
    return {
        "uuid": cid,
        "name": f"conversation {cid}",
        "created_at": "2026-01-01T10:00:00Z",
        "chat_messages": [
            {
                "uuid": f"{cid}-{index}",
                "sender": sender,
                "text": text,
                "created_at": "2026-01-01T10:00:00Z",
            }
            for index, (sender, text) in enumerate(messages)
        ],
    }


def chatgpt_conversation(cid: str, messages: list[tuple[str, str]]) -> dict:
    mapping: dict[str, dict] = {}
    previous: str | None = None
    for index, (role, text) in enumerate(messages):
        node_id = f"{cid}-n{index}"
        mapping[node_id] = {
            "id": node_id,
            "parent": previous,
            "children": [],
            "message": {
                "id": f"{cid}-m{index}",
                "author": {"role": role},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": 1767225600.0 + index,
            },
        }
        if previous is not None:
            mapping[previous]["children"].append(node_id)
        previous = node_id
    return {"id": cid, "title": cid, "mapping": mapping}


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestParsing:
    def test_claude_export_roles_and_timestamps(self):
        parsed = parse_claude_conversation(
            claude_conversation("c1", [("human", "Hi"), ("assistant", "Hello")])
        )
        assert [(m.role, m.text) for m in parsed] == [
            ("user", "Hi"),
            ("assistant", "Hello"),
        ]
        assert parsed[0].conversation_id == "c1"
        assert parsed[0].message_id == "c1-0"
        assert parsed[0].created_at is not None
        assert parsed[0].created_at.tzinfo is not None

    def test_claude_structured_content_blocks_drop_non_text(self):
        record = {
            "uuid": "c9",
            "chat_messages": [
                {
                    "uuid": "m1",
                    "sender": "assistant",
                    "content": [
                        {"type": "text", "text": "visible"},
                        {"type": "image", "source": {}},
                        {"type": "tool_use", "name": "search"},
                    ],
                }
            ],
        }
        parsed = parse_claude_conversation(record)
        assert [m.text for m in parsed] == ["visible"]

    def test_chatgpt_picks_longest_branch_of_the_edit_tree(self):
        # A regenerated answer forks the tree; the longer branch wins.
        record = chatgpt_conversation("g1", [("user", "Q"), ("assistant", "A1")])
        record["mapping"]["g1-n0"]["children"].append("fork")
        record["mapping"]["fork"] = {
            "id": "fork",
            "parent": "g1-n0",
            "children": [],
            "message": {
                "id": "fork-m",
                "author": {"role": "assistant"},
                "content": {"parts": ["short"]},
                "create_time": 1767225601.0,
            },
        }
        record["mapping"]["g1-n1"]["children"].append("tail")
        record["mapping"]["tail"] = {
            "id": "tail",
            "parent": "g1-n1",
            "children": [],
            "message": {
                "id": "tail-m",
                "author": {"role": "user"},
                "content": {"parts": ["follow up"]},
                "create_time": 1767225602.0,
            },
        }
        parsed = parse_chatgpt_conversation(record)
        assert [m.text for m in parsed] == ["Q", "A1", "follow up"]

    def test_chatgpt_skips_system_and_empty_parts(self):
        record = chatgpt_conversation("g2", [("user", "Q"), ("assistant", "")])
        parsed = parse_chatgpt_conversation(record)
        assert [m.text for m in parsed] == ["Q"]

    def test_payload_detects_export_arrays(self):
        name, parsed = parse_transcript_payload(
            [claude_conversation("c1", [("human", "a"), ("assistant", "b")])]
        )
        assert name == CLAUDE_EXPORT and len(parsed) == 2

        name, parsed = parse_transcript_payload(
            [chatgpt_conversation("g1", [("user", "a"), ("assistant", "b")])]
        )
        assert name == CHATGPT_EXPORT and len(parsed) == 2

    def test_anthropic_messages_include_system_prompt(self):
        name, parsed = parse_transcript_payload(
            {
                "id": "req",
                "system": "Be terse",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                ],
            }
        )
        assert name == ANTHROPIC_MESSAGES
        assert [(m.role, m.text) for m in parsed] == [
            ("system", "Be terse"),
            ("user", "hi"),
            ("assistant", "yo"),
        ]

    def test_role_aliases_and_unknown_roles(self):
        assert normalize_role("Human") == "user"
        assert normalize_role("model") == "assistant"
        assert normalize_role("tool") is None
        assert normalize_role(None) is None

    def test_timestamp_accepts_epoch_and_iso(self):
        assert parse_timestamp(1767225600.0) is not None
        assert parse_timestamp("2026-01-01T10:00:00Z") is not None
        assert parse_timestamp("not a date") is None
        assert parse_timestamp(True) is None

    def test_message_rejects_blank_text(self):
        with pytest.raises(ValueError, match="non-empty"):
            TranscriptMessage(
                role="user",
                text="   ",
                conversation_id="c",
                message_id="m",
                ordinal=0,
            )


class TestByteIndex:
    def test_array_spans_survive_structural_bytes_inside_strings(self, tmp_path):
        payload = [
            claude_conversation(
                "c1",
                [("human", 'text with {"a":[1,2]} and \\" quote and ] bracket')],
            ),
            claude_conversation("c2", [("human", "second")]),
        ]
        path = write_json(tmp_path / "conversations.json", payload)
        transcript = TranscriptFile(path)
        transcript.refresh()
        assert transcript.index is not None
        assert transcript.index.layout == "array"
        assert len(transcript.index.spans) == 2
        # Every indexed span must decode on its own.
        for span in transcript.index.spans:
            assert isinstance(transcript.decode_span(span), dict)

    def test_layouts_are_detected(self, tmp_path):
        cases = {
            "array.json": ([claude_conversation("c1", [("human", "x")])], "array"),
            "object.json": ({"messages": [{"role": "user", "content": "x"}]}, "object"),
        }
        for name, (payload, expected) in cases.items():
            path = write_json(tmp_path / name, payload)
            with map_file(path) as (data, size):
                assert detect_layout(data, size) == expected

        jsonl = tmp_path / "log.jsonl"
        jsonl.write_text(
            '{"role":"user","content":"a"}\n{"role":"assistant","content":"b"}\n',
            encoding="utf-8",
        )
        with map_file(jsonl) as (data, size):
            assert detect_layout(data, size) == "jsonl"

    def test_jsonl_messages_round_trip(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text(
            '{"role":"user","content":"first"}\n'
            '{"role":"assistant","content":[{"type":"text","text":"second"}]}\n',
            encoding="utf-8",
        )
        transcript = TranscriptFile(path)
        transcript.refresh()
        assert [(m.role, m.text) for m in transcript.iter_messages()] == [
            ("user", "first"),
            ("assistant", "second"),
        ]

    def test_empty_file_indexes_without_error(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        transcript = TranscriptFile(path)
        delta = transcript.refresh()
        assert delta.status == "new"
        assert transcript.index is not None
        assert transcript.index.spans == ()

    def test_span_read_after_truncation_is_rejected(self, tmp_path):
        path = write_json(
            tmp_path / "c.json",
            [claude_conversation("c1", [("human", "hello there")])],
        )
        transcript = TranscriptFile(path)
        transcript.refresh()
        assert transcript.index is not None
        span = transcript.index.spans[0]
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="past the current file size"):
            transcript.read_span(span)


class TestChangeDetection:
    def test_unchanged_file_reports_nothing_pending(self, tmp_path):
        path = write_json(
            tmp_path / "c.json",
            [claude_conversation("c1", [("human", "a"), ("assistant", "b")])],
        )
        transcript = TranscriptFile(path)
        assert transcript.refresh().status == "new"
        second = transcript.refresh()
        assert second.is_unchanged
        assert second.pending == ()

    def test_appended_conversation_is_the_only_pending_work(self, tmp_path):
        path = tmp_path / "c.json"
        payload = [claude_conversation("c1", [("human", "a"), ("assistant", "b")])]
        write_json(path, payload)
        transcript = TranscriptFile(path)
        transcript.refresh()

        payload.append(claude_conversation("c2", [("human", "q"), ("assistant", "r")]))
        write_json(path, payload)
        delta = transcript.refresh()

        assert delta.status == "appended"
        assert [span.index for span in delta.added] == [1]
        assert delta.changed == ()
        texts = [m.text for m in transcript.iter_messages(delta.pending)]
        assert texts == ["q", "r"]

    def test_edited_conversation_is_reported_as_changed(self, tmp_path):
        path = tmp_path / "c.json"
        payload = [claude_conversation("c1", [("human", "a"), ("assistant", "b")])]
        write_json(path, payload)
        transcript = TranscriptFile(path)
        transcript.refresh()

        payload[0]["chat_messages"][1]["text"] = "edited"
        write_json(path, payload)
        delta = transcript.refresh()

        assert delta.status == "rewritten"
        assert [span.index for span in delta.changed] == [0]
        assert delta.added == ()
        assert [m.text for m in transcript.iter_messages(delta.pending)] == [
            "a",
            "edited",
        ]

    def test_removed_conversation_is_reported(self, tmp_path):
        path = tmp_path / "c.json"
        payload = [
            claude_conversation("c1", [("human", "a")]),
            claude_conversation("c2", [("human", "b")]),
        ]
        write_json(path, payload)
        transcript = TranscriptFile(path)
        transcript.refresh()

        write_json(path, payload[:1])
        delta = transcript.refresh()
        assert delta.status == "rewritten"
        assert delta.removed == (1,)


class TestCondenserIngest:
    def test_transcript_ingest_is_incremental(self, tmp_path):
        from memory_condense.application.condenser import MemoryCondenser

        path = tmp_path / "conversations.json"
        payload = [
            claude_conversation(
                "c1",
                [("human", "Where did I leave my keys"), ("assistant", "On the desk")],
            )
        ]
        write_json(path, payload)

        condenser = MemoryCondenser(data_dir=tmp_path / "data", embedder=FakeEmbedder())
        try:
            transcript = TranscriptFile(path)
            first = condenser.ingest_transcript(transcript)
            assert first["status"] == "new"
            assert first["messages_ingested"] == 2
            assert first["layout"] == "array"

            # Nothing changed: no re-ingest, no duplicate turns.
            second = condenser.ingest_transcript(transcript)
            assert second["status"] == "unchanged"
            assert second["messages_ingested"] == 0

            # A live transcript grows; only the new conversation is ingested.
            payload.append(
                claude_conversation("c2", [("human", "And my wallet"), ("assistant", "In the drawer")])
            )
            write_json(path, payload)
            third = condenser.ingest_transcript(transcript)
            assert third["status"] == "appended"
            assert third["messages_ingested"] == 2
            assert third["conversations_ingested"] == 1
        finally:
            condenser.close()

    def test_ingest_binds_conversation_and_message_identity(self, tmp_path):
        from memory_condense.application.condenser import MemoryCondenser

        path = write_json(
            tmp_path / "c.json",
            [claude_conversation("conv-a", [("human", "remember the alarm code 4417")])],
        )
        condenser = MemoryCondenser(data_dir=tmp_path / "data", embedder=FakeEmbedder())
        try:
            transcript = TranscriptFile(path)
            condenser.ingest_transcript(transcript)
            turns = condenser._transcript.get_recent(10)
            assert turns
            assert turns[-1].source_id == "conv-a"
            assert turns[-1].turn_id == "conv-a-0"
        finally:
            condenser.close()
