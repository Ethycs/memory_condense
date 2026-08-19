"""Frozen prompt reconstruction, token accounting, and sentence packing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from dataclasses import dataclass
from typing import Iterable

import pysbd
import tiktoken

from .canonical import AuditError, canonical_sha256
from .frozen_source import FrozenSource


@dataclass(frozen=True, slots=True)
class TextVariant:
    text: str
    # prepared-text [start, end) -> chunk-text [start, end)
    mappings: tuple[tuple[int, int, int, int], ...]
    kind: str


class FrozenPromptRuntime:
    """Runtime whose identities must exactly match the frozen policy."""

    def __init__(self, source: FrozenSource, expected_identity: dict[str, object]):
        self.source = source
        try:
            self.encoder = tiktoken.get_encoding(source.prompt_encoding)
        except Exception as exc:
            raise AuditError(f"cannot load frozen tokenizer encoding: {exc}") from exc
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.lexical_re = re.compile(source.lexical_token_pattern, re.UNICODE)
        self._verify_locked_package("tiktoken")
        self._verify_locked_package("pysbd")
        actual_identity = self.tokenizer_identity()
        if canonical_sha256(actual_identity) != canonical_sha256(expected_identity):
            raise AuditError(
                "runtime tokenizer identity does not match the frozen policy"
            )

    def _verify_locked_package(self, package: str) -> None:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AuditError(f"required frozen package is unavailable: {package}") from exc
        text = self.source.environment_lock.decode("utf-8", errors="strict")
        matches = set(
            re.findall(
                rf"/{re.escape(package)}-([0-9][A-Za-z0-9.+_]*)-",
                text,
                flags=re.IGNORECASE,
            )
        )
        if not matches:
            raise AuditError(f"pixi.lock does not identify package {package}")
        if actual not in matches:
            raise AuditError(
                f"runtime {package} {actual} is not frozen by pixi.lock ({sorted(matches)})"
            )

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text, disallowed_special=()))

    def truncate(self, text: str, maximum: int) -> str:
        if maximum <= 0:
            return ""
        tokens = self.encoder.encode(text, disallowed_special=())
        return text if len(tokens) <= maximum else self.encoder.decode(tokens[:maximum])

    def tokenizer_identity(self) -> dict[str, str | int]:
        mergeable = getattr(self.encoder, "_mergeable_ranks", None)
        special = getattr(self.encoder, "_special_tokens", None)
        if not isinstance(mergeable, dict) or not isinstance(special, dict):
            raise AuditError("tiktoken does not expose the frozen vocabulary tables")
        digest = hashlib.sha256()
        digest.update(self.source.prompt_proxy_schema.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.source.prompt_encoding.encode("utf-8"))
        digest.update(b"\0")
        for token, rank in sorted(mergeable.items(), key=lambda item: item[1]):
            digest.update(int(rank).to_bytes(8, "big", signed=False))
            digest.update(len(token).to_bytes(8, "big", signed=False))
            digest.update(token)
        for token, rank in sorted(special.items()):
            encoded = token.encode("utf-8")
            digest.update(int(rank).to_bytes(8, "big", signed=False))
            digest.update(len(encoded).to_bytes(8, "big", signed=False))
            digest.update(encoded)
        return {
            "schema": self.source.prompt_proxy_schema,
            "implementation": "tiktoken",
            "implementation_version": importlib.metadata.version("tiktoken"),
            "encoding": self.source.prompt_encoding,
            "vocabulary_sha256": digest.hexdigest(),
            "chat_framing_tokens_per_message": self.source.framing_per_message,
            "chat_framing_tokens_fixed": self.source.framing_fixed,
        }

    def lexical_tokens(self, text: str) -> list[str]:
        return [
            token
            for token in self.lexical_re.findall(text.lower())
            if len(token) >= self.source.lexical_min_token_len
            and token not in self.source.lexical_stopwords
        ]

    def prompt_messages(
        self, dated_question: str, retrieved_chunks: list[str]
    ) -> tuple[str, list[dict[str, str]]]:
        context = (
            "\n".join(
                f"[{index}] {excerpt}"
                for index, excerpt in enumerate(retrieved_chunks, start=1)
            )
            if retrieved_chunks
            else self.source.qa_no_context
        )
        messages = [
            {"role": "system", "content": self.source.qa_system_prompt},
            {
                "role": "user",
                "content": self.source.qa_user_template.format(
                    context=context,
                    question=dated_question,
                ),
            },
        ]
        return context, messages

    def prompt_token_proxy(self, messages: list[dict[str, str]]) -> int:
        return (
            sum(self.count_tokens(message.get("content", "")) for message in messages)
            + self.source.framing_per_message * len(messages)
            + self.source.framing_fixed
        )

    def judge_messages(
        self,
        question: str,
        gold: str,
        prediction: str,
    ) -> list[dict[str, str]]:
        """Reconstruct the exact frozen semantic-judge messages."""

        return [
            {"role": "system", "content": self.source.judge_system_prompt},
            {
                "role": "user",
                "content": self.source.judge_user_template.format(
                    question=question,
                    gold=gold,
                    prediction=prediction,
                ),
            },
        ]

    def text_variants(
        self,
        chunk_text: str,
        query: str,
        *,
        query_aware: bool,
        max_sentences: int,
    ) -> tuple[TextVariant, ...]:
        """Return raw and exact frozen query-aware bodies with provenance maps."""

        leading = len(chunk_text) - len(chunk_text.lstrip())
        stripped = chunk_text.strip()
        raw = TextVariant(
            text=stripped,
            mappings=((0, len(stripped), leading, leading + len(stripped)),),
            kind="raw_chunk",
        )
        if not query_aware or not query.strip() or not stripped:
            return (raw,)
        segments = [
            segment.strip()
            for segment in self.segmenter.segment(stripped)
            if segment.strip()
        ]
        if len(segments) <= max_sentences:
            return (raw,)
        query_terms = set(self.lexical_tokens(query))
        if not query_terms:
            return (raw,)
        scored: list[tuple[float, int]] = []
        for index, sentence in enumerate(segments):
            sentence_terms = set(self.lexical_tokens(sentence))
            overlap = query_terms.intersection(sentence_terms)
            if not overlap:
                continue
            overlap_weight = sum(
                3.0 if term.isdigit() or len(term) >= 8 else 1.0
                for term in overlap
            )
            scored.append(
                (overlap_weight / math.sqrt(max(1, len(sentence_terms))), index)
            )
        if not scored:
            return (raw,)
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(index for _score, index in scored[:max_sentences])

        locations: list[tuple[int, int]] = []
        cursor = 0
        for sentence in segments:
            start = stripped.find(sentence, cursor)
            if start < 0:
                raise AuditError("pysbd sentence cannot be mapped back to its chunk")
            locations.append((start + leading, start + leading + len(sentence)))
            cursor = start + len(sentence)
        parts = [segments[index] for index in selected]
        prepared = " ".join(parts)
        mappings: list[tuple[int, int, int, int]] = []
        output_cursor = 0
        for position, index in enumerate(selected):
            if position:
                output_cursor += 1  # the frozen join inserts one synthetic space
            start, end = locations[index]
            mappings.append((output_cursor, output_cursor + len(segments[index]), start, end))
            output_cursor += len(segments[index])
        focused = TextVariant(
            text=prepared,
            mappings=tuple(mappings),
            kind="query_aware_sentences",
        )
        return (raw, focused) if focused.text != raw.text else (raw,)

    def matching_prefixes(
        self,
        body: str,
        variants: Iterable[TextVariant],
        *,
        max_tokens: int,
    ) -> tuple[tuple[TextVariant, int, int], ...]:
        """Find exact token-boundary prefixes, returning text and token lengths."""

        matches: list[tuple[TextVariant, int, int]] = []
        for variant in variants:
            tokens = self.encoder.encode(variant.text, disallowed_special=())
            for count in range(1, min(max_tokens, len(tokens)) + 1):
                candidate = (
                    variant.text
                    if count == len(tokens)
                    else self.encoder.decode(tokens[:count])
                )
                if candidate == body:
                    matches.append((variant, len(candidate), count))
        return tuple(matches)
