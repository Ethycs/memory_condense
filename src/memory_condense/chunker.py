from __future__ import annotations

import re

import pysbd

from memory_condense._tokenizer import count_tokens
from memory_condense.schemas import Chunk


class Chunker:
    """Splits turn text into chunks using sentence boundary detection + merge.

    Sentences are detected with pySBD, then greedily merged into chunks
    targeting the [min_tokens, max_tokens] range.
    """

    def __init__(
        self,
        min_tokens: int = 120,
        max_tokens: int = 250,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._segmenter = pysbd.Segmenter(language="en", clean=False)

    def chunk_turn(self, turn_id: str, text: str) -> list[Chunk]:
        """Split a single turn's text into Chunk objects."""
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        offsets = self._compute_offsets(text, sentences)
        return self._merge_sentences(sentences, offsets, turn_id)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using pySBD."""
        segments = self._segmenter.segment(text)
        result = []
        for seg in segments:
            seg = seg.strip()
            if seg:
                # Sub-split oversized sentences at clause boundaries
                if count_tokens(seg) > self.max_tokens:
                    result.extend(self._subsplit(seg))
                else:
                    result.append(seg)
        return result

    def _subsplit(self, text: str) -> list[str]:
        """Split an oversized sentence at clause boundaries."""
        # Try splitting at semicolons, then commas
        for delimiter in ["; ", ", "]:
            parts = text.split(delimiter)
            if len(parts) > 1:
                # Re-attach delimiters to each part (except last)
                restored = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    if i < len(parts) - 1:
                        restored.append(part + delimiter.rstrip())
                    else:
                        restored.append(part)
                # Check if all parts are within budget
                if all(count_tokens(p) <= self.max_tokens for p in restored):
                    return restored

        # Last resort: hard split by token count
        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        """Split text into roughly max_tokens-sized pieces by words."""
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = count_tokens(word)
            if current_tokens + word_tokens > self.max_tokens and current:
                parts.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(word)
            current_tokens += word_tokens

        if current:
            parts.append(" ".join(current))
        return parts

    def _compute_offsets(
        self, text: str, sentences: list[str]
    ) -> list[tuple[int, int]]:
        """Find (start_char, end_char) for each sentence in the original text."""
        offsets: list[tuple[int, int]] = []
        search_start = 0
        for sent in sentences:
            # Find the sentence in the original text, accounting for
            # whitespace differences from pySBD stripping
            # Use a simple word-based search to locate the span
            idx = text.find(sent, search_start)
            if idx == -1:
                # Fallback: find first word match
                first_word = sent.split()[0] if sent.split() else ""
                idx = text.find(first_word, search_start)
                if idx == -1:
                    idx = search_start
                end_idx = idx + len(sent)
            else:
                end_idx = idx + len(sent)
            offsets.append((idx, end_idx))
            search_start = end_idx
        return offsets

    def _merge_sentences(
        self,
        sentences: list[str],
        offsets: list[tuple[int, int]],
        turn_id: str,
    ) -> list[Chunk]:
        """Greedily merge consecutive sentences into chunks."""
        chunks: list[Chunk] = []
        current_sents: list[str] = []
        current_tokens = 0
        current_start = offsets[0][0] if offsets else 0

        for i, (sent, (start, end)) in enumerate(zip(sentences, offsets)):
            sent_tokens = count_tokens(sent)

            if current_tokens + sent_tokens > self.max_tokens and current_sents:
                # Emit current chunk
                chunk_text = " ".join(current_sents)
                chunks.append(
                    Chunk(
                        turn_id=turn_id,
                        text=chunk_text,
                        start_char=current_start,
                        end_char=offsets[i - 1][1],
                        token_count=current_tokens,
                    )
                )
                current_sents = []
                current_tokens = 0
                current_start = start

            current_sents.append(sent)
            current_tokens += sent_tokens

        # Emit final chunk
        if current_sents:
            chunk_text = " ".join(current_sents)
            last_end = offsets[-1][1]
            chunk = Chunk(
                turn_id=turn_id,
                text=chunk_text,
                start_char=current_start,
                end_char=last_end,
                token_count=current_tokens,
            )

            # Merge small final chunk into previous if possible
            if (
                current_tokens < self.min_tokens
                and chunks
                and chunks[-1].token_count + current_tokens <= self.max_tokens
            ):
                prev = chunks.pop()
                merged_text = prev.text + " " + chunk_text
                merged_tokens = prev.token_count + current_tokens
                chunks.append(
                    Chunk(
                        turn_id=turn_id,
                        text=merged_text,
                        start_char=prev.start_char,
                        end_char=last_end,
                        token_count=merged_tokens,
                    )
                )
            else:
                chunks.append(chunk)

        return chunks
