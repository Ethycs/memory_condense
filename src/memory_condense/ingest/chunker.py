from __future__ import annotations

import re

import pysbd

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk


_CONCEPTUAL_BOUNDARY_RE = re.compile(
    r"\s*(?:;|\u2014|\b(?:by the way|however|anyway|but|although|though),?)\s+",
    re.IGNORECASE,
)


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
        if min_tokens < 1:
            raise ValueError("min_tokens must be positive")
        if max_tokens < min_tokens:
            raise ValueError("max_tokens must be at least min_tokens")
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
        return self._merge_sentences(text, sentences, offsets, turn_id)

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

    def conceptual_spans(self, text: str) -> list[str]:
        """Return event-sized clauses for semantic membership probes.

        Retrieval chunks intentionally carry broad context.  CAV membership
        should not average a short completed event away merely because the
        same sentence begins with a plan, question, correction, or aside.
        """

        spans: list[str] = []
        for sentence in self._split_sentences(text):
            pieces = [
                piece.strip(" ,")
                for piece in _CONCEPTUAL_BOUNDARY_RE.split(sentence)
                if piece.strip(" ,")
            ]
            spans.extend(pieces or [sentence])
        return spans

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
        """Split at character boundaries while enforcing the exact maximum."""
        parts: list[str] = []
        remaining = text.strip()
        while remaining:
            if count_tokens(remaining) <= self.max_tokens:
                parts.append(remaining)
                break

            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if count_tokens(remaining[:middle]) <= self.max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                codepoint_tokens = count_tokens(remaining[0])
                raise ValueError(
                    "max_tokens cannot fit the next Unicode code point "
                    f"({codepoint_tokens} tokens required, {self.max_tokens} allowed)"
                )
            whitespace = remaining.rfind(" ", 0, best + 1)
            cut = whitespace if whitespace > 0 else best
            part = remaining[:cut].rstrip()
            if part:
                parts.append(part)
            remaining = remaining[cut:].lstrip()
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
        source_text: str,
        sentences: list[str],
        offsets: list[tuple[int, int]],
        turn_id: str,
    ) -> list[Chunk]:
        """Greedily merge sentences while preserving their exact source slice."""
        chunks: list[Chunk] = []
        current_sents: list[str] = []
        current_tokens = 0
        current_start = offsets[0][0] if offsets else 0

        for i, (sent, (start, end)) in enumerate(zip(sentences, offsets)):
            prospective_text = source_text[
                current_start if current_sents else start : end
            ]
            prospective_tokens = count_tokens(prospective_text)

            if prospective_tokens > self.max_tokens and current_sents:
                # Emit current chunk
                chunk_text = source_text[current_start : offsets[i - 1][1]]
                exact_tokens = count_tokens(chunk_text)
                chunks.append(
                    Chunk(
                        turn_id=turn_id,
                        text=chunk_text,
                        start_char=current_start,
                        end_char=offsets[i - 1][1],
                        token_count=exact_tokens,
                    )
                )
                current_sents = []
                current_tokens = 0
                current_start = start

                prospective_text = sent
                prospective_tokens = count_tokens(sent)

            current_sents.append(sent)
            current_tokens = prospective_tokens

        # Emit final chunk
        if current_sents:
            last_end = offsets[-1][1]
            chunk_text = source_text[current_start:last_end]
            chunk = Chunk(
                turn_id=turn_id,
                text=chunk_text,
                start_char=current_start,
                end_char=last_end,
                token_count=current_tokens,
            )

            # Merge small final chunk into previous if possible
            merged_text = (
                source_text[chunks[-1].start_char:last_end] if chunks else ""
            )
            merged_tokens = count_tokens(merged_text) if chunks else 0
            if (
                current_tokens < self.min_tokens
                and chunks
                and merged_tokens <= self.max_tokens
            ):
                prev = chunks.pop()
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
