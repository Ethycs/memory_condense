"""Source-aware loading for mixed local conversation and document corpora.

The eval loader intentionally handles a narrow directory of exported chats.
Real note repositories are messier: nested directories, authored documents,
ChatGPT mapping JSON, notebooks, exact duplicates, and repeated dated exports.
This module inventories that material without copying it or loading repository
internals such as ``.git`` into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from memory_condense.ingest.loader import parse_md, parse_txt


_SUPPORTED_SUFFIXES = {".txt", ".md", ".py", ".json", ".ipynb", ".docx"}
_SKIPPED_DIRS = {".git", ".hg", ".svn", "__pycache__"}
_CHATGPT_MARKER_RE = re.compile(
    r"^(You said|ChatGPT said|Claude said):\s*\n"
    r"(.*?)(?=^(?:You said|ChatGPT said|Claude said):\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_EXPORT_ID_RE = re.compile(r"(?:^|_)([0-9a-f]{8})(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """One canonical local source and the turns/documents extracted from it."""

    relative_path: str
    kind: str
    format: str
    turns: tuple[tuple[str, str], ...]
    sha256: str
    byte_size: int
    source_family: str | None = None
    duplicate_paths: tuple[str, ...] = ()

    @property
    def character_count(self) -> int:
        return sum(len(text) for _, text in self.turns)

    def manifest_record(self) -> dict[str, Any]:
        """Metadata-only representation; source text stays in its repository."""
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "format": self.format,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "source_family": self.source_family,
            "turn_count": len(self.turns),
            "character_count": self.character_count,
            "roles": dict(Counter(role for role, _ in self.turns)),
            "duplicate_paths": list(self.duplicate_paths),
        }


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    root: Path
    sources: tuple[CorpusSource, ...]
    skipped: tuple[tuple[str, str], ...]
    scanned_files: int
    scanned_bytes: int

    def manifest(self) -> dict[str, Any]:
        duplicate_files = sum(len(source.duplicate_paths) for source in self.sources)
        duplicate_bytes = sum(
            source.byte_size * len(source.duplicate_paths) for source in self.sources
        )
        return {
            "root": str(self.root),
            "scanned_files": self.scanned_files,
            "scanned_bytes": self.scanned_bytes,
            "canonical_sources": len(self.sources),
            "conversation_sources": sum(
                source.kind == "conversation" for source in self.sources
            ),
            "document_sources": sum(
                source.kind == "document" for source in self.sources
            ),
            "turns": sum(len(source.turns) for source in self.sources),
            "characters": sum(source.character_count for source in self.sources),
            "exact_duplicate_files_removed": duplicate_files,
            "exact_duplicate_bytes_removed": duplicate_bytes,
            "source_families": len(
                {source.source_family for source in self.sources if source.source_family}
            ),
            "skipped_by_reason": dict(Counter(reason for _, reason in self.skipped)),
            "sources": [source.manifest_record() for source in self.sources],
            "skipped": [
                {"relative_path": path, "reason": reason}
                for path, reason in self.skipped
            ],
        }


@dataclass(frozen=True, slots=True)
class CorpusRecallEpisode:
    episode_id: str
    source_family: str
    source_path: str
    source_turn: int
    text: str


@dataclass(frozen=True, slots=True)
class CorpusRecallQuestion:
    question_id: str
    source_family: str
    source_path: str
    source_turn: int
    question: str
    gold_episode_id: str


@dataclass(frozen=True, slots=True)
class CorpusRecallSlice:
    """Conversation answers plus omitted user prompts used as recall probes."""

    source_paths: tuple[str, ...]
    episodes: tuple[CorpusRecallEpisode, ...]
    questions: tuple[CorpusRecallQuestion, ...]


def build_conversation_recall_slice(
    inventory: CorpusInventory,
    *,
    path_pattern: str,
    max_source_families: int = 3,
    questions_per_family: int = 3,
    min_query_chars: int = 16,
    max_query_chars: int = 1000,
    min_answer_chars: int = 160,
) -> CorpusRecallSlice:
    """Build a deterministic QA slice with no prompt text in the index.

    At most one non-exact export from each ``source_family`` is admitted. The
    assistant turns form the retrieval corpus; selected preceding user turns
    are held out as queries and point to their paired assistant episode.
    """
    if max_source_families < 1:
        raise ValueError("max_source_families must be positive")
    if questions_per_family < 1:
        raise ValueError("questions_per_family must be positive")
    pattern = re.compile(path_pattern, re.IGNORECASE)
    grouped: dict[str, list[CorpusSource]] = defaultdict(list)
    for source in inventory.sources:
        if source.kind != "conversation" or not pattern.search(source.relative_path):
            continue
        family = source.source_family or source.sha256[:12]
        grouped[family].append(source)

    # Repeated dated exports from one conversation are one source family. Use
    # the most complete representative so a near-copy cannot leak into the set.
    representatives = [
        max(
            family_sources,
            key=lambda source: (
                len(source.turns),
                source.character_count,
                source.relative_path,
            ),
        )
        for family_sources in grouped.values()
    ]
    representatives.sort(key=lambda source: source.relative_path)
    representatives = representatives[:max_source_families]

    episodes: list[CorpusRecallEpisode] = []
    questions: list[CorpusRecallQuestion] = []
    for source in representatives:
        family = source.source_family or source.sha256[:12]
        episode_by_turn: dict[int, str] = {}
        for turn_index, (role, text) in enumerate(source.turns):
            if role != "assistant":
                continue
            episode_id = f"{family}:{source.sha256[:8]}:{turn_index}"
            episode_by_turn[turn_index] = episode_id
            episodes.append(
                CorpusRecallEpisode(
                    episode_id=episode_id,
                    source_family=family,
                    source_path=source.relative_path,
                    source_turn=turn_index,
                    text=text,
                )
            )

        eligible: list[tuple[int, str, int]] = []
        for turn_index, (role, text) in enumerate(source.turns[:-1]):
            answer_role, answer = source.turns[turn_index + 1]
            if (
                role == "user"
                and answer_role == "assistant"
                and min_query_chars <= len(text) <= max_query_chars
                and len(answer) >= min_answer_chars
                and turn_index + 1 in episode_by_turn
            ):
                eligible.append((turn_index, text, turn_index + 1))
        count = min(questions_per_family, len(eligible))
        if count == 1:
            selected_positions = [len(eligible) // 2]
        elif count > 1:
            selected_positions = [
                round(index * (len(eligible) - 1) / (count - 1))
                for index in range(count)
            ]
        else:
            selected_positions = []
        for question_number, position in enumerate(selected_positions):
            turn_index, query, answer_turn = eligible[position]
            questions.append(
                CorpusRecallQuestion(
                    question_id=f"{family}-q{question_number}",
                    source_family=family,
                    source_path=source.relative_path,
                    source_turn=turn_index,
                    question=query,
                    gold_episode_id=episode_by_turn[answer_turn],
                )
            )

    return CorpusRecallSlice(
        source_paths=tuple(source.relative_path for source in representatives),
        episodes=tuple(episodes),
        questions=tuple(questions),
    )


def parse_chatgpt_markdown(text: str) -> list[tuple[str, str]]:
    """Parse copy/paste exports using ``You said:`` / ``ChatGPT said:``."""
    role_map = {
        "You said": "user",
        "ChatGPT said": "assistant",
        "Claude said": "assistant",
    }
    return [
        (role_map[match.group(1)], match.group(2).strip())
        for match in _CHATGPT_MARKER_RE.finditer(text)
        if match.group(2).strip()
    ]


def parse_conversation_text(text: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Detect supported text-export syntax independently of file extension."""
    candidates = [
        ("claude-text", parse_txt(text)),
        ("bold-role-markdown", parse_md(text)),
        ("chatgpt-markdown", parse_chatgpt_markdown(text)),
    ]
    format_name, turns = max(candidates, key=lambda candidate: len(candidate[1]))
    roles = {role for role, _ in turns}
    if len(turns) >= 2 and {"user", "assistant"}.issubset(roles):
        return format_name, turns
    return None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    text: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, dict):
            value = part.get("text") or part.get("content")
            if isinstance(value, str):
                text.append(value)
    return "\n".join(text).strip()


def parse_chatgpt_json(data: Any) -> list[tuple[str, str]]:
    """Parse the longest active-looking branch of ChatGPT mapping JSON."""
    if not isinstance(data, dict) or not isinstance(data.get("mapping"), dict):
        return []
    mapping: dict[str, Any] = data["mapping"]
    leaves = [
        node_id
        for node_id, node in mapping.items()
        if isinstance(node, dict) and not node.get("children")
    ]
    paths: list[list[dict[str, Any]]] = []
    for leaf in leaves:
        path: list[dict[str, Any]] = []
        seen: set[str] = set()
        node_id: str | None = leaf
        while node_id and node_id not in seen:
            seen.add(node_id)
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                break
            path.append(node)
            parent = node.get("parent")
            node_id = parent if isinstance(parent, str) else None
        paths.append(list(reversed(path)))

    def usable(path: list[dict[str, Any]]) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for node in path:
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author")
            role = author.get("role") if isinstance(author, dict) else None
            if role not in {"user", "assistant"}:
                continue
            text = _message_text(message)
            if text:
                turns.append((role, text))
        return turns

    branches = [usable(path) for path in paths]
    return max(branches, key=len, default=[])


def _notebook_turns(data: Any) -> list[tuple[str, str]]:
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        return []
    turns: list[tuple[str, str]] = []
    for index, cell in enumerate(data["cells"]):
        if not isinstance(cell, dict) or cell.get("cell_type") not in {
            "markdown",
            "code",
        }:
            continue
        source = cell.get("source")
        if isinstance(source, list):
            text = "".join(str(part) for part in source).strip()
        elif isinstance(source, str):
            text = source.strip()
        else:
            text = ""
        if text:
            turns.append(("system", f"[notebook {cell['cell_type']} cell {index}]\n{text}"))
    return turns


def _docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            document = archive.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile):
        return ""
    root = ElementTree.fromstring(document)
    paragraphs: list[str] = []
    for paragraph in root.iter():
        if paragraph.tag.endswith("}p"):
            text = "".join(
                node.text or "" for node in paragraph.iter() if node.tag.endswith("}t")
            ).strip()
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _family(path: Path) -> str | None:
    match = _EXPORT_ID_RE.search(path.stem)
    return match.group(1).lower() if match else None


def _load_source(path: Path, relative_path: str, raw: bytes) -> CorpusSource | None:
    suffix = path.suffix.lower()
    digest = hashlib.sha256(raw).hexdigest()
    family = _family(path)

    if suffix == ".ipynb":
        try:
            turns = _notebook_turns(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not turns:
            return None
        return CorpusSource(
            relative_path,
            "document",
            "jupyter-notebook",
            tuple(turns),
            digest,
            len(raw),
            family,
        )

    if suffix == ".json":
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        turns = parse_chatgpt_json(data)
        if not turns:
            return None
        return CorpusSource(
            relative_path,
            "conversation",
            "chatgpt-json",
            tuple(turns),
            digest,
            len(raw),
            family,
        )

    if suffix == ".docx":
        text = _docx_text(path)
        if not text:
            return None
        return CorpusSource(
            relative_path,
            "document",
            "docx",
            (("system", text),),
            digest,
            len(raw),
            family,
        )

    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    conversation = parse_conversation_text(text)
    if conversation is not None:
        format_name, turns = conversation
        return CorpusSource(
            relative_path,
            "conversation",
            format_name,
            tuple(turns),
            digest,
            len(raw),
            family,
        )
    return CorpusSource(
        relative_path,
        "document",
        f"plain-{suffix.lstrip('.')}",
        (("system", text),),
        digest,
        len(raw),
        family,
    )


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _SKIPPED_DIRS for part in relative.parts):
            continue
        yield path


def load_corpus_directory(directory: str | Path) -> CorpusInventory:
    """Classify and exact-deduplicate a mixed local notes repository."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    files = list(_iter_files(root))
    supported: dict[str, list[tuple[Path, bytes]]] = defaultdict(list)
    skipped: list[tuple[str, str]] = []
    scanned_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        scanned_bytes += path.stat().st_size
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            skipped.append((relative, "unsupported-extension"))
            continue
        raw = path.read_bytes()
        supported[hashlib.sha256(raw).hexdigest()].append((path, raw))

    sources: list[CorpusSource] = []
    for group in supported.values():
        # Exact duplicates have identical semantic content.  Pick the shortest
        # stable relative path and retain every alias in manifest metadata.
        group.sort(key=lambda item: (len(item[0].relative_to(root).parts), item[0].relative_to(root).as_posix()))
        canonical_path, raw = group[0]
        relative = canonical_path.relative_to(root).as_posix()
        source = _load_source(canonical_path, relative, raw)
        aliases = tuple(path.relative_to(root).as_posix() for path, _ in group[1:])
        if source is None:
            reason = "unsupported-json-schema" if canonical_path.suffix.lower() == ".json" else "empty-or-unreadable"
            skipped.append((relative, reason))
            skipped.extend((alias, "exact-duplicate-of-skipped-source") for alias in aliases)
            continue
        sources.append(replace(source, duplicate_paths=aliases))

    sources.sort(key=lambda source: source.relative_path)
    skipped.sort()
    return CorpusInventory(
        root=root,
        sources=tuple(sources),
        skipped=tuple(skipped),
        scanned_files=len(files),
        scanned_bytes=scanned_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    inventory = load_corpus_directory(args.path)
    manifest = inventory.manifest()
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"root: {inventory.root}")
    print(
        f"scanned: {manifest['scanned_files']} files / "
        f"{manifest['scanned_bytes']:,} bytes"
    )
    print(
        f"canonical: {manifest['canonical_sources']} "
        f"({manifest['conversation_sources']} conversations, "
        f"{manifest['document_sources']} documents), "
        f"{manifest['turns']:,} turns"
    )
    print(
        f"exact duplicates removed: {manifest['exact_duplicate_files_removed']} "
        f"files / {manifest['exact_duplicate_bytes_removed']:,} bytes"
    )
    print(f"source families: {manifest['source_families']}")
    print(f"skipped: {manifest['skipped_by_reason']}")
    if args.manifest:
        print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
