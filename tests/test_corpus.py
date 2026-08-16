from __future__ import annotations

import json
from pathlib import Path

from memory_condense.corpus import (
    build_conversation_recall_slice,
    load_corpus_directory,
    parse_chatgpt_json,
    parse_chatgpt_markdown,
    parse_conversation_text,
)


def test_parse_chatgpt_markdown():
    text = "You said:\nHello\n\nChatGPT said:\nHi there\n"
    assert parse_chatgpt_markdown(text) == [
        ("user", "Hello"),
        ("assistant", "Hi there"),
    ]


def test_conversation_detection_is_extension_independent():
    detected = parse_conversation_text("User:\nQuestion\n\nClaude:\nAnswer\n")
    assert detected == (
        "claude-text",
        [("user", "Question"), ("assistant", "Answer")],
    )


def test_parse_chatgpt_json_chooses_longest_branch():
    data = {
        "mapping": {
            "root": {"parent": None, "children": ["u"], "message": None},
            "u": {
                "parent": "root",
                "children": ["short", "long"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Question"]},
                },
            },
            "short": {
                "parent": "u",
                "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Old answer"]},
                },
            },
            "long": {
                "parent": "u",
                "children": ["followup"],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Current answer"]},
                },
            },
            "followup": {
                "parent": "long",
                "children": [],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Continue"]},
                },
            },
        }
    }
    assert parse_chatgpt_json(data) == [
        ("user", "Question"),
        ("assistant", "Current answer"),
        ("user", "Continue"),
    ]


def test_inventory_is_recursive_deduplicated_and_source_aware(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    chat = "User:\nQuestion\n\nClaude:\nAnswer\n"
    (tmp_path / "chat.txt").write_text(chat, encoding="utf-8")
    (nested / "chat-copy.md").write_text(chat, encoding="utf-8")
    (nested / "notes.md").write_text("# Result\nA durable theorem.", encoding="utf-8")
    (nested / "image.png").write_bytes(b"not really an image")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "object.txt").write_text(chat, encoding="utf-8")

    inventory = load_corpus_directory(tmp_path)
    manifest = inventory.manifest()

    assert manifest["scanned_files"] == 4
    assert manifest["canonical_sources"] == 2
    assert manifest["conversation_sources"] == 1
    assert manifest["document_sources"] == 1
    assert manifest["exact_duplicate_files_removed"] == 1
    assert manifest["skipped_by_reason"] == {"unsupported-extension": 1}
    conversation = next(s for s in inventory.sources if s.kind == "conversation")
    assert conversation.duplicate_paths == ("nested/chat-copy.md",)


def test_notebook_ignores_outputs(tmp_path: Path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Idea"]},
            {
                "cell_type": "code",
                "source": ["print('small')"],
                "outputs": [{"text": ["huge generated output"]}],
            },
        ]
    }
    (tmp_path / "work.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    inventory = load_corpus_directory(tmp_path)
    source = inventory.sources[0]
    combined = " ".join(text for _, text in source.turns)
    assert "Idea" in combined
    assert "print('small')" in combined
    assert "huge generated output" not in combined


def test_recall_slice_keeps_one_export_per_family_and_omits_queries(tmp_path: Path):
    short = "User:\nWhich route?\n\nClaude:\n" + "short answer " * 20
    long = (
        "User:\nWhich route should memory use?\n\nClaude:\n"
        + "first answer " * 20
        + "\n\nUser:\nHow should it prune?\n\nClaude:\n"
        + "second answer " * 20
    )
    other = "User:\nWhat is recalled?\n\nClaude:\n" + "other answer " * 20
    (tmp_path / "Memory_aaaaaaaa_old.txt").write_text(short, encoding="utf-8")
    (tmp_path / "Memory_aaaaaaaa_new.txt").write_text(long, encoding="utf-8")
    (tmp_path / "Attention_bbbbbbbb.txt").write_text(other, encoding="utf-8")

    inventory = load_corpus_directory(tmp_path)
    recall_slice = build_conversation_recall_slice(
        inventory,
        path_pattern="memory|attention",
        max_source_families=2,
        questions_per_family=2,
    )

    assert len(recall_slice.source_paths) == 2
    assert "Memory_aaaaaaaa_new.txt" in recall_slice.source_paths
    assert "Memory_aaaaaaaa_old.txt" not in recall_slice.source_paths
    assert len(recall_slice.questions) == 3
    indexed_text = " ".join(episode.text for episode in recall_slice.episodes)
    assert "Which route should memory use?" not in indexed_text
    assert {
        question.gold_episode_id for question in recall_slice.questions
    }.issubset({episode.episode_id for episode in recall_slice.episodes})
