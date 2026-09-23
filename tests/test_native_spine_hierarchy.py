from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.attention_hierarchy import compile_attention_atoms
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.surprise_models import ScoredSurpriseSequence
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tests.test_attention_summary_sections import SummaryLinker
from tests.test_native_spine_exchanges import Backend
from tools import compile_native_spine_attention as attention
from tools import compile_native_spine_hierarchy as hierarchy
from tools.compile_native_spine_exchanges import NeutralJournal
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def group(name, count):
    turns = [Turn(turn_id=f"{name}-{i}", source_id=name,
                  role="user" if i % 2 == 0 else "assistant",
                  created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), text=f"RAW_CANARY_{i} " * 80)
             for i in range(count * 2)]
    texts = iter((f"Event {i}. " + ("User plans a visit and has not bought the tickets. " * 6))
                 if i % 2 == 0 else "Assistant suggests suitable places and possible visiting times. " * 8
                 for i in range(count * 2))
    atoms = compile_attention_atoms(turns, summarize_raw=lambda _: next(texts), summarizer_identity="raw-fixture",
                                    atom_token_cap=2048, max_summary_tokens=128)
    exchanges = compile_user_spine_exchanges(atoms, summarize=ReusingSpineSummarizer(lambda _: None),
                                            summarizer_identity="fixture", max_channel_tokens=128)
    return SimpleNamespace(sha256="exchanges-" + name), SimpleNamespace(sha256="atoms-" + name), atoms, exchanges


def test_bounded_hierarchy_work_preserves_atomic_addresses_and_replays(tmp_path):
    ready_sha, pending_sha = identity_sha256("ready"), identity_sha256("pending")
    groups = {ready_sha: group("ready", 1), pending_sha: group("pending", 4)}
    plan, _ = publish_sealed_json(tmp_path / "preflight.json", {"fixture": True})
    backend = Backend()
    scorer = QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128)
    journal = NeutralJournal(tmp_path, plan, backend, 0)
    partial = hierarchy.compile_groups(tmp_path, plan, groups, scorer, journal)
    assert set(partial) == {ready_sha} and backend.calls == 0
    assert not (tmp_path / "hierarchies" / f"{pending_sha}.json").exists()
    journal = NeutralJournal(tmp_path, plan, backend, 128)
    journal.replay()
    done = hierarchy.compile_groups(tmp_path, plan, groups, scorer, journal)
    assert set(done) == set(groups) and backend.calls > 0
    for sha, binding in done.items():
        saved = read_sealed_json(tmp_path / binding["path"])
        tree = SectionSummaryIndex.from_json(saved.payload["index_json"])
        atoms = SectionSummaryIndex.from_json(saved.payload["atomic_index_json"])
        assert atoms.sections == SectionSummaryIndex(groups[sha][2]).sections
        root = next(s for s in tree.sections if s.section_id == saved.payload["root_section_ids"][0])
        assert root.spans == tuple(s for a in groups[sha][2] for s in a.spans)
        assert saved.payload["original_atomic_addresses_preserved"] is True
    assert done[pending_sha]["parent_count"] > 0
    calls = backend.calls
    journal = NeutralJournal(tmp_path, plan, backend, 0)
    journal.replay()
    assert hierarchy.compile_groups(tmp_path, plan, groups, scorer, journal) == done
    assert backend.calls == calls


def signal_cache(root, *, wrong_dtype=False):
    cache = root / "cache"
    method = attention.cache_method(cache)
    texts = ("User plans a visit.", "User has not booked tickets.")
    signal = QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128).score_sequence(texts)
    fields = ("model_id", "model_revision", "checkpoint_sha256", "device", "dtype", "prefix_layers",
              "attention_layer", "head_vote_k", "max_input_spans", "span_token_cap", "linker_max_candidates",
              "linker_max_workspace_tokens", "owned_runtime_binding")
    # Synthetic receipt only: test the read-only cache contract without a GPU.
    receipt = replace(signal.receipt, **{field: method.payload[field] for field in fields}, receipt_sha256="")
    if wrong_dtype:
        receipt = replace(receipt, dtype="bfloat16", receipt_sha256="")
    signal = ScoredSurpriseSequence(signal.scores, signal.similarities, receipt)
    key = identity_sha256({"preflight_sha256": method.sha256, "texts": list(texts)})
    row, _ = publish_sealed_json(cache / "attention" / f"{key}.json", {
        "preflight_sha256": method.sha256, "scores": signal.scores,
        "similarities": signal.similarities, "receipt": signal.receipt.identity_payload(),
    })
    plan, _ = publish_sealed_json(root / "preflight.json", {
        "cache_root": str(cache.resolve()), "implementation": attention.implementation(),
        "cache_method_sha256": method.sha256, "raw_inputs_to_qwen": False, "jobs": {key: list(texts)},
    })
    publish_sealed_json(root / "result.json", {
        "preflight_sha256": plan.sha256, "cache_method_sha256": method.sha256,
        "all_prepared_attention_complete": True, "raw_inputs_to_qwen": False,
        "window_count": 1, "receipts": [{"key": key, "artifact_sha256": row.sha256,
                                           "signal_receipt_sha256": signal.receipt.receipt_sha256}],
    })
    return texts, signal, row.path


def test_frozen_attention_rejects_unknown_or_changed_signals_without_an_encoder(tmp_path):
    texts, signal, path = signal_cache(tmp_path)
    scorer = hierarchy.FrozenAttention(tmp_path)
    assert scorer.score_sequence(texts) == signal
    with pytest.raises(ValueError, match="unprepared"):
        scorer.score_sequence(["RAW_CANARY or an unknown summary"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        hierarchy.FrozenAttention(tmp_path).score_sequence(texts)


def test_frozen_attention_enforces_the_pinned_precision(tmp_path):
    texts, _, _ = signal_cache(tmp_path, wrong_dtype=True)
    with pytest.raises(ValueError, match="pinned method"):
        hierarchy.FrozenAttention(tmp_path).score_sequence(texts)
