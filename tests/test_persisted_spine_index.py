from dataclasses import replace
from datetime import datetime, timezone
import hashlib

import numpy as np
import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.compile_spine_semantic_index import load_index, persist_vectors, restore_turns
from tools.matched_eval.artifacts import publish_sealed_json
from tests.test_semantic_section_index import Encoder, hierarchy


def test_compilation_method_identity_ignores_only_bound_population_and_cache_metadata():
    from memory_condense.domain._discourse_identity import identity_sha256
    from tools.compile_spine_semantic_index import compilation_policy_sha256
    method = {"model": "qwen3-8b", "max_channel_tokens": 128, "implementation": {"builder": "a" * 64}}
    payload = {**method, "method_policy_sha256": identity_sha256(method), "atoms_sha256": "atoms-a",
               "raw_span_population_sha256": "raw-a", "complete_namespace": True,
               "parent_caches": [{"root": "parent-a", "preflight_sha256": "parent-sha-a"}]}
    changed_namespace = {**payload, "atoms_sha256": "atoms-b", "raw_span_population_sha256": "raw-b",
                         "parent_caches": []}
    assert compilation_policy_sha256(payload) == compilation_policy_sha256(changed_namespace)
    with pytest.raises(ValueError, match="policy changed"):
        compilation_policy_sha256({**payload, "max_channel_tokens": 64})
    assert compilation_policy_sha256({**method, "atoms_sha256": "legacy"}) == identity_sha256(method)


def test_persisted_vectors_replay_routes_and_reject_tampering(tmp_path):
    _, index = hierarchy()
    encoder = Encoder()
    identity = summary_embedding_identity(encoder)
    matrix = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32)
    semantic, digest = persist_vectors(tmp_path, index, matrix, identity)
    publish_sealed_json(tmp_path / "index.json", {"matrix_file_sha256": digest,
        "index_json": index.to_json(), "embedding_identity": identity, "semantic_index_sha256": semantic.receipt_sha256})
    _, replay = load_index(tmp_path)
    assert replay.route("trip", encoder=encoder).receipt_sha256 == semantic.route("trip", encoder=encoder).receipt_sha256
    with (tmp_path / "summary-vectors.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_index(tmp_path)


def test_exact_fragment_reassembly_preserves_unicode_and_rejects_gaps():
    turn = Turn(turn_id="turn", source_id="source", role="user", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
                text="  Unicode τ\r\nMore text.  ")
    fragments = [RawSummaryFragment(RawSectionSpan.from_turn(turn, start_char=start, end_char=end), turn.text[start:end])
                 for start, end in ((0, 8), (8, len(turn.text)))]
    assert restore_turns(fragments)[0].text == turn.text
    with pytest.raises(ValueError, match="gap"):
        restore_turns(fragments[1:])
    with pytest.raises(ValueError, match="changed"):
        restore_turns([replace(fragments[0], span=replace(fragments[0].span, turn_text_sha256="a" * 64, receipt_sha256="")), fragments[1]])
