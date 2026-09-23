from datetime import datetime, timezone
import json
from types import SimpleNamespace

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.user_spine_hierarchy import _render_channels
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.spine_parent_hierarchy import SourceSpineParentPlan
from tools import build_spine_corpus_hierarchy_resilient as builder
from tools.build_spine_corpus_hierarchy import compile_waves
from tools.matched_eval.artifacts import publish_sealed_json


def test_parent_dependency_executes_real_runtime_then_replays_without_a_client(tmp_path, monkeypatch):
    spans, leaves = [], []
    for i in range(2):
        turn = Turn(turn_id=f"private-{i}", source_id="private-source", role="user",
            text=f"RAW_CANARY_{i}", created_at=datetime(2026, 9, 8, tzinfo=timezone.utc))
        span = RawSectionSpan.from_turn(turn)
        spans.append(span)
        leaves.append(SectionSummary("spine-section-" + identity_sha256([span.receipt_sha256]),
            turn.source_id, _render_channels(("User owns a distinct tool. " * 16) + str(i), None, (span,)),
            (span,), "fixture"))
    sid = "spine-section-" + identity_sha256([s.receipt_sha256 for s in spans])
    plan = SourceSpineParentPlan(leaves, spans,
        [{"section_id": sid, "split_atom": 1, "attention_change": .2}])
    preflight, _ = publish_sealed_json(tmp_path / "preflight.json", {"synthetic_fixture": True})
    sent = []

    class Client:
        max_retries = 0
        chat = property(lambda self: SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        def with_options(self, **kwargs):
            return self
        def close(self):
            pass
        def create(self, **kwargs):
            sent.append(kwargs)
            assert "RAW_CANARY" not in repr(kwargs) and "private-" not in repr(kwargs)
            jobs = json.loads(kwargs["messages"][1]["content"])["jobs"]
            assert all(j["kind"] == "user_spine" for j in jobs)
            response = json.dumps({"summaries": [{"label": j["label"], "summary": "User owns distinct tools."} for j in jobs]})
            return SimpleNamespace(id="summary-response", model=builder.MODEL, usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content=response), finish_reason="stop")])

    monkeypatch.setattr(builder, "_completion_client", lambda *args: Client())
    journal = builder.RecoveryJournal(tmp_path, preflight, True, 1)
    def compile_with(journal):
        return compile_waves({"source":plan}, lambda p: p.compile(summarize=journal.cache,
            summarizer_identity=preflight.sha256), journal, "source_parents")["source"]
    result = compile_with(journal)
    assert len(sent) == journal.calls == 1
    assert len(result.sections) == 3
    assert sorted((s for s in result.sections if not s.child_section_ids), key=lambda s:s.section_id) == sorted(leaves, key=lambda s:s.section_id)
    monkeypatch.setattr(builder, "_completion_client", lambda *args: (_ for _ in ()).throw(AssertionError("replay called provider")))
    replay = builder.RecoveryJournal(tmp_path, preflight, False, 0)
    replay.replay()
    assert compile_with(replay).to_json() == result.to_json()
    assert replay.calls == 0 and replay.hits == 1
