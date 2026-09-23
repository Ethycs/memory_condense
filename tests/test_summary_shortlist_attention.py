from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_shortlist_attention import rerank_summary_shortlist


class Linker:
    max_candidates = 4
    max_workspace_tokens = 1024
    layer = 1
    head_vote_k = 2
    cav_bank = None

    def __init__(self, mode=None):
        self.mode, self.inputs = mode, []
        self.encoder = SimpleNamespace(checkpoint_identity=SimpleNamespace(
            model_id="Qwen/Qwen3-8B", model_revision="test", checkpoint_sha256="a" * 64),
            device="cpu", dtype_name="float32", layers=2)

    def inspect_coverage(self, query, candidates):
        self.inputs.append((query, [c.text for c in candidates]))
        hits = [SimpleNamespace(episode_id=c.episode_id, qk_score=-1.0 if "completed" in c.text else -5.0,
                               ov_transport=0.5, transport_signature=object()) for c in candidates]
        if self.mode == "foreign":
            hits[0].episode_id = "foreign"
        if self.mode == "partial":
            hits.pop()
        if self.mode == "nan":
            hits[0].qk_score = float("nan")
        return SimpleNamespace(hits=hits, passes=2 if self.mode == "multiple" else 1,
            total_candidate_inspections=len(hits), workspace_candidates=len(hits), workspace_tokens=100)


def population():
    turns = [Turn(turn_id=str(i), source_id="source", role="user", text=f"raw secret {i} \r\nτ",
                  created_at=datetime(2026, 9, 9, tzinfo=timezone.utc)) for i in range(2)]
    index = SectionSummaryIndex([SectionSummary(str(i), "source", summary,
        (RawSectionSpan.from_turn(t),), "terra") for i, (t, summary) in enumerate(zip(turns,
        ["mountain travel planned", "mountain travel completed"]))])
    return turns, index, index.route("mountain travel", max_sections=2)


def test_one_pass_reads_only_summaries_then_hydrates_exact_selected_raw():
    turns, index, shortlist = population()
    linker = Linker()
    plan = rerank_summary_shortlist("mountain travel", index, shortlist, linker=linker, max_sections=1)
    assert len(linker.inputs) == 1 and "raw secret" not in repr(linker.inputs)
    assert plan.routes[0].section.section_id == "1"
    assert plan.attention_receipt.rounds[0].model_passes == 1
    assert plan.attention_receipt.rounds[0].selected_qk_scores == (-1.0,)
    assert plan.attention_receipt.retained_transformer_token_state_bytes == 0
    hydrated = hydrate_section_plan(plan, load_turn={t.turn_id: t for t in turns}.get)
    assert hydrated.sections[0].evidence[0].text == turns[1].text


@pytest.mark.parametrize("mode", ["partial", "foreign", "nan", "multiple"])
def test_partial_or_invalid_attention_cannot_silently_drop_candidates(mode):
    _, index, shortlist = population()
    linker = Linker(mode)
    with pytest.raises(ValueError):
        rerank_summary_shortlist("mountain travel", index, shortlist, linker=linker)
    assert len(linker.inputs) == 1


def test_binding_and_workspace_fail_before_model_call():
    _, index, shortlist = population()
    linker = Linker()
    with pytest.raises(ValueError, match="binding"):
        rerank_summary_shortlist("different query", index, shortlist, linker=linker)
    linker.max_candidates = 1
    with pytest.raises(ValueError, match="workspace"):
        rerank_summary_shortlist("mountain travel", index, shortlist, linker=linker)
    assert not linker.inputs


def test_empty_scope_requires_no_attention():
    _, index, _ = population()
    linker = Linker()
    shortlist = index.route("mountain travel", eligible_source_ids=[])
    plan = rerank_summary_shortlist("mountain travel", index, shortlist, linker=linker)
    assert not linker.inputs and not plan.routes and not plan.attention_receipt.rounds
