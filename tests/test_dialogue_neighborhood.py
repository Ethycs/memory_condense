import numpy as np
import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.dialogue_neighborhood import DialogueNeighborhoodRouter
from memory_condense.search.native_spine_context_routing import NativeSpineContextRouter
from tests.test_native_spine_context_routing import parent_fixture, QUERY, DATED


def build(future=False):
    history,semantic,hierarchy,_=parent_fixture(future=future)
    native=NativeSpineContextRouter(semantic,hierarchy)
    base=native.route_vector(QUERY,DATED,np.array([1,0]),embedding_identity=semantic.embedding_identity,
        max_direct=1,lexical_reserve=0,protected_direct=0,max_additions=0)
    episodes=[s for s in hierarchy.sections if not s.child_section_ids]
    survey=DialogueNeighborhoodRouter(native,np.array([[1,0]]*len(episodes),dtype=np.float32),
        embedding_identity='survey-encoder')
    return history,base,survey


def test_survey_keeps_exact_sources_roles_and_dated_scope():
    history,base,survey=build(future=True)
    plan,coverage=survey.route(base,np.array([1,0]),embedding_identity='survey-encoder')
    assert {r.section.section_id for r in base.expanded.routes}<={r.section.section_id for r in plan.routes}
    assert {r.section.spans[0].role for r in plan.routes}=={'user','assistant'}
    assert all(r.section.source_id in base.baseline.eligible_source_ids for r in plan.routes)
    assert coverage['eligible_episodes']==2 and not coverage['exhaustive']
    assert coverage['raw_reads_during_routing']==coverage['query_qwen_passes']==0
    result=hydrate_section_plan(plan,load_turn=history.get_turn,max_context_tokens=160,max_raw_spans=8)
    assert result.sections and result.context_token_count<=160
    for section in result.sections:
        for e in section.evidence:
            assert e.text==history.get_turn(e.span.turn_id).text[e.span.start_char:e.span.end_char]


def test_survey_rejects_encoder_change_and_corrupt_sources():
    history,base,survey=build()
    with pytest.raises(ValueError,match='encoder changed'):
        survey.route(base,np.array([1,0]),embedding_identity='other')
    plan,_=survey.route(base,np.array([1,0]),embedding_identity='survey-encoder')
    result=hydrate_section_plan(plan,load_turn=lambda tid:history.get_turn(tid).model_copy(update={'text':'corrupt'}))
    assert not result.sections
    assert {d.reason for d in result.diagnostics}=={'raw_turn_identity_changed'}


def test_closing_reply_precedes_activity_without_inventing_a_final_reply_for_open_leaf():
    history,base,survey=build()
    plan,coverage=survey.route(base,np.array([1,0]),embedding_identity='survey-encoder',closing_responses=True)
    # First leaf is user+assistant; the second ends with a user, so has no answer.
    assert plan.routes[0].section==history.atoms[1]
    assert coverage['closing_atomic_ids']==[history.atoms[1].section_id]
    assert not coverage['exhaustive']
