from copy import deepcopy

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_frozen_native_spine_full100 import synthetic_population, fake_observations
from tests.test_native_spine_memory import body_and_atoms, source
from tools import audit_frozen_native_spine_full100 as audit
from tools.matched_eval.artifacts import SealedArtifactError


def packet():
    body, summaries = body_and_atoms()
    sessions = [source(body, 0, '2026-02-02')]
    history = materialize_history(sessions, load_body=lambda _: body,
        load_summaries=lambda _: summaries, compiler_identity='fixture')
    index = SectionSummaryIndex(history.atoms)
    plan = index.route('ROUTING_ADDRESS', max_sections=len(history.atoms))
    hydration = hydrate_section_plan(plan, load_turn=history.get_turn, max_context_tokens=1024,
        max_raw_spans=128).identity_payload()
    routing = {'expanded': plan.identity_payload(), 'protected_direct': 0, 'ancestor_hops': 1,
        'query_qwen_passes': 0, 'raw_reads_during_routing': 0}
    question = {'retrieval_query': 'What did I plan?',
        'prompt_question': '[Question asked at 2026/02/03 (Tue) 00:00] What did I plan?'}
    return body, sessions, question, hydration, routing


def test_independent_statistics_reproduce_known_scores_and_visible_tail(tmp_path):
    calls = synthetic_population(tmp_path)
    report = audit.independent_statistics(*fake_observations(calls))
    assert report['accuracy']['parent_context'] == {'correct': 95, 'questions': 100}
    assert report['latency']['parent_context']['e2e_total_s']['median_s'] == 4.8
    assert report['latency']['parent_context']['e2e_total_s']['p95_s'] == 7.0
    assert report['candidate_answers_under_five_seconds'] == 94
    assert report['target_gate_passed'] is True
    assert not audit.independent_statistics(*fake_observations(calls, correct=94))['target_gate_passed']
    assert not audit.independent_statistics(*fake_observations(calls, total=5.0))['target_gate_passed']
    observations, judgments = fake_observations(calls)
    judgments[0]['response_sha256'] = 'different-response'
    with pytest.raises(ValueError, match='measured response'):
        audit.independent_statistics(observations, judgments)
    with pytest.raises(ValueError, match='all 300'):
        audit.independent_statistics(observations[:-1], judgments)


def test_packet_audit_recovers_exact_original_unicode_and_omits_routing_text():
    body, sessions, q, h, r = packet()
    messages, spans = audit.verify_packet(q, h, r, sessions, lambda _: body)
    assert spans == sum(len(s['evidence']) for s in h['sections'])
    assert 'café' in messages[1]['content'] and '🌍' in messages[1]['content']
    assert 'ROUTING_ADDRESS' not in messages[1]['content']


@pytest.mark.parametrize('damage', ['raw_text', 'foreign_occurrence', 'future', 'changed_body', 'changed_plan', 'budget'])
def test_packet_audit_rejects_changed_evidence_scope_and_policy(damage):
    body, sessions, q, h, r = packet()
    if damage == 'raw_text':
        h['sections'][0]['evidence'][0]['text'] += ' forged'
    elif damage == 'foreign_occurrence':
        sessions = [source(body, 1, '2026-02-02')]
    elif damage == 'future':
        q['prompt_question'] = '[Question asked at 2026/01/03 (Sat) 00:00] What did I plan?'
    elif damage == 'changed_body':
        body['turns'][0]['text'] += ' Changed.'
    elif damage == 'changed_plan':
        r['expanded'] = deepcopy(r['expanded'])
        r['expanded']['query_sha256'] = '0'*64
    else:
        h['max_context_tokens'] = 3072
    with pytest.raises(ValueError):
        audit.verify_packet(q, h, r, sessions, lambda _: body)


def test_audit_cannot_open_references_before_a_completed_report(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.evaluation, 'load_references', lambda _: pytest.fail('opened gold before answers'))
    with pytest.raises(SealedArtifactError, match='artifact must be a regular file'):
        audit.audit(tmp_path)
