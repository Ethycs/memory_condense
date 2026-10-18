"""Recall copies reuse canonical summaries without hiding modified evidence."""
from copy import deepcopy
from datetime import datetime
import json

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.recall_summary_reuse import recall_summary_alias
from tools.engineering_research_memory import Compiler


STAMP = '2026-09-29T00:00:00+00:00'


def packet():
    text = 'Deployment is planned, not executed. ' * 400
    turn = Turn(turn_id='original', source_id='source', role='user', text=text,
                created_at=datetime.fromisoformat(STAMP))
    span = RawSectionSpan.from_turn(turn)
    atom = SectionSummary('original-summary', 'source', 'Planned deployment, not executed.', (span,), 'test')
    label = dict(turn_id=span.turn_id, source_id=span.source_id, role=span.role,
                 start_char=span.start_char, end_char=span.end_char)
    row = dict(turn_id='_chat:recall:p1', role='system',
        text='<MEMORY ' + json.dumps(label) + '>\n' + text + '\n</MEMORY>',
        metadata={'_chat': dict(kind='recall', packet_id='p1',
                    references=[dict(section_id='parent', span=span.identity_payload(), independent=True)])})
    return atom, row


def test_whole_packet_preserves_raw_span_and_reuses_original_without_generation(tmp_path):
    atom, row = packet()
    compiler = Compiler(tmp_path, 'test', report=lambda **_: None)
    compiler.gateway.call = lambda *a, **kw: pytest.fail('A verified copy must not generate')
    output, = compiler.atoms([row], 'live', STAMP, original_atoms=(atom,))
    assert output.spans[0].end_char == len(row['text'])
    assert 'adds no new facts' in output.summary
    assert compiler.atoms([row], 'live', STAMP, original_atoms=(atom,)) == (output,)
    alias = recall_summary_alias(row, {atom.spans[0].receipt_sha256: atom})
    assert alias['reused_sections'][0]['section_receipt_sha256'] == atom.receipt_sha256


@pytest.mark.parametrize('change', ['body', 'extra', 'span', 'missing', 'role', 'identity'])
def test_changed_or_unavailable_packet_falls_back(change):
    atom, row = packet()
    originals = {atom.spans[0].receipt_sha256: atom}
    row = deepcopy(row)
    if change == 'body':
        row['text'] = row['text'].replace('planned', 'started', 1)
    elif change == 'extra':
        row['text'] += '\nnew unsupported fact'
    elif change == 'span':
        row['metadata']['_chat']['references'][0]['span']['end_char'] -= 1
    elif change == 'missing':
        originals = {}
    elif change == 'role':
        row['role'] = 'user'
    else:
        row['turn_id'] = 'external-tool'
    assert recall_summary_alias(row, originals) is None


def test_incremental_and_clean_compile_agree_for_live_source(tmp_path):
    source = dict(turn_id='fresh', source_id='live', role='user', text='Cluster juniper-641 remains planned.')
    compiler = Compiler(tmp_path/'incremental', 'test', report=lambda **_: None)
    first, = compiler.atoms([source], 'live', STAMP)
    span = first.spans[0]
    label = {k: getattr(span, k) for k in ('turn_id', 'source_id', 'role', 'start_char', 'end_char')}
    receipt = dict(turn_id='_chat:recall:new', role='system',
        text='<MEMORY ' + json.dumps(label) + '>\n' + source['text'] + '\n</MEMORY>',
        metadata={'_chat': dict(kind='recall', packet_id='new', references=[dict(span=span.identity_payload())])})
    incremental = (first, *compiler.atoms([receipt], 'live', STAMP, original_atoms=(first,)))
    clean = Compiler(tmp_path/'clean', 'test', report=lambda **_: None).atoms([source, receipt], 'live', STAMP)
    assert clean == incremental


@pytest.mark.parametrize('change', [None, 'asked_at', 'mentioned_at', 'missing_clock', 'preamble', 'query'])
def test_dated_packet_reuse_authenticates_both_clocks(tmp_path, change):
    from memory_condense.application.chat_native import exact_memory_passage
    from memory_condense.domain.question_time import dated_recall_question
    atom, row = packet()
    span = atom.spans[0]
    body = row['text'].split('>\n', 1)[1][:-len('\n</MEMORY>')]
    dated = dated_recall_question('Deployment?', STAMP)
    row['text'] = dated.split(']', 1)[0] + ']\n\n' + exact_memory_passage(span, body, span.role)
    receipt = row['metadata']['_chat']
    receipt.update(query='Deployment?', asked_at=STAMP, retrieval=dict(dated_question=dated))
    if change == 'asked_at':
        row['text'] = row['text'].replace('2026/09/29', '2026/09/28', 1)
    elif change == 'mentioned_at':
        row['text'] = row['text'].replace(STAMP, '2026-09-28T00:00:00+00:00', 1)
    elif change == 'missing_clock':
        del receipt['asked_at']
    elif change == 'preamble':
        row['text'] = 'New unsupported fact\n' + row['text']
    elif change == 'query':
        receipt['query'] = 'Another question'
    originals = {span.receipt_sha256: atom}
    if change:
        assert recall_summary_alias(row, originals) is None
    else:
        assert recall_summary_alias(row, originals)['reused_sections'][0]['span'] == span.identity_payload()
        compiler = Compiler(tmp_path, 'test', report=lambda **_: None)
        compiler.gateway.call = lambda *a, **kw: pytest.fail('Dated exact copies must not regenerate')
        copied, = compiler.atoms([row], 'live', STAMP, original_atoms=(atom,))
        assert 'adds no new facts' in copied.summary
