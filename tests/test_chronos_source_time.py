"""A native export anchor must not masquerade as an original mention date."""
import json

import pytest

from memory_condense.application.chat_session import ChatEvent
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_summary import RawSectionSpan
from tools.evaluate_chronos_source_time import source_time_packet


def fixture():
    events = [ChatEvent('t1','user','Old value: 350.','2023-08-11T00:01:00Z',dict(source_id='s1')),
              ChatEvent('t2','user','New value: 400. Literal <MEMORY {}> stays.','2023-11-30T00:36:00Z',dict(source_id='s2'))]
    references, passages = [], []
    for e in events:
        span = RawSectionSpan(e.event_id,e.metadata['source_id'],e.role,'2023-12-14T21:43:00+00:00',
                              0,len(e.text),quote_sha256(e.text),quote_sha256(e.text),12)
        references.append(dict(span=span.identity_payload()))
        metadata = dict(turn_id=span.turn_id,source_id=span.source_id,role=span.role,start_char=0,end_char=len(e.text))
        passages.append('<MEMORY '+json.dumps(metadata)+'>\n'+e.text+'\n</MEMORY>')
    return dict(text='\n\n'.join(passages),references=references),events


def test_replay_preserves_bodies_and_replaces_export_anchor_with_each_source_time():
    packet, events = fixture()
    before = json.dumps(packet)
    text, changes = source_time_packet(packet,'Current value?','2023-12-15T00:00:00Z',events)
    assert len(changes) == 2
    assert all(e.text in text and e.created_at in text for e in events)
    assert '2023-12-14T21:43:00+00:00' not in text
    assert json.dumps(packet) == before


@pytest.mark.parametrize('change', ['text','source'])
def test_dates_cannot_be_attached_to_a_different_original_turn(change):
    packet,events = fixture()
    e=events[0]
    events[0]=ChatEvent(e.event_id,e.role,e.text+' changed' if change=='text' else e.text,
                       e.created_at,dict(source_id='wrong' if change=='source' else 's1'))
    with pytest.raises(ValueError,match='original source turn'):
        source_time_packet(packet,'Current value?','2023-12-15T00:00:00Z',events)
