from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools.assess_native_spine_packet_width import support_coverage


def inputs():
    source = {'occurrence_id': 'source-one', 'body_sha256': 'body-one'}
    turn_id = 'native-turn-' + identity_sha256({**source, 'turn_ordinal': 0})
    raw = SimpleNamespace(source_id='native-source-source-one', role='user', text='A café choice.')
    app = SimpleNamespace(transcript=SimpleNamespace(get_turn=lambda key: raw if key == turn_id else None))
    ref = {'source': {'source': source}, 'supports': [{'quote': 'café choice', 'turn_index': 0}]}
    def evidence(start, end):
        return {'span': {'source_id': raw.source_id, 'turn_id': turn_id, 'start_char': start, 'end_char': end},
                'text': raw.text[start:end]}
    packet = {'sections': [{'evidence': [evidence(0, 5), evidence(5, len(raw.text))]}]}
    return packet, ref, app


def test_coverage_rejoins_adjacent_spans_but_never_crosses_gaps_or_sources():
    packet, ref, app = inputs()
    assert support_coverage(packet, ref, app) == 'all'
    gapped = deepcopy(packet)
    gapped['sections'][0]['evidence'][1]['span']['start_char'] += 1
    gapped['sections'][0]['evidence'][1]['text'] = gapped['sections'][0]['evidence'][1]['text'][1:]
    assert support_coverage(gapped, ref, app) == 'none'
    foreign = deepcopy(packet)
    foreign['sections'][0]['evidence'][1]['span']['source_id'] = 'different-source'
    assert support_coverage(foreign, ref, app) == 'none'


def test_missing_or_invented_reference_support_does_not_pass_coverage():
    packet, ref, app = inputs()
    ref['supports'][0]['quote'] = 'An invented answer'
    with pytest.raises(ValueError, match='persisted original'):
        support_coverage(packet, ref, app)
    ref['supports'] = []
    with pytest.raises(ValueError, match='recorded quote'):
        support_coverage(packet, ref, app)
