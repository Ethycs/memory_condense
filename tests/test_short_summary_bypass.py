"""Raw generation is unnecessary below 1.5 times the summary output budget."""
import json

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from tools import engineering_research_memory as memory


STAMP = '2026-09-29T00:00:00+00:00'


def test_strict_threshold_exact_text_provenance_and_cache(tmp_path, monkeypatch):
    calls = []
    class Gateway:
        def __init__(self, root):
            pass
        def call(self, kind, messages, **kwargs):
            assert kind == 'raw'
            fragments = json.loads(messages[1]['content'])['fragments']
            calls.extend(f['fragment'] for f in fragments)
            return dict(request_sha256='0'*64, content=json.dumps(dict(atoms=[
                dict(label=f['label'], summary='Generated summary.', support=['word']) for f in fragments])))
    monkeypatch.setattr(memory, 'Gateway', Gateway)
    compiler = memory.Compiler(tmp_path, 'threshold', report=lambda **_: None)
    texts = ['word' + ' word'*(n-1) for n in (191, 192, 193)]
    assert [count_tokens(t) for t in texts] == [191, 192, 193]
    rows = [dict(turn_id=f't{i}', role='user', text=t) for i,t in enumerate(texts)]
    atoms = compiler.atoms(rows, 'source', STAMP)
    assert calls == texts[1:]
    assert atoms[0].summary == texts[0]
    assert atoms[0].spans[0].span_text_sha256 == quote_sha256(texts[0])
    assert atoms[0].spans[0].turn_id == 't0'
    assert [a.summary for a in atoms[1:]] == ['Generated summary.']*2
    assert compiler.atoms(rows, 'source', STAMP) == atoms
    assert calls == texts[1:]
    records = [memory.read(p) for p in (tmp_path/'cache/atoms').glob('*.json')]
    verbatim, = [r for r in records if r.get('mode') == 'verbatim-short-v1']
    assert verbatim['support'] == [texts[0]] and 'request_sha256' not in verbatim


def test_existing_generated_receipt_stays_stable(tmp_path):
    text = 'Previously compiled short fact.'
    key = identity_sha256(dict(role='user', text=text, system=memory.RAW_SYSTEM))
    value = dict(role='user', text_sha256=quote_sha256(text), summary='Old summary.',
                 support=['short fact'], request_sha256='0'*64, support_escape_repairs=0)
    memory.save(tmp_path/'cache/atoms'/f'{key}.json', value)
    compiler = memory.Compiler(tmp_path, 'legacy', report=lambda **_: None)
    atom, = compiler.atoms([dict(turn_id='old', role='user', text=text)], 'source', STAMP)
    assert atom.summary == value['summary']
    assert atom.summarizer_identity == identity_sha256(value)
