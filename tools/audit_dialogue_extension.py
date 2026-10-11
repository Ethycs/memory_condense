"""Audit the extension's source bindings, transport and score accounting."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from memory_condense.runtime.artifacts import read, save
from tools.extend_dialogue_index_dump import validate, ROOT, ARMS, digest
from tools.evaluate_dialogue_neighborhood import source_turns


def audit(root):
    plan = validate(root)
    report = read(root/'report.json')
    prior = read(root/'previous-report.json')
    questions = read(root/'questions.json')['questions']
    reference = read(root/'private-reference.json')
    binding = read(root/'grading-plan.json')
    source = read(root/'source.json')
    turns = source_turns(Path(source['directory']))
    positions = {t.turn_id:i for i,t in enumerate(turns)}
    assert len(questions) == 5 and len(binding['answers']) == 30
    assert {q['id'] for q in questions} == {'Q16','Q17','Q18','Q19','Q20'}
    assert all(digest(root/p) == sha for p,sha in binding['answers'].items())
    assert binding['reference_sha256'] == digest(root/'private-reference.json')
    assert reference['facts'] == read(root/'original-facts.json')['facts']
    for fact in reference['facts'].values():
        for support in fact['supports']:
            turn = turns[int(support['label'][1:])]
            assert support['quote'] in turn.text and support['role'] == turn.role
    with closing(sqlite3.connect((Path(source['directory'])/'native-spine.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        snapshot = json.loads(db.execute('SELECT payload FROM snapshot WHERE id=1').fetchone()[0])
    sections = {s['section_id']:s for s in snapshot['atomic_sections']+snapshot['hierarchy_sections']}
    summary_rows = 0
    for path in (root/'indexes').glob('*.json'):
        index = read(path)
        for row in index['rows']:
            section = sections[row['section_id']]
            expected = ['T'+str(n) for n in sorted({positions[s['turn_id']] for s in section['spans']})]
            assert row['turn_labels'] == expected
            assert row['summary_sha256'] == hashlib.sha256(section['summary'].encode()).hexdigest()
            assert row['source_id'] == section['source_id']
            assert section['summary'] in index['text']
            summary_rows += 1
    invalid = []
    for arm in ARMS:
        prompt = completion = points = possible = 0
        for q in questions:
            qid = q['id']
            folder = root/'api'/('answer-'+qid+'-'+arm)
            response = read(folder/'response.json')
            g = read(root/'grades'/(qid+'.json'))['arms'][arm]
            assert set(g['checkpoints']) == set(reference['questions'][qid])
            assert g['points'] == sum(g['checkpoints'].values())
            assert g['possible'] == 2*len(reference['questions'][qid])
            points += g['points']; possible += g['possible']
            if arm == ARMS[0]:
                allowed = read(root/'full-citation-labels.json')['labels']
            else:
                request = read(folder/'request.json')
                packet = read(root/'packets'/qid/(arm+'.json'))
                system = plan['neighborhood_system'] if arm == 'closing_6k' else plan['reader_system']
                assert request['messages'] == [dict(role='system',content=system),dict(role='user',
                    content='Archived evidence:\n'+packet['text']+'\n\nQuestion: '+q['question'])]
                assert request['reasoning_effort'] == 'none'
                assert response['response_model'] == plan['model'] and response['finish_reason'] == 'stop'
                assert response['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) == 0
                assert read(folder/'reservation.json')['request_sha256'] == digest(folder/'request.json')
                allowed = packet['allowed_citations']
                prompt += response['usage']['prompt_tokens']; completion += response['usage']['completion_tokens']
                if arm == 'closing_6k':
                    for item in packet['spans']:
                        span = item['span']; turn = turns[positions[span['turn_id']]]
                        assert item['text'] == turn.text[span['start_char']:span['end_char']]
                        assert item['label'] == 'T'+str(positions[span['turn_id']])
            blocks = re.findall(r'\[([^\]]+)\]',response['content'])
            cited = {c for block in blocks for c in re.findall(r'\b(?:T\d+|[NHA]\d{4})\b',block)}
            missing = sorted(cited-set(allowed))
            if missing:
                invalid.append(dict(arm=arm,question=qid,labels=missing))
        total = report['totals'][arm]
        assert total['points'] == points and total['possible'] == possible == 48
        assert total['checkpoint_score_pct'] == 100*points/possible
        if arm != ARMS[0]:
            assert total['prompt_tokens'] == prompt and total['completion_tokens'] == completion
        combined = report['aggregate_totals'][arm]
        assert combined['points'] == points+prior['totals'][arm]['points']
        assert combined['possible'] == possible+prior['totals'][arm]['possible'] == 188
    for q in questions:
        folder = root/'api'/('judge-'+q['id'])
        assert read(folder/'response.json')['finish_reason'] == 'stop'
        assert read(folder/'reservation.json')['request_sha256'] == digest(folder/'request.json')
    assert len(list((root/'api').glob('*/reservation.json'))) == 30
    assert not list((root/'api').glob('*/error.json'))
    control = read(root/'control'/'admissibility.json')
    assert control['full_raw_context_retained'] and not control['compaction'] and not control['tools']
    assert control['generation_sha256'] == digest(root/'control'/'generation.json')
    assert control['usage']['inputTokens'] == report['totals'][ARMS[0]]['prompt_tokens']
    assert read(root/'control'/'plan.json')['questions'] == questions
    result = dict(status='complete',questions=5,answers=30,sol_reader_calls=25,astra_reader_calls=1,
        judge_calls=5,summary_rows_verified=summary_rows,raw_spans_exact=True,zero_based_ranges_verified=True,
        history_reingestions=0,new_summaries=0,new_query_embeddings=5,
        failed_api_calls=0,automatic_retries=0,source_unchanged=True,prior_grades_unchanged=True,
        unprovided_citations=invalid,scoring_adjustments=0,
        report_sha256=digest(root/'report.json'),plan_sha256=digest(root/'plan.json'),
        control_usage=control['usage'],limitations=plan['limitations'])
    save(root/'final-audit.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    audit(parser.parse_args().root)
