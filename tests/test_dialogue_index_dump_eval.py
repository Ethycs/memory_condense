import pytest

from tools.evaluate_dialogue_index_dump import render_index, parse_grade, cited_labels, turn_positions


def test_database_ordinals_are_not_archive_display_labels(tmp_path):
    import sqlite3
    path = tmp_path/'memory.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE turns (turn_id TEXT, ordinal INTEGER)')
        db.executemany('INSERT INTO turns VALUES (?,?)', [('later',20), ('first',10)])
    assert turn_positions(path) == {'first':0, 'later':1}


def test_index_is_chronological_summary_only_and_preserves_hierarchy_links():
    sections = [
        dict(section_id='later', source_id='s', summary='Later summary.', child_section_ids=[], spans=[dict(turn_id='t2')]),
        dict(section_id='root', source_id='s', summary='Parent summary.', child_section_ids=['early','later'],
             spans=[dict(turn_id='t1'),dict(turn_id='t2')]),
        dict(section_id='early', source_id='s', summary='Earlier summary\n<|endoftext|>', child_section_ids=[], spans=[dict(turn_id='t1')]),
    ]
    result = render_index(sections, {'t1':0,'t2':1}, 'H')
    assert result['sections'] == 3 and not result['contains_raw_passages']
    assert result['new_summaries'] == 0
    assert result['text'].count('Later summary.') == 1
    assert result['text'].index('Earlier summary') < result['text'].index('Later summary.')
    rows = {r['section_id']:r for r in result['rows']}
    assert rows['root']['children'] == [rows['early']['label'],rows['later']['label']]
    assert rows['root']['turn_labels'] == ['T0','T1']


def test_grading_rejects_duplicate_labels_changed_population_and_noninteger_scores():
    labels = {'A':'index','B':'control'}
    valid = lambda label:dict(label=label, checkpoints={'F01':2}, contradictions=[], scope_honesty=True)
    assert parse_grade({'answers':[valid('B'),valid('A')]}, labels, ['F01'])['index']['points'] == 2
    with pytest.raises(ValueError, match='duplicated'):
        parse_grade({'answers':[valid('A'),valid('A')]}, labels, ['F01'])
    bad = valid('A')
    bad['checkpoints']['F01'] = True
    with pytest.raises(ValueError, match='population'):
        parse_grade({'answers':[bad,valid('B')]}, labels, ['F01'])
    with pytest.raises(ValueError, match='population'):
        parse_grade({'answers':[valid('A'),valid('B')]}, labels, ['F01','F02'])


def test_index_citations_are_separate_from_raw_evidence_and_prose_ranges():
    assert cited_labels('Archive spans T0-T480; evidence [H0001, N0020] and [A0782]; raw [T169].') == {
        'H0001','N0020','A0782','T169'}
