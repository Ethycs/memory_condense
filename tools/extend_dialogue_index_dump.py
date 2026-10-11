"""Five new retrospective questions for all six frozen-index configurations.

Reuse the original memory and source-verified facts, never ingest or regenerate
summaries. Freeze new question/checkpoint mappings before any reader execution.
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time

import numpy as np

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.artifacts import read, save, emit
from tools.evaluate_dialogue_neighborhood import ACTOR_SYSTEM, MODEL, GATEWAY, digest, source_turns
from tools.evaluate_dialogue_index_dump import (
    ARMS, NEW_ARMS, INDEX_NOTE, render_index, turn_positions,
    api_job, bounded, parse_grade, cited_labels,
)
from tools.grade_dialogue_neighborhood import JUDGE, extract_json

ROOT = Path('eval_results/dialogue-index-extension-20261006-r2')
PRIOR = Path('eval_results/dialogue-index-dump-20261006-r2')
ORIGINAL = Path('eval_results/dialogue-neighborhood-20261006-r1')
ASTRA = Path('eval_results/dialogue-astra-appserver-20261006-r1')
QUESTIONS = [
    dict(id='Q16', question='A teammate proposes enabling every experimental retrieval option by default because the tests pass. What shipping recommendation would you make at T480, based on production wiring, replication, and the difference between committed work and unfinished delivery work?'),
    dict(id='Q17', question='We have a fixed context-token budget and must choose the next retrieval configuration to validate. What does the dialogue justify choosing, what tempting efficiency argument should we reject, and which workload still needs a safety check? Use the measured tradeoffs rather than assuming more recall always wins.'),
    dict(id='Q18', question='Suppose this conversation is reopened after a month without new messages. Should its memories have faded, and can the earlier forgetting experiment answer that? Trace the requirement correction through failed attempts, evaluation repairs, and the final reported behavior, distinguishing claims from validation.'),
    dict(id='Q19', question='A reviewer blames the reader model for weak temporal answers and the apparent drop from the original demo to short dialogue, then compares our percentage directly with a published benchmark. What alternative explanations does the archive establish, and what comparison should be run before accepting that diagnosis?'),
    dict(id='Q20', question='Construct a compact causal audit of how green component tests repeatedly missed the actual user workflow. Which integration failures were discovered, how were they repaired, and what measurement gap still remained at the end of the archive?'),
]
CHECKPOINTS = {
    'Q16': ['F04','F14','F18','F21','F22'],
    'Q17': ['F09','F10','F15','F16','F17'],
    'Q18': ['F19','F20','F21','F24'],
    'Q19': ['F06','F07','F08','F13','F18'],
    'Q20': ['F02','F03','F05','F20','F21'],
}


def validate(root):
    plan = read(root/'plan.json')
    for filename, sha in plan['implementation'].items():
        if digest(filename) != sha:
            raise ValueError('Prepared implementation changed: '+filename)
    for filename, sha in plan['frozen'].items():
        if digest(root/filename) != sha:
            raise ValueError('Frozen experiment input changed: '+filename)
    source = read(root/'source.json')
    for name, sha in source['files'].items():
        if digest(Path(source['directory'])/name) != sha:
            raise ValueError('Original source changed')
    return plan


def prepare(root):
    from memory_condense.persistence import native_spine_store, native_spine_parent_store
    from memory_condense.runtime.embedding import FastEmbedBGE
    from memory_condense.runtime.config import RuntimeAssets
    from memory_condense.search.summary_semantic_index import summary_embedding_identity, compatible_summary_embedding
    from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
    from memory_condense.application.dialogue_retrieval import PreparedDialogueRetrieval, DialogueRetrievalPolicy
    from memory_condense.application.native_spine_policy import CAP8_LIMITS
    from memory_condense.application.section_retrieval import hydrate_section_plan

    root.mkdir(parents=True, exist_ok=False)
    frozen = {}
    def freeze(relative, value):
        save(root/relative, value)
        frozen[str(relative)] = digest(root/relative)

    source = read(PRIOR/'source.json')
    directory = Path(source['directory'])
    for name, sha in source['files'].items():
        if digest(directory/name) != sha:
            raise ValueError('Original source changed')
    turns = source_turns(directory)
    positions = turn_positions(directory/'memory.db')
    reference = read(PRIOR/'private-reference.json')
    for fact in reference['facts'].values():
        for support in fact['supports']:
            turn = turns[int(support['label'][1:])]
            if support['quote'] not in turn.text or support['role'] != turn.role:
                raise ValueError('Reference is not an exact archived excerpt')
    reference['questions'] = CHECKPOINTS
    reference['method'] = 'New scenario questions and checkpoint mappings frozen before generation; unchanged source-verified facts from the original assay.'
    freeze('private-reference.json', reference)
    freeze('questions.json', dict(questions=QUESTIONS))
    freeze('source.json', source)
    freeze('previous-report.json', read(PRIOR/'report.json'))
    freeze('original-facts.json', dict(facts=read(PRIOR/'private-reference.json')['facts']))
    freeze('full-citation-labels.json', dict(labels=['T'+str(i) for i in range(len(turns))]))

    snapshot = native_spine_store.load(directory/'native-spine.sqlite', turns=turns)
    if snapshot.receipt != source['snapshot']:
        raise ValueError('Stored snapshot changed')
    with closing(sqlite3.connect((directory/'native-spine.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        raw = json.loads(db.execute('SELECT payload FROM snapshot WHERE id=1').fetchone()[0])
    hierarchy = raw['hierarchy_sections']
    indexes = dict(
        index_neighborhoods=render_index([s for s in hierarchy if not s['child_section_ids']], positions, 'N'),
        index_hierarchy=render_index(hierarchy, positions, 'H'),
        index_atomic=render_index(raw['atomic_sections'], positions, 'A'))
    for arm, packet in indexes.items():
        old = read(PRIOR/'indexes'/(arm+'.json'))
        fields = ('label','section_id','source_id','summary_sha256','children')
        if [[r[k] for k in fields] for r in old['rows']] != [[r[k] for k in fields] for r in packet['rows']]:
            raise ValueError('Stored summary content or alias changed')
        freeze(Path('indexes')/(arm+'.json'), packet)

    embeddings = read(ORIGINAL/'embedding.json')
    if digest(ORIGINAL/'episode-vectors.npy') != embeddings['episode_vectors_sha256']:
        raise ValueError('Cached episode vectors changed')
    if [s.section_id for s in snapshot.hierarchy.sections if not s.child_section_ids] != embeddings['episodes']:
        raise ValueError('Cached episode vector order changed')
    started = time.perf_counter()
    encoder = FastEmbedBGE(assets=RuntimeAssets.resolve('.cache'))
    try:
        if (summary_embedding_identity(encoder) != embeddings['identity']
                or not compatible_summary_embedding(encoder, snapshot.semantic.embedding_identity)):
            raise ValueError('Query embedding execution changed')
        vectors = encoder.embed_queries([q['question'] for q in QUESTIONS])
    finally:
        encoder.close()
    np.save(root/'query-vectors.npy', vectors)
    frozen['query-vectors.npy'] = digest(root/'query-vectors.npy')
    freeze('embedding.json', dict(identity=embeddings['identity'], fresh_query_embeddings=5,
        new_summary_embeddings=0, elapsed_s=time.perf_counter()-started,
        episode_vectors_sha256=embeddings['episode_vectors_sha256']))
    parents, _ = native_spine_parent_store.load(directory/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
    native = NativeSpineUserCompletionRouter(snapshot.semantic, snapshot.hierarchy, parents)
    policy = DialogueRetrievalPolicy()
    survey = PreparedDialogueRetrieval(native, np.load(ORIGINAL/'episode-vectors.npy'),
        embedding_identity=embeddings['identity'], policy=policy)
    by_id = {t.turn_id:t for t in turns}
    options = {k:v for k,v in CAP8_LIMITS.items() if k not in ('max_context_tokens','max_raw_spans')}
    asked = max(t.created_at for t in turns).strftime('%Y/%m/%d (%a)')
    for q, vector in zip(QUESTIONS, vectors, strict=True):
        started = time.perf_counter()
        base = native.route_vector(q['question'], f'[Question asked at {asked} 23:59] '+q['question'],
            vector, embedding_identity=snapshot.semantic.embedding_identity, **options)
        route = survey.route(base, vector, embedding_identity=embeddings['identity'])
        routing_s = time.perf_counter()-started
        started = time.perf_counter()
        hydrated = hydrate_section_plan(route.expanded, load_turn=by_id.get,
            max_raw_spans=policy.max_raw_spans, max_context_tokens=policy.max_context_tokens)
        spans, pieces = [], []
        for section in hydrated.sections:
            for evidence in section.evidence:
                span = evidence.span
                label = 'T'+str(positions[span.turn_id])
                if evidence.text != by_id[span.turn_id].text[span.start_char:span.end_char]:
                    raise ValueError('Hydrated evidence differs from original raw text')
                pieces.append(f'[{label} role={span.role} chars={span.start_char}:{span.end_char}]\n'+evidence.text)
                spans.append(dict(label=label, span=span.identity_payload(), text=evidence.text))
        neighborhood = '\n\n'.join(pieces)
        freeze(Path('packets')/q['id']/'closing_6k.json', dict(arm='closing_6k', text=neighborhood,
            spans=spans, allowed_citations=sorted({s['label'] for s in spans}),
            sent_evidence_tokens=count_tokens(neighborhood), rendered_tokens=hydrated.context_token_count,
            route_sha256=route.receipt_sha256, hydration_sha256=hydrated.receipt_sha256,
            policy_sha256=policy.receipt_sha256, routing_s=routing_s, hydration_s=time.perf_counter()-started,
            diagnostics=dict(Counter(d.reason for d in hydrated.diagnostics))))
        for arm in NEW_ARMS:
            index = indexes['index_hierarchy' if arm == 'hierarchy_plus_neighborhoods' else arm]
            text = 'STORED SUMMARY INDEX\n'+index['text']
            allowed = [r['label'] for r in index['rows']]
            if arm == 'hierarchy_plus_neighborhoods':
                text += '\n\nEXACT NEIGHBORHOOD EVIDENCE\n'+neighborhood
                allowed += [s['label'] for s in spans]
            freeze(Path('packets')/q['id']/(arm+'.json'), dict(arm=arm, text=text,
                allowed_citations=sorted(set(allowed)), sent_evidence_tokens=count_tokens(text)))
        emit(phase='packets_prepared', question=q['id'])

    # Use the same known-working app-server protocol, once for all five questions.
    control = root/'control'
    (control/'empty-workspace').mkdir(parents=True)
    freeze(Path('control')/'archive-item.json', read(ASTRA/'archive-item.json'))
    full_plan = read(ASTRA/'plan.json')
    full_plan['questions'] = QUESTIONS
    full_plan['reference_sha256'] = digest(root/'private-reference.json')
    full_plan['base_instructions'] = full_plan['base_instructions'].replace('fifteen questions','five questions')
    schema = full_plan['output_schema']['properties']['answers']
    schema['minItems'] = schema['maxItems'] = len(QUESTIONS)
    schema['items']['properties']['id']['enum'] = [q['id'] for q in QUESTIONS]
    full_plan['protocol'] = 'Same exact archive once, five new questions jointly, no tools or compaction.'
    full_plan['admissibility'] = 'Complete archive acknowledged; no tools or compaction; full input usage; all five answers.'
    freeze(Path('control')/'plan.json', full_plan)
    implementations = [__file__, 'tools/evaluate_dialogue_index_dump.py',
        'tools/evaluate_dialogue_astra_appserver.py', 'tools/evaluate_dialogue_full_context_batch.py',
        'src/memory_condense/application/dialogue_retrieval.py',
        'src/memory_condense/search/dialogue_neighborhood.py']
    save(root/'plan.json', dict(model=MODEL, gateway=GATEWAY, arms=ARMS,
        reader_system=ACTOR_SYSTEM+'\n'+INDEX_NOTE, neighborhood_system=ACTOR_SYSTEM,
        judge_system=JUDGE.replace('three blinded answers','six blinded answers'),
        max_concurrency=3, max_output_tokens=1000, judge_output_tokens=2600,
        prompt_limit=60000, judge_prompt_limit=14000, timeout_s=180, automatic_retries=0,
        frozen=frozen, implementation={p:digest(p) for p in implementations},
        questions=5, checkpoint_occurrences=sum(map(len, CHECKPOINTS.values())),
        sol_reader_calls=25, astra_reader_calls=1, judge_calls=5,
        new_summaries=0, history_reingestions=0, fresh_query_embeddings=5,
        labels='All index and raw turn labels use the same zero-based convention.',
        limitations=[
            'Five new questions on the same development archive and existing reference facts; not independent histories or a held-out corpus.',
            'Checkpoint coverage, not general QA accuracy or exhaustive factual correctness.',
            'Astra medium answers jointly; Sol none answers independently. Different models and call structures.',
            'Sol grades all six arms jointly under shuffled labels; no independent human scoring.',
            'Original 15 scores remain fixed; this extension corrects their index range display offset.',
            'Aggregate 20-question scores combine two rounds and must not be read as a homogeneous rerun.',
        ]))
    validate(root)
    emit(phase='prepared', questions=5, answer_calls=26, judge_calls=5,
        points_per_arm=2*sum(map(len,CHECKPOINTS.values())))


def generate(root):
    plan = validate(root)
    def answer(job):
        q, arm = job
        packet = read(root/'packets'/q['id']/(arm+'.json'))
        system = plan['neighborhood_system'] if arm == 'closing_6k' else plan['reader_system']
        messages = [dict(role='system',content=system),dict(role='user',
            content='Archived evidence:\n'+packet['text']+'\n\nQuestion: '+q['question'])]
        result = api_job(root, 'answer-'+q['id']+'-'+arm, messages, plan=plan)
        emit(phase='answered', question=q['id'], arm=arm, elapsed_s=result['elapsed_s'])
    bounded([(q,a) for q in read(root/'questions.json')['questions'] for a in ARMS[1:]], answer)
    validate(root)


def control(root):
    from tools.evaluate_dialogue_astra_appserver import run
    validate(root)
    if not (root/'control'/'reader-result.json').exists():
        run(root/'control')
    generation = read(root/'control'/'generation.json')
    result = read(root/'control'/'reader-result.json')
    injection = read(root/'control'/'injection-receipt.json')
    usage = generation['token_updates'][-1]['tokenUsage']['last']
    archive = read(root/'control'/'archive-item.json')['content'][0]['text']
    if (generation['compaction_observed'] or generation['turn']['status'] != 'completed'
            or not injection['acknowledged'] or set(generation['item_types'])-{'agentMessage','reasoning','userMessage'}
            or usage['inputTokens'] < .95*count_tokens(archive)):
        raise ValueError('Full-context control not admitted')
    for answer in result['answers']:
        save(root/'api'/('answer-'+answer['id']+'-full_context_batch')/'response.json',
            dict(content=answer['content'], model='gpt-6-astra', shared_generation=True,
                generation_sha256=digest(root/'control'/'generation.json')))
    save(root/'control'/'admissibility.json', dict(full_raw_context_retained=True,
        compaction=False, tools=False, usage=usage, elapsed_s=generation['elapsed_s'],
        archive_sha256=digest(root/'control'/'archive-item.json'),
        generation_sha256=digest(root/'control'/'generation.json')))
    validate(root)
    emit(phase='control_admitted', questions=len(result['answers']), usage=usage)


def grade(root):
    plan = validate(root)
    questions = read(root/'questions.json')['questions']
    reference = read(root/'private-reference.json')
    paths = [Path('api')/('answer-'+q['id']+'-'+a)/'response.json' for q in questions for a in ARMS]
    save(root/'grading-plan.json', dict(answers={str(p):digest(root/p) for p in paths},
        reference_sha256=digest(root/'private-reference.json')))
    def judge(q):
        qid = q['id']
        order = sorted(ARMS, key=lambda a:hashlib.sha256((qid+a).encode()).hexdigest())
        labels = dict(zip('ABCDEF',order,strict=True))
        required = reference['questions'][qid]
        checkpoints = {fid:dict(claim=reference['facts'][fid]['claim'],
            sources=[dict(label=s['label'],role=s['role'],quote=s['quote'][:450])
                for s in reference['facts'][fid]['supports']]) for fid in required}
        answers = {label:read(root/'api'/('answer-'+qid+'-'+arm)/'response.json')['content'] for label,arm in labels.items()}
        messages = [dict(role='system',content=plan['judge_system']),dict(role='user',content=json.dumps(
            dict(question=q['question'],checkpoints=checkpoints,answers=answers),ensure_ascii=False))]
        response = api_job(root,'judge-'+qid,messages,plan=plan,judge=True)
        parsed = extract_json(response['content'])
        scored = parse_grade(parsed,labels,required)
        save(root/'grades'/(qid+'.json'),dict(question=qid,arms=scored,labels=labels,comparison=parsed['comparison']))
        emit(phase='graded',question=qid,points={a:s['points'] for a,s in scored.items()})
    bounded(questions,judge)
    report(root)


def report(root):
    plan = validate(root)
    for filename, sha in read(root/'grading-plan.json')['answers'].items():
        if digest(root/filename) != sha:
            raise ValueError('Graded answer changed')
    questions = read(root/'questions.json')['questions']
    grades = {q['id']:read(root/'grades'/(q['id']+'.json')) for q in questions}
    control_audit = read(root/'control'/'admissibility.json')
    totals, rows = {}, []
    for arm in ARMS:
        points = sum(g['arms'][arm]['points'] for g in grades.values())
        possible = sum(g['arms'][arm]['possible'] for g in grades.values())
        responses = [read(root/'api'/('answer-'+q['id']+'-'+arm)/'response.json') for q in questions]
        invalid = []
        for q, answer in zip(questions,responses,strict=True):
            allowed = (read(root/'full-citation-labels.json')['labels'] if arm == ARMS[0]
                else read(root/'packets'/q['id']/(arm+'.json'))['allowed_citations'])
            missing = sorted(cited_labels(answer['content'])-set(allowed))
            if missing:
                invalid.append(dict(question=q['id'],labels=missing))
        totals[arm] = dict(points=points,possible=possible,checkpoint_score_pct=100*points/possible,
            unprovided_citations=invalid, answers_with_reference_contradictions=sum(bool(g['arms'][arm]['contradictions']) for g in grades.values()))
        if arm == ARMS[0]:
            usage = control_audit['usage']
            totals[arm].update(prompt_tokens=usage['inputTokens'],completion_tokens=usage['outputTokens'],
                model_calls=1,reader_model='gpt-6-astra',shared_batch_elapsed_s=control_audit['elapsed_s'])
        else:
            totals[arm].update(prompt_tokens=sum(r['usage']['prompt_tokens'] for r in responses),
                completion_tokens=sum(r['usage']['completion_tokens'] for r in responses),
                model_calls=len(questions),reader_model=MODEL,mean_reader_s=statistics.mean(r['elapsed_s'] for r in responses))
    for q in questions:
        records = grades[q['id']]['arms']
        rows.append(dict(**q,possible=records[ARMS[0]]['possible'],points={a:records[a]['points'] for a in ARMS}))
    previous = read(root/'previous-report.json')
    aggregate = {}
    for arm in ARMS:
        old, new = previous['totals'][arm], totals[arm]
        points, possible = old['points']+new['points'], old['possible']+new['possible']
        aggregate[arm] = dict(points=points,possible=possible,checkpoint_score_pct=100*points/possible,
            prompt_tokens=old['prompt_tokens']+new['prompt_tokens'], model_calls=old['model_calls']+new['model_calls'])
    comparisons = {}
    for arm in ARMS[1:]:
        differences = [r['points'][arm]-r['points'][ARMS[0]] for r in rows]
        comparisons[arm] = dict(wins=sum(d>0 for d in differences),ties=sum(d==0 for d in differences),
            losses=sum(d<0 for d in differences),difference_pp=totals[arm]['checkpoint_score_pct']-totals[ARMS[0]]['checkpoint_score_pct'])
    save(root/'report.json',dict(questions=len(questions),totals=totals,rows=rows,
        aggregate_questions=previous['questions']+len(questions), aggregate_totals=aggregate,
        comparisons=comparisons,limitations=plan['limitations'],source_unchanged=True,
        old_grades_reused=True,scoring='Five new blinded six-arm grades; old 15 grades unchanged.'))
    emit(phase='report',totals=totals,aggregate_totals=aggregate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','generate','control','grade','report'))
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    globals()[args.phase](args.root)
