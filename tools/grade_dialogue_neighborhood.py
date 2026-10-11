"""Blind paired checkpoint scoring for the frozen dialogue-neighborhood probe."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
import statistics

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.artifacts import read,save,emit
from tools.evaluate_dialogue_neighborhood import api_job,digest,DEFAULT_ROOT

ARMS=('cap8_2k','cap8_6k','neighborhood_6k')
JUDGE='''Evaluate three blinded answers to the SAME question about an archived engineering dialogue.
Score ONLY the supplied independently authored reference checkpoints. They were
frozen before generation and include exact source excerpts. Do not reward fluency
or presume an answer has seen the full archive. Historical assistant assertions
are claims, not independent proof of implementation or execution. Do not import
knowledge of this project from outside the excerpts.
For each checkpoint: 2 = its substantive point is correctly conveyed; 1 = useful
but incomplete coverage; 0 = absent or incorrect. Equivalent wording is enough;
do not require every incidental number or name. For multi-part checkpoints, full
credit requires the main relationship, not a passing mention of one component.
Acknowledging missing evidence is appropriate but does not earn coverage of a
positive fact the answer never provides. A correct cautious conclusion can earn
full credit for an uncertainty checkpoint. Do not mark extra details wrong merely
because the reference is selective. List only material contradictions with the
supplied reference, not omissions. scope_honesty is false only for unsupported
claims of exhaustive coverage or treating proposal/test/commit/release as equivalent.
Return only JSON: {"answers":[{"label":"A","checkpoints":{"F01":0},
"contradictions":[],"scope_honesty":true}],"comparison":"brief evidence-based comparison"}.
Include every label and every requested checkpoint once, using integer scores 0/1/2.'''


def extract_json(text):
    value=text.strip()
    if value.startswith('```') and value.endswith('```'):
        value=value.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(value)


def citation_labels(text):
    """Count explicit references, not prose mentioning the archive cutoff T480."""
    groups=re.findall(r'\[(T\d+(?:\s*[,;]\s*T\d+)*)\]',text)
    return {label for group in groups for label in re.findall(r'T\d+',group)}


def load_packet(root, qid, arm):
    # Full-context evidence is identical for all questions and stored once.
    if arm in ('full_context', 'full_context_batch'):
        return read(root/'full-context.json')
    return read(root/'packets'/qid/(arm+'.json'))


def job(root,q,reference,*,arms=ARMS,grade_dir='grades',key_prefix='judge-'):
    qid=q['id']
    # Alternate/permuted labels conceal route names and avoid fixed-position bias.
    order=sorted(arms,key=lambda arm:hashlib.sha256((qid+arm).encode()).hexdigest())
    labels=dict(zip('ABC',order))
    required=reference['questions'][qid]
    answers={label:read(root/'api'/('answer-'+qid+'-'+arm)/'response.json')['content']
             for label,arm in labels.items()}
    for excerpt_chars in (450,300,160):
        checkpoints={fid:dict(claim=reference['facts'][fid]['claim'],
            sources=[dict(label=s['label'],role=s['role'],quote=s['quote'][:excerpt_chars])
                     for s in reference['facts'][fid]['supports']]) for fid in required}
        messages=[dict(role='system',content=JUDGE),dict(role='user',content=json.dumps(
            dict(question=q['question'],checkpoints=checkpoints,answers=answers),ensure_ascii=False))]
        if sum(count_tokens(m['content']) for m in messages)<=7168: break
    result=api_job(root,key_prefix+qid,messages,max_tokens=1300)
    grade=extract_json(result['content'])
    if set(r['label'] for r in grade['answers'])!=set(labels) or len(grade['answers'])!=3:
        raise ValueError('Judge omitted an answer')
    scored={}
    for r in grade['answers']:
        if set(r['checkpoints'])!=set(required) or any(type(s) is not int or s not in (0,1,2) for s in r['checkpoints'].values()):
            raise ValueError('Judge changed the checkpoint population')
        if type(r.get('scope_honesty')) is not bool or type(r.get('contradictions')) is not list:
            raise ValueError('Judge omitted scope/contradiction fields')
        arm=labels[r['label']]
        answer=answers[r['label']]
        packet=load_packet(root,qid,arm)
        present={s['label'] for s in packet['spans']}
        citations=citation_labels(answer)
        scored[arm]=dict(**r,points=sum(r['checkpoints'].values()),possible=2*len(required),
            citation_labels=sorted(citations),unprovided_citations=sorted(citations-present))
    save(root/grade_dir/(qid+'.json'),dict(question=qid,labels=labels,arms=scored,comparison=grade['comparison'],
        reference_sha256=digest(root/'private-reference.json'),judge_response_sha256=digest(root/'api'/(key_prefix+qid)/'response.json')))
    return qid,scored


def report(root,*,arms=ARMS,grade_dir='grades',report_name='report.json',primary='neighborhood_6k'):
    questions=read(root/'questions.json')['questions']
    grades={q['id']:read(root/grade_dir/(q['id']+'.json')) for q in questions}
    reference=read(root/'private-reference.json')
    rows=[]; totals={}
    for arm in arms:
        records=[g['arms'][arm] for g in grades.values()]
        packets=[load_packet(root,q['id'],arm) for q in questions]
        answers=[read(root/'api'/('answer-'+q['id']+'-'+arm)/'response.json') for q in questions]
        # Recompute this deterministic check even for historical judge receipts.
        # The model's checkpoint scores and comments remain untouched.
        for record,answer,packet in zip(records,answers,packets,strict=True):
            cited=citation_labels(answer['content'])
            present={s['label'] for s in packet['spans']}
            record['citation_labels']=sorted(cited)
            record['unprovided_citations']=sorted(cited-present)
        usage=[a.get('usage') or {} for a in answers]
        points=sum(r['points'] for r in records); possible=sum(r['possible'] for r in records)
        totals[arm]=dict(points=points,possible=possible,checkpoint_score_pct=100*points/possible,
            answers_with_reference_contradictions=sum(bool(r['contradictions']) for r in records),
            answers_with_unprovided_citations=sum(bool(r['unprovided_citations']) for r in records),
            answers_with_citations=sum(bool(r['citation_labels']) for r in records),
            scope_honest=sum(r['scope_honesty'] for r in records),
            mean_evidence_tokens=statistics.mean(p['sent_evidence_tokens'] for p in packets),
            mean_reader_s=statistics.mean(a['elapsed_s'] for a in answers),
            mean_hydration_s=statistics.mean(p['hydration_s'] for p in packets),
            prompt_tokens=sum(u.get('prompt_tokens',0) for u in usage),
            completion_tokens=sum(u.get('completion_tokens',0) for u in usage),
            reader_models=sorted({a.get('response_model','unknown') for a in answers}),
            source_roles=dict(Counter(s['span']['role'] for p in packets for s in p['spans'])))
        if arm == 'full_context_batch':
            batch=read(root/'batch/response.json')
            usage=batch.get('usage') or {}
            totals[arm].update(prompt_tokens=usage.get('prompt_tokens'),
                completion_tokens=usage.get('completion_tokens'),
                mean_reader_s=None,shared_batch_elapsed_s=batch['elapsed_s'],
                model_calls=1,timing_scope='All fifteen answers generated in one shared-context request.')
    for q in questions:
        scores={a:grades[q['id']]['arms'][a]['points'] for a in arms}
        rows.append(dict(id=q['id'],question=q['question'],points=scores,
            possible=2*len(reference['questions'][q['id']]),comparison=grades[q['id']]['comparison']))
    comparisons={}
    for control in (a for a in arms if a!=primary):
        diffs=[r['points'][primary]-r['points'][control] for r in rows]
        comparisons[control]=dict(wins=sum(v>0 for v in diffs),ties=sum(v==0 for v in diffs),
            losses=sum(v<0 for v in diffs),score_difference_pp=totals[primary]['checkpoint_score_pct']-totals[control]['checkpoint_score_pct'])
    source=read(root/'source.json')
    unchanged=all(digest(Path(source['directory'])/name)==sha for name,sha in source['files'].items())
    if not unchanged: raise ValueError('Original memory changed during experiment')
    result=dict(source=source['snapshot'],questions=len(questions),primary=primary,totals=totals,comparisons=comparisons,
        rows=rows,original_memory_unchanged=unchanged,
        citation_validation='Explicit bracketed source references checked against each actual packet; archive-cutoff mentions are not citations.',
        limitations=['One archived engineering memory; 15 correlated retrospective questions.',
            'Selective manually authored checkpoints, not exhaustive gold facts across all 776K tokens.',
            'Same Sol model generates and grades; labels are blinded, not independent adjudication.',
            'Prototype surveys summary neighborhoods and balances roles; no reader-directed follow-up loop or Hebbian traversal.',
            'Memory arms use a 6K maximum with different realized lengths; the full-context arm, if present, has no retrieval budget.',
            'Question 1 used ambiguous working-against-goals wording in every arm; checkpoint coverage, not a binary yes/no, is scored.',
            'Closing-response variant, when present, is a development follow-up on the same questions after inspecting initial misses.',
            'Read-only retrieval/answer comparison, not a continuous ingestion or live-proxy latency test.'])
    context_path=root/'comparison-context.json'
    if context_path.exists():
        result['comparison_context']=read(context_path)
        result['limitations'].extend(result['comparison_context'].get('limitations',[]))
    save(root/report_name,result)
    emit(phase='report',totals=totals,comparisons=comparisons)


def main(root,*,arms=ARMS,grade_dir='grades',key_prefix='judge-',report_name='report.json',primary='neighborhood_6k'):
    reference=read(root/'private-reference.json')
    plan=read(root/'run-plan.json')
    if digest(root/'private-reference.json')!=plan['reference_sha256']:
        raise ValueError('Reference changed after generation began')
    questions=read(root/'questions.json')['questions']
    paths=[root/'api'/('answer-'+q['id']+'-'+arm)/'response.json' for q in questions for arm in arms]
    if not all(p.exists() for p in paths): raise ValueError('All matched answers are required before grading')
    bindings={str(p.relative_to(root)):digest(p) for p in paths}
    if len(bindings)!=45: raise ValueError('All 45 matched answers are required before grading')
    plan_name='grading-plan.json' if key_prefix=='judge-' else key_prefix+'plan.json'
    save(root/plan_name,dict(system=JUDGE,answers=bindings,reference_sha256=plan['reference_sha256'],
        implementation_sha256=digest(__file__),score='0 absent/incorrect, 1 partial, 2 substantive coverage per required checkpoint',
        primary_comparison=primary+' versus cap8_6k',max_concurrency=3,judge_calls=15))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job,root,q,reference,arms=arms,grade_dir=grade_dir,key_prefix=key_prefix) for q in questions]
        for future in as_completed(futures):
            qid,scored=future.result()
            emit(phase='graded',question=qid,points={a:r['points'] for a,r in scored.items()})
    report(root,arms=arms,grade_dir=grade_dir,report_name=report_name,primary=primary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    comparison=parser.add_mutually_exclusive_group()
    comparison.add_argument('--closing',action='store_true')
    comparison.add_argument('--full',action='store_true')
    args=parser.parse_args()
    if args.full:
        main(args.root,arms=('cap8_6k','closing_6k','full_context_batch'),grade_dir='full-grades',
             key_prefix='full-judge-',report_name='full-report.json',primary='closing_6k')
    elif args.closing:
        main(args.root,arms=('cap8_6k','neighborhood_6k','closing_6k'),grade_dir='closing-grades',
             key_prefix='closing-judge-',report_name='closing-report.json',primary='closing_6k')
    else:
        main(args.root)
