"""Apply a documented human audit to the sealed experimental judge output."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from memory_condense.runtime.artifacts import read, save
from tools.characterize_native_experiments import ARMS, digest, validate


# Only clear point errors are changed. Each rationale states the rubric mismatch.
OVERRIDES = {
    ('Q05','qwen_associations','F21'): (1,
        'The answer omits evaluation without reheating. The same five-of-six coverage was scored partial on L03.'),
    ('Q05','qwen_summary_routing','F21'): (1,
        'The answer reports the reheat leak and later correction but omits the outstanding commit; the multi-part checkpoint is incomplete.'),
    ('Q15','cav_associations','F04'): (1,
        'The answer links supersession fixes to signed commits and distinguishes later uncommitted decay, but never identifies the allocator fix.'),
}


def main(root):
    plan=validate(root)
    raw=read(root/'answer-report.json')
    grades={q['id']:read(root/'grades'/(q['id']+'.json')) for q in plan['questions']}
    changes=[]
    for (qid,arm,fid),(new,rationale) in OVERRIDES.items():
        row=grades[qid]['arms'][arm]
        old=row['checkpoints'][fid]
        if old==new:
            raise ValueError('Adjudication no longer changes the frozen grade')
        row['checkpoints'][fid]=new
        row['points']=sum(row['checkpoints'].values())
        changes.append(dict(question=qid,arm=arm,checkpoint=fid,raw=old,
                            adjudicated=new,rationale=rationale))
    totals={}
    categories={}
    for category in ['all',*dict.fromkeys(q['category'] for q in plan['questions'])]:
        questions=[q for q in plan['questions'] if category=='all' or q['category']==category]
        values={}
        for arm in ARMS:
            scores=[grades[q['id']]['arms'][arm] for q in questions]
            base=[grades[q['id']]['arms']['baseline'] for q in questions]
            deltas=[a['points']-b['points'] for a,b in zip(scores,base)]
            values[arm]=dict(
                questions=len(questions),
                macro_checkpoint_pct=100*mean(s['points']/s['possible'] for s in scores),
                points=sum(s['points'] for s in scores),
                possible=sum(s['possible'] for s in scores),
                wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),
                losses=sum(d<0 for d in deltas))
        if category=='all': totals=values
        else: categories[category]=values
    unchanged=[]
    for q in plan['questions']:
        for arm in ARMS:
            for fid,value in grades[q['id']]['arms'][arm]['checkpoints'].items():
                if (q['id'],arm,fid) not in OVERRIDES:
                    unchanged.append((q['id'],arm,fid,value))
    diagnostic_review=[
        dict(question='L01',finding='confirmed_non_score_error',
             detail='Most answers mislabel ingested-haystack presence as retrieval-stage presence, but all still convey every scored F06 element.'),
        dict(question='Q07',finding='judge_diagnostic_error',
             detail='The contradiction note conflates a pending stratified single-sample result with the separately replicated pooled-span result. No point correction follows.'),
    ]
    result=dict(status='human_semantic_adjudication',raw_report_sha256=digest(root/'answer-report.json'),
        grade_sha256={q['id']:digest(root/'grades'/(q['id']+'.json')) for q in plan['questions']},
        changes=changes,diagnostic_review=diagnostic_review,totals=totals,categories=categories,
        preserved_checkpoint_scores=len(unchanged),
        limitations=['One human semantic audit by the implementing agent, not an independent second judge.',
                     'Raw automated grades remain immutable; this file is a transparent correction layer.',
                     'Corrections address rubric consistency, not whether added passages caused answer variation.'],
        raw_totals=raw['totals'])
    save(root/'scoring-adjudication.json',result)
    print({'changes':changes,'totals':totals})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
