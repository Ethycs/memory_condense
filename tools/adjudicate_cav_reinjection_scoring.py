"""Apply the documented semantic audit to the sealed CAV-reinjection grades."""
import argparse
from pathlib import Path
from statistics import mean

from memory_condense.runtime.artifacts import read, save
from tools.characterize_cav_reinjection import ARMS, digest, validate


# Raw grades remain immutable.  These are only clear checkpoint-level errors
# established by the frozen prior audit, explicit answer text, or answer identity.
OVERRIDES = {
    ('Q03', 'baseline', 'F20'): (
        1,
        'The byte-identical frozen baseline already received an audited partial: '
        'it reports the evaluation reheating leak but omits stale documentation.',
    ),
    ('Q07', 'baseline', 'F15'): (
        1,
        'The byte-identical frozen baseline already received an audited partial '
        'for the pooled-span replication checkpoint.',
    ),
    ('Q07', 'cav_reinjection', 'F12'): (
        2,
        'The answer explicitly gives the 6.0% mixed-granularity collapse, short-text '
        'cosine domination, and the stratified-retrieval correction.',
    ),
    ('Q12', 'baseline', 'F18'): (
        2,
        'The frozen baseline answer and checkpoint were already audited at full credit.',
    ),
    ('Q12', 'baseline', 'F20'): (
        1,
        'The frozen baseline answer and checkpoint were already audited at partial credit.',
    ),
    ('Q12', 'cav_reinjection', 'F18'): (
        2,
        'Q12 binds both arms to the exact same response; its checkpoint score must match '
        'the frozen audited baseline.',
    ),
    ('Q12', 'cav_reinjection', 'F20'): (
        1,
        'Q12 binds both arms to the exact same response; its checkpoint score must match '
        'the frozen audited baseline.',
    ),
    ('Q15', 'baseline', 'F22'): (
        2,
        'The byte-identical frozen baseline distinguishes the staged roadmap from the '
        'unshipped work in its remaining-evidence section.',
    ),
}


def main(root: Path) -> None:
    plan = validate(root)
    raw = read(root / 'answer-report.json')
    prior_root = Path(plan['prior_root'])
    prior_audit = read(prior_root / 'scoring-adjudication.json')
    jobs = read(root / 'reader-jobs.json')['bindings']
    prior_jobs = read(prior_root / 'reader-jobs.json')['bindings']
    if any(jobs[qid]['baseline'] != prior_jobs[qid]['baseline'] for qid in jobs):
        raise ValueError('Current baseline does not reuse every frozen answer identity')
    if jobs['Q12']['baseline'] != jobs['Q12']['cav_reinjection']:
        raise ValueError('Q12 arms no longer bind the same answer')

    grades = {
        q['id']: read(root / 'grades' / (q['id'] + '.json'))
        for q in plan['questions']
    }
    changes = []
    for (qid, arm, fid), (new, rationale) in OVERRIDES.items():
        row = grades[qid]['arms'][arm]
        old = row['checkpoints'][fid]
        if old == new:
            raise ValueError('Adjudication no longer changes the frozen grade')
        row['checkpoints'][fid] = new
        row['points'] = sum(row['checkpoints'].values())
        changes.append(
            dict(
                question=qid,
                arm=arm,
                checkpoint=fid,
                raw=old,
                adjudicated=new,
                rationale=rationale,
            )
        )

    totals = {}
    categories = {}
    for category in ['all', *dict.fromkeys(q['category'] for q in plan['questions'])]:
        questions = [
            q for q in plan['questions']
            if category == 'all' or q['category'] == category
        ]
        values = {}
        for arm in ARMS:
            scores = [grades[q['id']]['arms'][arm] for q in questions]
            baseline = [grades[q['id']]['arms']['baseline'] for q in questions]
            deltas = [a['points'] - b['points'] for a, b in zip(scores, baseline)]
            values[arm] = dict(
                questions=len(questions),
                macro_checkpoint_pct=100 * mean(
                    score['points'] / score['possible'] for score in scores
                ),
                points=sum(score['points'] for score in scores),
                possible=sum(score['possible'] for score in scores),
                wins=sum(delta > 0 for delta in deltas),
                ties=sum(delta == 0 for delta in deltas),
                losses=sum(delta < 0 for delta in deltas),
            )
        if category == 'all':
            totals = values
        else:
            categories[category] = values

    preserved = sum(
        (q['id'], arm, fid) not in OVERRIDES
        for q in plan['questions']
        for arm in ARMS
        for fid in grades[q['id']]['arms'][arm]['checkpoints']
    )
    result = dict(
        status='human_semantic_adjudication',
        raw_report_sha256=digest(root / 'answer-report.json'),
        prior_adjudication_sha256=digest(prior_root / 'scoring-adjudication.json'),
        prior_baseline_points=prior_audit['totals']['baseline']['points'],
        reused_baseline_answer_identities=len(jobs),
        identical_arm_answer_identities=['Q12'],
        grade_sha256={
            q['id']: digest(root / 'grades' / (q['id'] + '.json'))
            for q in plan['questions']
        },
        changes=changes,
        totals=totals,
        categories=categories,
        preserved_checkpoint_scores=preserved,
        limitations=[
            'One human semantic audit by the implementing agent, not an independent second judge.',
            'Raw automated grades remain immutable; this file is a transparent correction layer.',
            'The treatment has one generation per distinct input, so generation variance is not estimated.',
            'This assay uses one archived engineering dialogue and a two-concept bank.',
        ],
        raw_totals=raw['totals'],
    )
    save(root / 'scoring-adjudication.json', result)
    print({'changes': changes, 'totals': totals})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
