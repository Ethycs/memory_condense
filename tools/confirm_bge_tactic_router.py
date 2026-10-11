"""Second-stage BGE scope confirmation after the first real-query domain shift.

The exposed first 100 real queries become 70 training/30 development examples.
Confirmation uses different source-history questions and new authored scenarios.
Original held-out scores are preserved and never relabeled as fresh validation.
"""
from collections import Counter
from pathlib import Path

from memory_condense.application.tactic_router import render_input
from memory_condense.runtime.artifacts import read,save,emit
from tools.evaluate_dialogue_neighborhood import digest

ROOT = Path('eval_results/bge-tactic-router-20261007-r2')
FIRST = Path('eval_results/bge-tactic-router-20261007-r1')

FOCUSED = [
    'What opinion did I express about commuter rail funding?',
    'What concerns did I mention about adopting a rescue cat?',
    'Which ideas did I ask the workshop facilitator to explain?',
    'What expectations did I state for my first management role?',
    'What arguments did I give for choosing the electric car?',
    'Which challenges did I describe in learning to paint portraits?',
    'What did I decide to try after the gardening advice?',
    'What topics did I request for the environmental newsletter?',
    'What did I say my priorities were for furnishing the apartment?',
    'How did I describe the appeal of the history lecture?',
    'What beliefs about volunteering did I mention in that discussion?',
    'Which factors did I say mattered for choosing a summer camp?',
    'What reservations did I voice about the proposed contract?',
    'What plans did I mention for continuing my language practice?',
    'What did I say about the role of art in education?',
    'What criteria did I ask you to include in the vendor comparison?',
]
OVERVIEW = [
    'Have we become better at separating hypotheses from established results during this engagement?',
    'Which disagreements throughout the dialogue ultimately changed our priorities?',
    'Explain why this effort took longer than expected using the sequence of setbacks.',
    'What consistent weaknesses emerge from the way we tested our successive designs?',
    'Reconstruct how the accepted solution emerged from competing proposals.',
    'Which unresolved themes have followed us from the first discussion to now?',
    'What do the changes in my feedback reveal about my underlying priorities?',
    'Assess the cumulative value of our work after accounting for failed experiments.',
    'Where did we repeatedly confuse a successful demonstration with completed delivery?',
    'What earlier findings should constrain the direction we take from here?',
    'How have our explanations for poor performance changed over this investigation?',
    'Connect the major compromises we made to the evidence available at each stage.',
    'What is the overall pattern in how we handled corrections and contradictory evidence?',
    'Which repeated requests from me have remained insufficiently addressed?',
    'What conclusions about our working process can you draw from the full dialogue?',
    'If someone inherited this project, what would they misunderstand without its history?',
]
UNCERTAIN = ['Is it right?', 'What next then?', 'Does that change things?',
             'Should I worry?', 'Was that the issue?', 'What about now?']
PAIRS = [
    ('What should we conclude?',
     'A single unit test returns the wrong integer. The user asks what that output means.',
     'Several competing theories were investigated over many sessions. The user wants an assessment of the accumulated evidence.'),
    ('Have we made progress?',
     'The current upload was at 40 percent and is now at 65 percent. We are checking this transfer.',
     'The project has had many iterations, reversals and partial fixes. The user is evaluating whether its original goals are closer.'),
    ('Does that explain it?',
     'A function divides by zero for one supplied input. The last message identifies the denominator.',
     'A retrospective is relating recurring project failures to earlier design and evaluation decisions across the session.'),
    ('What can we take away?',
     'The last log entry shows that the access token expired. We are diagnosing only that request.',
     'The research programme has concluded after several dead ends and successful replications. The user wants lessons across the programme.'),
]


def prepare(root=ROOT):
    root.mkdir(parents=True,exist_ok=False)
    train,dev = [read(FIRST/(split+'.json'))['cases'] for split in ('train','dev')]
    previous_real = read(FIRST/'real_factual.json')['cases']
    # Fixed every-tenth split: last three are development, no outcome-dependent selection.
    for i,row in enumerate(previous_real):
        split = 'dev' if i%10>=7 else 'train'
        value = dict(row,split=split,source='r1 exposed real-query development')
        (dev if split=='dev' else train).append(value)
    confirmation = []
    for label,queries in [('focused',FOCUSED),('overview',OVERVIEW),('uncertain',UNCERTAIN)]:
        for i,q in enumerate(queries):
            confirmation.append(dict(id=f'confirm-{label}-{i:02}',split='test',label=label,
                query=q,recent=[],group=f'confirm-{label}-{i:02}',source='new authored confirmation'))
    for i,(q,f,o) in enumerate(PAIRS):
        for label,context in [('focused',f),('overview',o)]:
            confirmation.append(dict(id=f'confirm-pair-{i:02}-{label}',split='test',label=label,
                query=q,recent=[context],group=f'confirm-pair-{i:02}',source='new authored context pair'))
    source = Path('eval_results/scale-5m-20261003-r2/run-plan.json')
    old_texts = {r['query'] for r in train+dev}
    real = []
    for i,q in enumerate(read(source)['questions']):
        if i%5 not in (1,2):
            continue
        text = q['retrieval_query']
        if text in old_texts:
            raise ValueError('New real confirmation query duplicates development')
        real.append(dict(id='real-fact-'+q['question_id'],split='real_factual',label='focused',
            query=text,recent=[],group=q['question_id'],source=str(source)))
    known = read(FIRST/'known_overview.json')['cases']
    rows = train+dev+confirmation+real+known
    inputs = [render_input(r['query'],r['recent'])[0] for r in rows]
    if len(set(inputs))!=len(inputs):
        raise ValueError('Duplicate complete classifier inputs')
    frozen = {}
    for split in ('train','dev','test','real_factual','known_overview'):
        name = split+'.json'
        save(root/name,dict(cases=[r for r in rows if r['split']==split]))
        frozen[name] = digest(root/name)
    save(root/'inputs.json',dict(cases=[{k:r[k] for k in ('id','query','recent')} for r in rows]))
    frozen['inputs.json'] = digest(root/'inputs.json')
    save(root/'previous-report.json',read(FIRST/'report.json'))
    frozen['previous-report.json'] = digest(root/'previous-report.json')
    previous_plan = read(FIRST/'plan.json')
    paths = [__file__,*previous_plan['implementation']]
    plan = dict(previous_plan,frozen=frozen,implementation={p:digest(p) for p in paths},
        counts=dict(Counter(r['split'] for r in rows)),
        adaptation='First 100 factual queries exposed a domain shift; now 70 train and 30 dev. Original first-stage results remain unchanged. New history-2/history-3 queries and authored scenarios are confirmation.',
        limitations=previous_plan['limitations']+[
            'Second-stage confirmation follows inspection of first-stage errors. Its new test labels were frozen before second-stage fitting.',
            'Same source question-generation pipeline across real-query histories; this is not an independent external dataset.',
        ])
    save(root/'plan.json',plan)
    emit(phase='confirmation_prepared',counts=plan['counts'],total=len(rows))


if __name__ == '__main__':
    prepare()
