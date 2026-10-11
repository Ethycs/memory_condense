"""Second confirmation after the first history head under-selected outcomes.

The first confirmation is now development data. Its report is retained. This
run selects on development balanced accuracy and freezes a new authored test.
"""
from pathlib import Path

from memory_condense.application.tactic_router import render_input
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.runtime.artifacts import read, save

YES = '''
Recall the remedies we tried for the failed backup and their results.
What evidence supported the decision to retain the old supplier?
What became of the complaint after we attempted to address it?
Which claims about fixing the outage were subsequently withdrawn?
Show the sequence of changes to our staffing decision.
Did our repair attempts actually resolve the hardware problem?
What happened after we applied the proposed remedy?
Which research recommendations changed after the new evidence?
Why did we choose the final maintenance approach?
What did the validation show about our attempted fix?
Explain the history behind our choice of testing framework.
Summarize the earlier trials and whether they supported the chosen approach.
Which decisions were tentative and which were explicitly approved?
Which observed outcomes contradict our earlier progress reports?
What did we do about the issue raised in the previous meeting?
Did we complete the promised follow-up or leave it pending?
What caused the original recommendation to be replaced?
What supported the final choice among the options we discussed?
Which earlier intervention failed to address the problem?
Trace the changing conclusions about the root cause.
'''
NO = '''
What was the name of the supplier we chose?
What staffing budget did we agree on?
Which test framework version is installed?
What does this one validation error mean?
Make a new maintenance schedule.
Choose a supplier for the new order.
What color is the replacement part?
Give me the raw output of yesterday's command.
What are the main themes of this archived correspondence?
Which topics have we researched?
List the restaurants we considered.
What dietary restrictions should you remember about me?
Please correct the spelling of this sentence.
How do I fix a broken zipper?
Design an experiment to compare these two options.
Show me the git diff for the current change.
'''
TEST_YES = '''
How did we arrive at the accepted disaster-recovery plan?
Which remedies for duplicate invoices were tried, and what happened afterward?
Recover the sequence of decisions about the experiment's exclusion criteria.
Did the follow-up measurements validate the intervention we selected?
Why did the team stop pursuing the original packaging design?
Which conclusions about the training programme were later revised?
What is still unfinished among the corrective actions we committed to?
Which proposed security changes have evidence of actual deployment?
Trace the warehouse incident from initial diagnosis to the last attempted remedy.
What led us to reverse the decision about the onboarding flow?
Which data-cleaning strategies have we already attempted unsuccessfully?
What supported the decision to keep the mechanical redesign?
Did our last attempt solve the notification problem or only mask it?
What happened to the action items after we reported the issue resolved?
Show where a later test overturned an earlier success claim.
Which contract clauses did we revise and what prompted those changes?
What decisions in the simulation setup changed because of validation results?
How did the explanation for the intermittent failure evolve during debugging?
Which rejected alternatives should we avoid retrying, given the earlier evidence?
What did we learn from the attempts to eliminate the duplicate messages?
What evidence explains the final choice of build system?
Which asserted improvements were never demonstrated by a subsequent check?
Which problems reappeared despite the earlier patches?
What was the reasoning for the architectural compromise we settled on?
'''
TEST_NO = '''
What is the invoice number?
Which exclusion threshold did we select?
What is the name of the accepted build system?
What date did the notification outage begin?
How many duplicate messages were observed?
Which warehouse was affected?
What package version is deployed?
Who approved the final contract?
What exact error did the simulator return?
What percentage did the validation report show?
Implement the onboarding change we agreed on.
Resolve the duplicate invoices now.
Suggest a new way to debug intermittent faults.
Review this patch for security problems.
Explain the difference between a hypothesis and a conclusion.
What do all my conversations suggest about my preferred writing style?
Which recurring subjects do we discuss?
Summarize the emotional tone of this diary.
List every instrument I said I could play.
How have my reading interests changed over the year?
What does it mean to supersede a contract?
Create a template for documenting attempted fixes.
Can you continue?
Is that correct?
'''


def prepare(root):
    old = Path('eval_results/history-intent-20261008-r1')
    root.mkdir(parents=True, exist_ok=False)
    train = read(old/'train.json')['cases']
    for label, block in ((1,YES),(0,NO)):
        train += [dict(query=q,recent=[],label=label) for q in block.strip().splitlines()]
    dev = read(old/'dev.json')['cases'] + read(old/'test.json')['cases']
    test = [dict(query=q,recent=[],label=label) for label,block in ((1,TEST_YES),(0,TEST_NO))
            for q in block.strip().splitlines()]
    for query,yes,no in [
        ('Did it help?', 'The user wants the outcome of earlier attempts to remedy database contention.',
         'We are checking whether the current tooltip explains the button clearly.'),
        ('Why did we do that?', 'The user wants to reconstruct a past architecture decision and the evidence behind it.',
         'We are explaining why the line shown in this sorting example swaps two array entries.')]:
        test.extend([dict(query=query,recent=[yes],label=1),dict(query=query,recent=[no],label=0)])
    seen=set()
    for split,rows in (('train',train),('dev',dev),('test',test)):
        keys=[render_input(r['query'],r['recent'])[0] for r in rows]
        if len(set(keys))!=len(keys) or seen.intersection(keys): raise ValueError('Overlapping splits')
        seen.update(keys)
        save(root/(split+'.json'),dict(cases=[dict(r,id=f'{split}-{i:03}') for i,r in enumerate(rows)]))
    save(root/'plan.json',dict(C=[.1,1.,10.,100.],thresholds=[.45,.5,.55,.6,.65,.7],
        selection='Development balanced accuracy minus 0.1 false-positive rate; then fewer false positives and smaller C',
        objective='balanced',first_confirmation_promoted_to_development=True,
        limitations='Fresh authored confirmation; same author and taxonomy. No new answer-quality claim.',
        split_hashes={s:identity_sha256(read(root/(s+'.json'))) for s in ('train','dev','test')}))


if __name__=='__main__':
    prepare(Path('eval_results/history-intent-20261008-r2'))
