"""Frozen query-intent examples; labels do not claim answer quality.

Positive: reconstruct decisions, fixes, attempts, outcomes or their revisions.
Negative: isolated value lookups, new work, or unrelated broad retrospectives.
Context pairs stay in one partition. No archive evidence is classifier input.
"""

TRAIN_YES = '''
What have we tried so far to fix ingestion?
Which decisions did we reverse and why?
Trace the fixes we attempted for the timeout.
What did we decide about deployment, and what evidence supported it?
Which earlier problems were fixed and which remain open?
Why did we abandon the first implementation?
List the failed approaches before the solution that worked.
Show the evidence behind our decision to change models.
What did we declare solved that later failed again?
Which fixes actually passed tests rather than just being proposed?
Reconstruct the history of our caching decisions.
How did the architecture change after each experiment?
Which experiments caused us to revise our approach?
What is the current plan and which earlier plan did it replace?
Track the bug from report through attempted fixes and retests.
What remained unresolved after our last repair attempt?
Explain why we switched away from batching.
What did we choose for storage and why did we reject the alternatives?
Which performance optimizations worked, with the measurements that support them?
Connect the latency problems to the fixes and their measured outcomes.
Which implementation claims were later corrected?
How did the proposed release turn into the tested release?
What mistakes did we make and how did we correct them?
Build a timeline of the decisions we made about the proxy.
Which fixes were only workarounds?
Have we tried this approach before and what happened?
Summarize the design choices and their rationale.
Which proposed features were implemented, verified or abandoned?
What evidence made us change our mind about the database?
What changed after we discovered the regression?
Explain the sequence of debugging hypotheses and what disproved each.
Which fixes should we keep based on earlier tests?
What is the final supported decision after the conflicting recommendations?
Did the later run confirm or overturn the initial conclusion?
What did we do to address the complaints and did it help?
What patches have we applied to this defect?
Which rollout strategy did we settle on after trying the alternatives?
Show the changes from the original decision to the replacement.
What was tried to resolve the missing evidence problem?
What open issues remain after accounting for subsequent fixes?
'''

TRAIN_NO = '''
What port does the proxy use?
Which reader model is configured?
What was the final test count?
Where is the configuration file?
When did the release happen?
Who owns the migration task?
What exact command runs the server?
How many benchmarks did we run?
What was the error message?
What is the current timeout value?
Write a patch to fix this exception.
Implement the proposed caching strategy.
Run the test again.
Commit the fix.
Explain how database indexes work.
Recommend a new deployment strategy.
How would you fix this function?
Compare these two code snippets.
What is a revision history?
Show me how to annotate evidence in Python.
List all cities I visited.
What topics did I mention in the workshop?
Which books did I buy?
What preferences have I expressed across our chats?
What themes connect our conversations?
What is my overall communication style?
Summarize the article I pasted.
What patterns recur in my travel plans?
How have my interests evolved?
What did my sister say about the concert?
Which concerns did I express about the contract?
What arguments did I mention about public transit?
What was the purpose of that meeting?
Is this working?
What now?
Continue.
What changed?
Was that it?
Trace this function for the supplied input.
Design a system that stores decisions and fixes.
'''

DEV_YES = '''
Which attempted repairs resolved the authentication failures?
Why was the migration plan revised?
What led us to choose the replacement architecture?
Which reported successes were contradicted by later evidence?
Trace our changes to the retry policy and the reasons for them.
Summarize what we tried to improve recall and how each trial performed.
Did we ever resolve the race condition or just work around it?
Which decision is still in force after the later correction?
What alternatives were tested before we selected the current scheduler?
What was the outcome of our earlier effort to fix the leak?
Which claims of completion were actually backed by verification?
Reconstruct the decisions that led to the production configuration.
'''
DEV_NO = '''
Which version of Python did we install?
How much memory did the model use?
What day did the deployment start?
What does the retry setting mean?
Fix the race condition now.
Please implement a migration checker.
What did I say about the apartment layout?
List my professional interests across these sessions.
What is the overall mood of this dialogue?
What topics did our research cover?
How does a compiler resolve symbols?
Write a chronological history of computing.
What should the new product support?
Should I worry?
What about now?
What is the identifier of the merged patch?
'''
TEST_YES = '''
How did we settle on the backup strategy after the unsuccessful restores?
Walk me through the attempts to repair the sensor and their outcomes.
Which conclusion about the research protocol survived the replication?
What changed in our hiring decision after the reference checks?
Which plumbing remedies were attempted before the leak stopped?
Explain why the team reversed its choice of supplier.
Was the payment outage actually fixed or merely declared fixed?
Compare the original recommendation with the correction that replaced it.
What support do we have that the accessibility changes addressed the complaint?
Recover the reasoning and evidence for the final data-retention decision.
Which attempted remedies for flaky browser tests failed, and what finally helped?
What unresolved problems are left from the sequence of upgrade attempts?
Which workaround did we later remove, and what replaced it?
Which trial results changed the experimental design?
Have we already attempted a rollback for this problem, and with what result?
Separate what we promised to fix from what we demonstrated was fixed.
'''
TEST_NO = '''
What is the backup bucket name?
Which sensor serial number did I give you?
What sample size was used in the replication?
Who did we hire?
What supplier email address did I mention?
How many browser tests passed yesterday?
What was the retention period we selected?
What time did the outage end?
Repair the payment webhook handler.
Create a workflow for reviewing procurement decisions.
Explain the causes of flaky browser tests generally.
What do our discussions reveal about my learning style?
Summarize the recurring themes in our literature reading.
List the dietary preferences I mentioned during these conversations.
What cities did the research team visit?
What are the tradeoffs of this proposed design?
Show the git command for listing revisions.
How do I attach annotations to a PDF?
Is that better?
Can you check it?
'''

PAIRS = {
    'train': [('Did that work?', 'We are tracing several earlier attempts to repair the upload failure.',
               'The user just supplied a single arithmetic expression and wants its result.')],
    'dev': [('Why the change?', 'The team reversed an earlier design decision after a failed benchmark.',
             'A tutorial is explaining why this line increments a loop variable.')],
    'test': [('What happened next?', 'We are reconstructing the attempted fixes for the login outage and their outcomes.',
              'We are discussing the plot of the novel the user just pasted.'),
             ('Was it resolved?', 'The user is reviewing the successive repairs and conflicting completion reports for a defect.',
              'The user asks whether the single DNS lookup shown in the last tool output succeeded.')],
}


def cases(split):
    rows = []
    for label, suffix in ((1, 'YES'), (0, 'NO')):
        for line in globals()[split.upper()+'_'+suffix].strip().splitlines():
            rows.append(dict(query=line, recent=[], label=label))
    for query, yes, no in PAIRS[split]:
        for label, context in ((1, yes), (0, no)):
            rows.append(dict(query=query, recent=[context], label=label))
    return [dict(row, id=f'{split}-{i:03}') for i, row in enumerate(rows)]
