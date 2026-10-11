"""Human-authored scope labels, frozen before the local classifier experiment.

Focused includes factual set/count questions and current tasks. Overview means
synthesis of the conversation's trajectory, decisions, goals, or repeated patterns.
Uncertain means the supplied text does not establish a scope. These are tactic
preferences, not claims that one retrieval policy must win answer accuracy.
"""

# Within a split, repeated short queries have opposing contexts intentionally.
# Context pairs never cross splits. Test domains and wording are held out.
TRAIN = {
    'focused': [
        'What port is the staging server listening on?',
        'Which embedding model did we choose?',
        'What date did I book the hotel for?',
        'List all the museums I visited in Madrid.',
        'How many different bicycles have I owned?',
        'What exact command ran the migration?',
        'Remind me of the default timeout value.',
        'Who owns the API key rotation task?',
        'Where is the Docker configuration file?',
        'Which restaurant did my sister recommend?',
        'What is the latest reported test count?',
        'When did I last replace the battery?',
        'What error did the last build print?',
        'Explain how a database index works.',
        'Please fix this off-by-one error.',
        'Write a unit test for the parser below.',
        'Compare these two functions in the pasted code.',
        'Summarize the article I just pasted.',
        'Show every argument accepted by this function.',
        'What happened in the failing HTTP request?',
        'How do I configure the proxy?',
        'Which of the two monitor models was cheaper?',
        'List the ingredients in the soup recipe you gave me.',
        'What was the budget for my kitchen renovation?',
        'Which cities did I visit last summer?',
        'What were the names of all the books I mentioned?',
        'What is the purpose of this one setting?',
        'Change the button text to Save.',
        'Commit the patch we just reviewed.',
        'Yes, run that command.',
        'Could you repeat the address?',
        'Which release first added that flag?',
    ],
    'overview': [
        'What have we tried so far and which attempts actually helped?',
        'How has our architecture evolved throughout this project?',
        'Compare our original goals with what we ultimately delivered.',
        'Which mistakes kept recurring over our work together?',
        'Are we ready to ship, considering all the evidence we collected?',
        'Trace the decisions we reversed and explain why.',
        'Build a timeline of our experiments and their conclusions.',
        'Which earlier problems were fixed and which remain open?',
        'What did we declare solved prematurely during this conversation?',
        'What have you learned about my preferences across our conversations?',
        'How did my career priorities change over the last year of discussions?',
        'Review our entire planning process and identify recurring obstacles.',
        'What patterns emerge across my abandoned side projects?',
        'What is the strongest evidence that our approach is working overall?',
        'Reconcile the conflicting benchmark claims made at different stages.',
        'What changed between the first prototype and our current system?',
        'Summarize the whole investigation, including failed hypotheses.',
        'Which assumptions did we repeatedly discover were wrong?',
        'How did the cost and reliability tradeoffs develop over this session?',
        'What lessons should our next project take from everything we did here?',
        'Look across all our debugging episodes for common root causes.',
        'What commitments did we make and later forget to fulfill?',
        'Explain the sequence of reasoning that led to our current design.',
        'Write a handoff that reconciles completed work with outstanding risks.',
        'Have my travel priorities remained consistent over our planning sessions?',
        'Why have we changed the research plan so many times?',
        'Which results genuinely support our overall conclusion?',
        'Audit the gap between what was proposed, implemented and validated.',
        'What broad strategy would you change after reviewing our history?',
        'What themes connect the feedback I have given you over time?',
        'Did we lose important requirements as this project changed?',
        'Explain how our understanding of the problem developed.',
    ],
    'uncertain': [
        'Is this working?', 'What now?', 'Why did that happen?', 'Better?',
        'Can you check it?', 'What do you think?', 'Is that enough?',
        'Which way?', 'What about the other one?', 'Continue.',
        'Any issues?', 'What changed?',
    ],
}
DEV = {
    'focused': [
        'What was the final score of the Tuesday test?',
        'Give me the names of every dog I have mentioned.',
        'Which branch contains the authentication fix?',
        'When did the conference start?',
        'What were all the dates of my piano lessons?',
        'Tell me what the reset parameter does.',
        'Write installation instructions for this command.',
        'Explain the traceback in this message.',
        'How many tickets did we buy for the concert?',
        'What was the address of the rental office?',
        'Compare the two measurements in this table.',
        'List every supported HTTP verb in that endpoint.',
    ],
    'overview': [
        'Across this investigation, which conclusions survived later testing?',
        'How did our objectives shift from kickoff to the current milestone?',
        'Evaluate whether our team has been making sustained progress.',
        'Which shortcuts created downstream problems repeatedly?',
        'Connect the major turning points in our planning history.',
        'Did our implementation ever catch up with the promised behavior?',
        'What common themes explain my changing plans for graduate school?',
        'Give a retrospective of every research direction we pursued.',
        'Which past decisions explain the current unfinished work?',
        'What did our unsuccessful experiments teach us collectively?',
        'Assess how well the entire session met my stated requirements.',
        'Reconcile old success reports with the later regressions.',
    ],
    'uncertain': [
        'Should we do that?', 'Any better now?', 'What is going on?',
        'Do you agree?', 'Can it?', 'And then?', 'Worth it?', 'Same problem?',
    ],
}
TEST = {
    'focused': [
        'Which camera lens did I use on the birdwatching trip?',
        'Name all the languages my brother speaks.',
        'Which commit changed the request size limit?',
        'What was the throughput in the most recent measurement?',
        'Show me the URL for the astronomy dataset.',
        'When did the insurance renewal notice arrive?',
        'What time is my pottery workshop?',
        'Which five movies did I recommend to Luis?',
        'Calculate the difference between the two prices I gave you.',
        'What is the expected output of this regular expression?',
        'Explain the purpose of the retry counter in this code.',
        'Could you implement the validation rule described above?',
        'Summarize just the attached incident report.',
        'How many distinct suppliers did I contact for the wedding?',
        'List the dates and locations of all my climbing classes.',
        'What happened when the final database statement executed?',
        'Which method in this class handles authentication?',
        'What were the two alternatives proposed for the cache key?',
        'How does the attention softmax calculation work?',
        'Run the approved regression check and report its result.',
    ],
    'overview': [
        'Which themes recur in how I approach difficult negotiations?',
        'Were the promises made early in this engagement eventually kept?',
        'Trace how our explanation of the laboratory anomaly changed with new evidence.',
        'Distinguish enduring lessons from false starts throughout our collaboration.',
        'How well does the final product reflect the priorities I repeatedly expressed?',
        'Tell the story of how this effort reached its present state.',
        'Which apparently successful fixes turned out to address only symptoms?',
        'Assess whether our confidence has tracked the strength of our evidence.',
        'Looking across the whole investigation, where did we waste effort?',
        'Which recurring constraints drove the compromises we ultimately accepted?',
        'Explain the pattern behind our repeated changes of direction.',
        'What remains uncertain after reconciling the results of all our experiments?',
        'Where have our retrospective claims overstated what we actually demonstrated?',
        'How have my goals for this business changed since we began discussing it?',
        'What did the unsuccessful attempts contribute to the approach that survived?',
        'Synthesize the recurring objections I raised and how well you addressed them.',
        'Are our remaining blockers new discoveries or unfinished versions of older ones?',
        'Make a reasoned case for closing this investigation based on its entire record.',
        'Audit the evolution from initial requirements to the final handoff.',
        'What cross-cutting lessons emerge from the sequence of decisions and reversals?',
    ],
    'uncertain': [
        'How is it looking?', 'Are we there?', 'Why again?', 'Would that help?',
        'What is your take?', 'Still broken?', 'Ready?', 'Which is best?',
        'Anything else?', 'Can we trust this?',
    ],
}
PAIRS = {
    'train': [
        ('Is this working?', 'The last code change adjusts one parser branch. The user wants the behavior of that patch checked.', 'We have tried many architectures over several months. The user wants to know whether the project fulfills its original goals.'),
        ('What now?', 'The compiler reports a missing import on line 12. We are fixing that one error.', 'We have completed a long experimental program and need to reconcile its findings before choosing a new direction.'),
        ('What changed?', 'Two adjacent benchmark rows use different batch sizes. The user asks about the configuration difference.', 'The original prototype succeeded but subsequent redesigns and evaluations produced conflicting results across the project.'),
        ('Is that enough?', 'The revised function checks for empty input. We are reviewing its validation logic.', 'The project has accumulated many tests and success claims. We are deciding whether the evidence supports release readiness.'),
        ('What went wrong?', 'A single request failed with an expired certificate. The last response contains its error log.', 'Several releases missed the same requirements despite repeated claims of completion. We are reviewing that history.'),
        ('Can we move on?', 'The typo was fixed and the requested unit test passed.', 'We have been revisiting the research assumptions and contradictory findings from the whole investigation.'),
    ],
    'dev': [
        ('Do we have what we need?', 'The tool requires a filename and a destination directory. Both were just provided.', 'Several months of research are ending. The team is assessing evidence, remaining hypotheses, and readiness to publish.'),
        ('Was it worth it?', 'The new laptop costs 200 dollars more and has twice the RAM. We are choosing between two purchases.', 'The investigation spanned several redesigns and benchmarks. The user is assessing the value of the whole effort.'),
        ('Did you miss anything?', 'The user supplied a ten-item shopping list and asks whether the copied list contains all ten.', 'The discussion is a handoff of the entire project, including earlier promises, reversed decisions and unfinished requirements.'),
        ('Can we trust it?', 'The test output reports one checksum match for the downloaded file.', 'The work contains years of inconsistent performance claims and shifting evaluation methods; we are assessing the overall conclusion.'),
    ],
    'test': [
        ('Are we on track?', 'We are checking whether the current download has reached its expected byte count.', 'The programme has gone through repeated scope changes and milestones. We are comparing its trajectory with the initial objectives.'),
        ('What is the verdict?', 'The user asks which of two supplied expressions matches the string in the last message.', 'A lengthy investigation produced competing explanations, reversals and mixed evidence. We need an overall conclusion.'),
        ('What have we learned?', 'We just examined one stack trace and identified the function that raised the exception.', 'Many research attempts have finished, with several abandoned approaches. We are extracting lessons across the whole process.'),
        ('What is still missing?', 'A configuration file requires exactly four fields. The last message contains three of them.', 'The user asks about unfulfilled commitments, unresolved old problems and incomplete validation throughout our project.'),
        ('Did that settle it?', 'Two numerical outputs in the last message are equal. The question is about that specific check.', 'We are evaluating whether the cumulative evidence across experiments resolves the original research question.'),
        ('Where does that leave us?', 'The branch currently contains a merge conflict in one named file. We are resolving that conflict.', 'A long series of design changes and reversals has ended. The user wants a synthesis of remaining risks and overall readiness.'),
    ],
}


def cases():
    rows = []
    for split, examples in [('train',TRAIN),('dev',DEV),('test',TEST)]:
        for label, queries in examples.items():
            for i, query in enumerate(queries):
                rows.append(dict(id=f'{split}-{label}-{i:02}',split=split,label=label,
                    query=query,recent=[],group=f'{split}-{label}-{i:02}',source='authored'))
        for i, (query, focused, overview) in enumerate(PAIRS[split]):
            for label, context in [('focused',focused),('overview',overview)]:
                rows.append(dict(id=f'{split}-pair-{i:02}-{label}',split=split,label=label,
                    query=query,recent=[context],group=f'{split}-pair-{i:02}',source='authored_context_pair'))
    return rows
