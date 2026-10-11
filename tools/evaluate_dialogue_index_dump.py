"""Four summary-index arms against saved neighborhood and full-context answers.

No history ingestion, summary generation, embeddings, or new full-context call.
All questions and reference checkpoints come from the frozen original assay.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.artifacts import read, save, emit
from tools.evaluate_dialogue_neighborhood import MODEL, GATEWAY, digest
from tools.grade_dialogue_neighborhood import JUDGE, extract_json

DEFAULT_ROOT = Path('eval_results/dialogue-index-dump-20261006-r2')
ORIGINAL = Path('eval_results/dialogue-neighborhood-20261006-r1')
CONTROL = Path('eval_results/dialogue-astra-comparison-20261006-r1')
NEW_ARMS = ('index_neighborhoods', 'index_hierarchy', 'index_atomic', 'hierarchy_plus_neighborhoods')
ARMS = ('full_context_batch', 'closing_6k', *NEW_ARMS)
INDEX_NOTE = '''Evidence can include stored summaries labeled [N0000], [H0000], or [A0000].
Cite those labels for summary-derived claims. Original [T169]-style citations
require original passages actually supplied in the evidence; turn ranges in an
index are provenance pointers. Summaries are lossy historical reports, not
independent verification of an assistant's claims. Use the supplied information
to answer the question, clearly distinguishing claims from established results.'''


def render_index(sections, positions, prefix):
    """Preserve each stored summary exactly once with compact chronological IDs."""
    ordered = sorted(sections, key=lambda s: (
        min(positions[p['turn_id']] for p in s['spans']), s['section_id']))
    aliases = {s['section_id']: f'{prefix}{i:04}' for i, s in enumerate(ordered)}
    rows, lines = [], []
    for s in ordered:
        turns = sorted({positions[p['turn_id']] for p in s['spans']})
        label = aliases[s['section_id']]
        children = [aliases[c] for c in s['child_section_ids']]
        line = f'[{label}] turns T{turns[0]}-T{turns[-1]}'
        if children:
            line += ' | children '+', '.join(children)
        lines.append(line+'\n'+s['summary'])
        rows.append(dict(label=label, section_id=s['section_id'], source_id=s['source_id'],
            turn_labels=['T'+str(i) for i in turns], children=children,
            summary_sha256=hashlib.sha256(s['summary'].encode()).hexdigest()))
    text = '\n\n'.join(lines)
    return dict(text=text, rows=rows, sections=len(rows), tokens=count_tokens(text),
                contains_raw_passages=False, new_summaries=0)


def turn_positions(database):
    """Use the same zero-based archive labels as the saved raw-context control."""
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        return {tid:i for i,(tid,) in enumerate(db.execute('SELECT turn_id FROM turns ORDER BY ordinal'))}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    source = read(ORIGINAL/'source.json')
    directory = Path(source['directory'])
    if any(digest(directory/n) != sha for n, sha in source['files'].items()):
        raise ValueError('Original memory changed')
    with closing(sqlite3.connect((directory/'native-spine.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        payload, sha = db.execute('SELECT payload,receipt_sha256 FROM snapshot WHERE id=1').fetchone()
    snapshot = json.loads(payload)
    if identity_sha256(snapshot) != sha or sha != source['snapshot']['snapshot_sha256']:
        raise ValueError('Stored index identity changed')
    positions = turn_positions(directory/'memory.db')
    hierarchy = snapshot['hierarchy_sections']
    indexes = {
        'index_neighborhoods': render_index([s for s in hierarchy if not s['child_section_ids']], positions, 'N'),
        'index_hierarchy': render_index(hierarchy, positions, 'H'),
        'index_atomic': render_index(snapshot['atomic_sections'], positions, 'A'),
    }
    reference_sha = digest(ORIGINAL/'private-reference.json')
    if reference_sha != read(ORIGINAL/'run-plan.json')['reference_sha256'] or reference_sha != digest(CONTROL/'private-reference.json'):
        raise ValueError('Previously frozen reference changed')
    prior_bindings = read(CONTROL/'full-judge-plan.json')['answers']
    frozen = {}
    def freeze(relative, value):
        save(root/relative, value)
        frozen[str(relative)] = digest(root/relative)
    for name in ('source.json', 'questions.json', 'private-reference.json'):
        freeze(name, read(ORIGINAL/name))
    freeze('previous-report.json', read(CONTROL/'full-report.json'))
    audit = read(CONTROL/'final-audit.json')
    if not audit['raw_context_admitted'] or audit['raw_context_compaction_events']:
        raise ValueError('Saved full-context control was not admitted')
    freeze('previous-audit.json', audit)
    if digest(CONTROL/'full-report.json') != audit['comparison_report_sha256']:
        raise ValueError('Saved control scores changed after audit')
    for arm, packet in indexes.items():
        freeze(Path('indexes')/(arm+'.json'), packet)
    questions = read(root/'questions.json')['questions']
    reader_system = read(ORIGINAL/'run-plan.json')['reader_system']+'\n'+INDEX_NOTE
    for q in questions:
        qid = q['id']
        for arm in ('closing_6k', 'full_context_batch'):
            rel = Path('api')/('answer-'+qid+'-'+arm)/'response.json'
            if digest(CONTROL/rel) != prior_bindings[str(rel)]:
                raise ValueError('Previously scored control answer changed')
            freeze(rel, read(CONTROL/rel))
        neighborhood = read(ORIGINAL/'packets'/qid/'closing_6k.json')
        freeze(Path('packets')/qid/'closing_6k.json', neighborhood)
        for arm in NEW_ARMS:
            index = indexes['index_hierarchy' if arm == 'hierarchy_plus_neighborhoods' else arm]
            text = 'STORED SUMMARY INDEX\n'+index['text']
            allowed = [r['label'] for r in index['rows']]
            raw = arm == 'hierarchy_plus_neighborhoods'
            if raw:
                text += '\n\nEXACT NEIGHBORHOOD EVIDENCE\n'+neighborhood['text']
                allowed += [s['label'] for s in neighborhood['spans']]
            freeze(Path('packets')/qid/(arm+'.json'), dict(arm=arm, text=text,
                index_sections=index['sections'], contains_raw_passages=raw,
                allowed_citations=sorted(set(allowed)), sent_evidence_tokens=count_tokens(text)))
    freeze('full-citation-labels.json', dict(labels=['T'+str(i) for i in sorted(positions.values())]))
    save(root/'plan.json', dict(model=MODEL, gateway=GATEWAY, reasoning_effort='none',
        reader_system=reader_system, judge_system=JUDGE.replace('three blinded answers', 'six blinded answers'),
        new_arms=NEW_ARMS, arms=ARMS, actor_calls=60, reused_answers=30, judge_calls=15,
        max_concurrency=3, max_output_tokens=1000, judge_output_tokens=2600,
        prompt_limit=60000, judge_prompt_limit=14000, timeout_s=180, automatic_retries=0,
        implementation_sha256=digest(__file__), frozen=frozen, questions=len(questions),
        history_reingestions=0, fresh_embeddings=0, new_summaries=0,
        combination='Entire hierarchy index plus the existing closing_6k exact raw neighborhood packet.',
        scoring='Six answers jointly blinded for each question; original 70 checkpoints, 140 possible points per arm.',
        limitations=[
            'One archived engineering memory; the same 15 development questions, not held-out validation.',
            'Selective checkpoint coverage, not exhaustive factual accuracy.',
            'New and neighborhood reader: Sol with no reasoning; saved full raw control: Astra with medium reasoning.',
            'Full raw control answered all questions in one request; other arms answer independently.',
            'Sol both answers and grades; blinded scores are not independent human adjudication.',
            'New arms add only the index citation/provenance instructions to the original reader system prompt.',
            'Summary-only arms deliberately omit raw hydration; their index citations do not verify factual support.',
            'Prior answers are reused and regraded jointly; changes to baseline scores reflect grading variability.',
            'No new live ingestion, embedding, GPU, or full-context execution measurements.',
        ]))
    emit(phase='prepared', questions=len(questions), actor_calls=60,
         index_sizes={a:dict(sections=p['sections'], tokens=p['tokens']) for a,p in indexes.items()})


def validate(root):
    plan = read(root/'plan.json')
    if digest(__file__) != plan['implementation_sha256']:
        raise ValueError('Prepared implementation changed')
    for name, sha in plan['frozen'].items():
        if digest(root/name) != sha:
            raise ValueError('Frozen input changed: '+name)
    source = read(root/'source.json')
    if any(digest(Path(source['directory'])/n) != sha for n,sha in source['files'].items()):
        raise ValueError('Source memory changed')
    return plan


def api_job(root, key, messages, *, plan, judge=False):
    import httpx, ssl, truststore
    from memory_condense.interfaces.summary_api import ProviderSummaryAPI
    from memory_condense.interfaces.proxy_memory import WireResponse
    from tools.run_hot_reduced30_answer_judge import _completion_client
    folder = root/'api'/key
    tokens = sum(count_tokens(m['content']) for m in messages)
    limit = plan['judge_prompt_limit'] if judge else plan['prompt_limit']
    if tokens > limit:
        raise ValueError('Frozen prompt budget exceeded; never truncate')
    request = dict(model=plan['model'], gateway=plan['gateway'], messages=messages,
        reasoning_effort='none', max_tokens=plan['judge_output_tokens'] if judge else plan['max_output_tokens'])
    if (folder/'response.json').exists():
        if read(folder/'request.json') != request:
            raise ValueError('Cached request changed')
        return read(folder/'response.json')
    if (folder/'reservation.json').exists():
        raise RuntimeError('Unknown API outcome; retained without resend: '+key)
    receipt = save(folder/'request.json', request)
    save(folder/'reservation.json', dict(request_sha256=receipt.sha256, prompt_proxy_tokens=tokens))
    credentials = _completion_client('LITELLM_KEY', plan['gateway'])
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=plan['timeout_s'], verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as client:
            def send(payload):
                response = client.post(plan['gateway']+'/chat/completions', content=payload,
                    headers={'authorization':'Bearer '+credentials.api_key, 'content-type':'application/json'})
                save(folder/'transport.json', dict(status=response.status_code,
                    content_type=response.headers.get('content-type',''),
                    body=response.text.replace(credentials.api_key, '[REDACTED]'),
                    elapsed_s=time.perf_counter()-started))
                return WireResponse(response.status_code, {'content-type':response.headers.get('content-type','')}, response.content)
            result = ProviderSummaryAPI('openai', plan['model'], send)(messages, max_tokens=request['max_tokens'])
        usage = result.get('usage') or {}
        if type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < .8*tokens:
            raise ValueError('Complete prompt processing not supported by usage receipt')
        save(folder/'response.json', result)
        return result
    except Exception as exc:
        save(folder/'error.json', dict(error_type=type(exc).__name__, elapsed_s=time.perf_counter()-started))
        raise
    finally:
        credentials.close()


def bounded(jobs, function, workers=3):
    """Stop admitting jobs on failure; let already dispatched requests finish."""
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {pool.submit(function, job) for job in list_next(iterator, workers)}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            active.update(pool.submit(function, job) for job in list_next(iterator, len(done)))


def list_next(iterator, size):
    from itertools import islice
    return list(islice(iterator, size))


def generate(root, *, first_only=False):
    plan = validate(root)
    questions = read(root/'questions.json')['questions']
    def answer(job):
        q, arm = job
        packet = read(root/'packets'/q['id']/(arm+'.json'))
        messages = [dict(role='system', content=plan['reader_system']),
                    dict(role='user', content='Archived evidence:\n'+packet['text']+'\n\nQuestion: '+q['question'])]
        result = api_job(root, 'answer-'+q['id']+'-'+arm, messages, plan=plan)
        emit(phase='answered', question=q['id'], arm=arm, elapsed_s=result['elapsed_s'], usage=result['usage'])
    # Largest pure index first: verify successful complete transport before fanout.
    first = (questions[0], 'index_atomic')
    answer(first)
    if not first_only:
        bounded([(q,a) for q in questions for a in NEW_ARMS if (q,a) != first], answer)
    validate(root)


def parse_grade(value, labels, required):
    answers = value['answers']
    if len(answers) != len(labels) or {a['label'] for a in answers} != set(labels):
        raise ValueError('Judge omitted or duplicated an answer')
    result = {}
    for a in answers:
        if set(a['checkpoints']) != set(required) or any(type(n) is not int or n not in (0,1,2) for n in a['checkpoints'].values()):
            raise ValueError('Judge changed the checkpoint population')
        if type(a.get('contradictions')) is not list or type(a.get('scope_honesty')) is not bool:
            raise ValueError('Judge omitted scope/contradiction fields')
        result[labels[a['label']]] = dict(a, points=sum(a['checkpoints'].values()), possible=2*len(required))
    return result


def grade(root):
    plan = validate(root)
    reference = read(root/'private-reference.json')
    questions = read(root/'questions.json')['questions']
    paths = [Path('api')/('answer-'+q['id']+'-'+a)/'response.json' for q in questions for a in ARMS]
    if not all((root/p).exists() for p in paths):
        raise ValueError('All 90 answers must be complete before grading')
    binding = dict(answers={str(p):digest(root/p) for p in paths}, reference_sha256=digest(root/'private-reference.json'))
    target = root/'grading-plan.json'
    if target.exists() and read(target) != binding:
        raise ValueError('Answers changed during grading')
    save(target, binding)
    def judge(q):
        qid = q['id']
        order = sorted(ARMS, key=lambda a:hashlib.sha256((qid+a).encode()).hexdigest())
        labels = dict(zip('ABCDEF', order, strict=True))
        required = reference['questions'][qid]
        checkpoints = {fid:dict(claim=reference['facts'][fid]['claim'],
            sources=[dict(label=s['label'], role=s['role'], quote=s['quote'][:450])
                     for s in reference['facts'][fid]['supports']]) for fid in required}
        answers = {label:read(root/'api'/('answer-'+qid+'-'+arm)/'response.json')['content'] for label,arm in labels.items()}
        messages = [dict(role='system', content=plan['judge_system']), dict(role='user', content=json.dumps(
            dict(question=q['question'], checkpoints=checkpoints, answers=answers), ensure_ascii=False))]
        response = api_job(root, 'judge-'+qid, messages, plan=plan, judge=True)
        parsed = extract_json(response['content'])
        scored = parse_grade(parsed, labels, required)
        save(root/'grades'/(qid+'.json'), dict(question=qid, arms=scored, labels=labels, comparison=parsed['comparison']))
        emit(phase='graded', question=qid, points={a:s['points'] for a,s in scored.items()})
    bounded(questions, judge)
    report(root)


def cited_labels(text):
    groups = re.findall(r'\[((?:T\d+|[NHA]\d{4})(?:\s*[,;]\s*(?:T\d+|[NHA]\d{4}))*)\]', text)
    return {label for group in groups for label in re.findall(r'T\d+|[NHA]\d{4}', group)}


def report(root):
    plan = validate(root)
    binding = read(root/'grading-plan.json')
    if any(digest(root/p) != sha for p,sha in binding['answers'].items()):
        raise ValueError('Scored answers changed')
    questions = read(root/'questions.json')['questions']
    grades = {q['id']:read(root/'grades'/(q['id']+'.json')) for q in questions}
    previous = read(root/'previous-report.json')
    totals, rows = {}, []
    for arm in ARMS:
        records, inputs, outputs, elapsed, evidence, bad_citations = [], [], [], [], [], []
        for q in questions:
            qid = q['id']
            records.append(grades[qid]['arms'][arm])
            response = read(root/'api'/('answer-'+qid+'-'+arm)/'response.json')
            if arm == 'full_context_batch':
                allowed = set(read(root/'full-citation-labels.json')['labels'])
            else:
                packet = read(root/'packets'/qid/(arm+'.json'))
                allowed = set(packet['allowed_citations']) if arm in NEW_ARMS else {s['label'] for s in packet['spans']}
                evidence.append(packet['sent_evidence_tokens'])
                usage = response['usage']
                inputs.append(usage['prompt_tokens'])
                outputs.append(usage['completion_tokens'])
                elapsed.append(response['elapsed_s'])
            missing = sorted(cited_labels(response['content'])-allowed)
            if missing:
                bad_citations.append(dict(question=qid, labels=missing))
        points = sum(r['points'] for r in records)
        possible = sum(r['possible'] for r in records)
        totals[arm] = dict(points=points, possible=possible, checkpoint_score_pct=100*points/possible,
            prompt_tokens=sum(inputs), completion_tokens=sum(outputs), mean_reader_s=statistics.mean(elapsed) if elapsed else None,
            mean_evidence_tokens=statistics.mean(evidence) if evidence else None,
            answers_with_reference_contradictions=sum(bool(r['contradictions']) for r in records),
            scope_honest=sum(r['scope_honesty'] for r in records), unprovided_citations=bad_citations,
            model_calls=15, reader_model=MODEL, reused_answers=arm not in NEW_ARMS)
        if arm == 'full_context_batch':
            old = previous['totals'][arm]
            totals[arm].update(prompt_tokens=old['prompt_tokens'], completion_tokens=old['completion_tokens'],
                mean_evidence_tokens=old['mean_evidence_tokens'], model_calls=1, reader_model='gpt-6-astra',
                shared_batch_elapsed_s=old['shared_batch_elapsed_s'])
        if arm in previous['totals']:
            totals[arm]['previous_points'] = previous['totals'][arm]['points']
            totals[arm]['previous_score_pct'] = previous['totals'][arm]['checkpoint_score_pct']
    for q in questions:
        records = grades[q['id']]['arms']
        rows.append(dict(id=q['id'], question=q['question'], possible=records[ARMS[0]]['possible'],
                         points={a:records[a]['points'] for a in ARMS}))
    comparisons = {}
    for arm in NEW_ARMS:
        comparisons[arm] = {}
        for control in ('full_context_batch', 'closing_6k'):
            delta = [r['points'][arm]-r['points'][control] for r in rows]
            comparisons[arm][control] = dict(wins=sum(n>0 for n in delta), ties=sum(n==0 for n in delta),
                losses=sum(n<0 for n in delta), score_difference_pp=totals[arm]['checkpoint_score_pct']-totals[control]['checkpoint_score_pct'])
    save(root/'report.json', dict(questions=len(questions), totals=totals, rows=rows, comparisons=comparisons,
        limitations=plan['limitations'], source_unchanged=True, new_actor_calls=60, reused_answers=30, judge_calls=15,
        scoring='Joint blinded regrade of all six arms against the same frozen 70 checkpoints.',
        previous_report_sha256=digest(root/'previous-report.json')))
    emit(phase='report', totals=totals, comparisons=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','first','generate','grade','report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.phase == 'first':
        generate(args.root, first_only=True)
    else:
        globals()[args.phase](args.root)
