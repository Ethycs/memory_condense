"""Execute the frozen artifact battery with matched full-context and memory arms."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from tools import engineering_research_battery as battery
from tools.engineering_research_gateway import Gateway, save, read, emit
from tools.engineering_research_execution import execute_tests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'eval_results/engineering-research-live-20260925-r1'
PILOT = ('E01', 'R05')
WORKING_CAP = 8192
ACTOR_TOOLS = '''
You operate on an initially empty isolated workspace. Produce the requested files.
Return one JSON action with no Markdown fences. Available actions:
{"action":"write","path":"file.py","content":"complete text"}
{"action":"write_many","artifacts":{"file.py":"complete text","test_file.py":"complete text"}}
{"action":"read","path":"file.py","start_line":1,"line_count":120}
{"action":"list"}
{"action":"test"}
{"action":"recall","query":"a specific missing prior requirement or discussion"}
{"action":"finish","message":"work delivered and actual validation"}
write_many accepts up to eight files. Writes replace the named file. read returns
at most 120 lines. test runs stdlib unittest discovery over test_*.py in your
workspace with a 60-second CPU, 75-second wall, 512-MiB and one-process limit.
Only workspace files are accessible; no network, credentials, installs or external
processes are available. Tool observations are bounded to 16,000 characters.
You have at most 24 responses. Use source turn IDs exactly when citing evidence.
For research, claims.json is a nonempty JSON list of claim/status/evidence entries.
Do not replace missing evidence with a confident guess. Finish when complete.
'''
JUDGE = '''You assess two anonymous engineering/research artifacts against the same
current task, prior source and four source-grounded criteria. The historical
assistant is fallible, not a gold answer. Accept equivalent correct solutions.
Do not demand incidental wording or details absent from the user's requirements.
Source citations prove quotation membership, not entailment or scientific truth.
Do not convert proposals, reported measurements or assumptions into proved facts.
For each candidate and criterion return score 0=unmet, 1=partial, 2=met, or null
if genuinely ambiguous. Justify defects with specific source and artifact evidence.
An absence can be justified as missing; never invent a quote for an absent feature.
Return JSON: {"candidates":{"A":{"criteria":[{"id":"E01-C1","score":2,
"reason":"...","artifact_file":"file.py","artifact_quote":"exact excerpt or empty if absent",
"source_turn_ids":["family:T0000"]}]},"B":{"criteria":[...]}}}.
All four criteria are required for both candidates. No arm identity or model names
are available. Mechanical check failures are assessed separately by the controller.
'''


def prepare(run, bundle, reuse=None, web_baseline=None):
    checked = battery.audit(bundle)
    manifest = json.loads((bundle / 'battery.json').read_text(encoding='utf-8'))
    files = [Path(__file__), *Path(__file__).parent.glob('engineering_research_*.py')]
    files += list((ROOT / 'src/memory_condense').rglob('*.py'))
    files += [ROOT / 'tools/native_spine_engineering_session.py', ROOT / 'tools/build_spine_corpus_hierarchy.py',
              ROOT / 'tools/compile_native_spine_attention.py']
    plan = dict(schema='engineering-research-live-v1', battery_sha256=checked['battery_sha256'],
        bundle=str(bundle.resolve()), cases=manifest['cases'], pilot=list(PILOT),
        gateway='https://central-dev.zt:4000/v1',
        models=dict(raw='codex_sdk/gpt-5.6-sol', merge='qwen3-8b', actor='codex_sdk/gpt-5.6-sol', judge='codex_sdk/gpt-5.6-terra'),
        budgets={
            'raw': dict(calls=400, prompt_cap=7000, output_cap=4096, input_token_budget=2_800_000),
            'merge': dict(calls=600, prompt_cap=2048, output_cap=768, input_token_budget=1_228_800),
            'actor': dict(calls=960, prompt_cap=131072, output_cap=4096, input_token_budget=40_000_000),
            'judge': dict(calls=40, prompt_cap=131072, output_cap=4096, input_token_budget=5_242_880)},
        actor_calls_per_arm_case=24, memory_prompt_cap=24576, full_context_local_cap=131072,
        model_context_capacity='Local ceiling only; provider limit not independently verified. Overflow is an explicit failure.',
        working_tokens=WORKING_CAP, tool_output_chars=16000, retries=0, concurrency=1,
        raw_summary_retries=0, merge_retries=0,
        chat_ingestion=True,
        chat_batch_exchanges=6,
        ingestion='Every IO event is durably captured; recall runs per request with a recent window, and native ingestion batches six complete exchanges.',
        memory='Public cap-8 unchanged; exact hydrated spans, original role/source labels, no experimental hints.',
        timestamp_policy='Export time is only a technical storage/as-of anchor. No invented event times are shown to actors.',
        local_attention='Existing six-layer float16 Qwen prefix; summary-only inputs; FP32 embedding matrices.',
        current_task_policy='Identical current request, task, tools and working-window instructions in both arms.',
        grading='One blinded Terra pair review; second reversed-order review on every non-perfect pair and E01/R05 passing controls.',
        actor_system_suffix=ACTOR_TOOLS, judge_system=JUDGE,
        implementation={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if not (ROOT / '.cache/models/Qwen3-8B/model.safetensors.index.json').is_file():
        raise ValueError('The existing local Qwen checkpoint is required before any provider work')
    if reuse is not None:
        prior = read(reuse / 'run-plan.json')
        if prior['battery_sha256'] != plan['battery_sha256'] or prior['models'] != plan['models']:
            raise ValueError('Cache predecessor must use this battery and models')
        if list((reuse/'cases').glob('*/ */result.json')) or list((reuse/'cases').glob('*/*/result.json')):
            raise ValueError('Preparation repair may not discard completed actor results')
        for name in ('cache','gateway'):
            source = reuse/name
            if source.exists():
                shutil.copytree(source,run/name)
        plan['preparation_repair'] = dict(predecessor=str(reuse.resolve()),
            prior_plan_sha256=identity_sha256(prior), retained_calls=len(list((run/'gateway').glob('*.response.json'))),
            reason='Resolve legacy worktree-relative checkpoint location; preserve completed compilation and generation journals.')
    if web_baseline is not None:
        from tools.engineering_research_web import WEB_TOOLS
        prior = read(web_baseline / 'run-plan.json')
        if prior['battery_sha256'] != plan['battery_sha256'] or prior['models'] != plan['models']:
            raise ValueError('Web follow-up must preserve the battery and models')
        plan['cases'] = [r for r in plan['cases'] if r['id'] in ('R04', 'R09', 'R10')]
        plan['pilot'] = []
        plan['web'] = dict(enabled=True, calls_per_arm=4, bridge_timeout_s=900,
            tools=WEB_TOOLS, baseline=str(web_baseline.resolve()),
            purpose='Check completed research omissions with equal public web access; retain baseline failures.',
            selection='All three completed memory research outputs with semantic misses; full-context controls rerun once.',
            scoring='Frozen historical criteria unchanged. Checked public URLs allowed in analysis.md; claims.json retains exact historical citations.')
        plan['budgets'] = {
            'raw': dict(calls=80, prompt_cap=7000, output_cap=4096, input_token_budget=560_000),
            'merge': dict(calls=160, prompt_cap=2048, output_cap=768, input_token_budget=327_680),
            'actor': dict(calls=144, prompt_cap=131072, output_cap=4096, input_token_budget=12_000_000),
            'judge': dict(calls=6, prompt_cap=131072, output_cap=4096, input_token_budget=786_432)}
        shutil.copytree(web_baseline / 'cache', run / 'cache')
        plan['cache_reuse'] = 'Completed source compilation cache copied; new stores ingest exact original cutoffs and all new events.'
    path = run / 'run-plan.json'
    save(path, plan)
    emit(phase='prepared', cases=len(plan['cases']), pilot=list(PILOT), run=str(run), provider_calls=0)


def parse_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[text.index('\n')+1:text.rfind('```')].strip()
    return json.loads(text)


def safe_path(workspace, value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Use a relative workspace filename')
    path = Path(value)
    if path.root or path.drive or '..' in path.parts:
        raise ValueError('Path is outside the workspace')
    target = (workspace / path).resolve()
    target.relative_to(workspace.resolve())
    return target


def source_query(actor):
    # Query uses public current input only. No rubric or reference enters routing.
    return actor['original_request'].strip() + '\n' + actor['task']


def memory_phase(run, folder, actor, rows, scope, *, query=None, ingest=None, chat=False, learning=None):
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / ('ingest.json' if ingest is None else 'reopened.json')
    request = dict(operation='install' if ingest is None else 'reopen', case_root=str(folder.parents[0] / 'store'),
        rows=rows, source_id=actor['source']['family'], storage_timestamp=actor['source']['export_timestamp'],
        output=str(output), scope=scope, query=query, ingest_receipt=str(ingest) if ingest else None)
    if chat:
        request['chat'] = True
        if actor.get('native_seed'):
            request['native_seed'] = actor['native_seed']
    if learning is not None:
        request.update(operation='learn', **learning)
    request_path = folder / ('install-request.json' if ingest is None else 'reopen-request.json')
    save(request_path, request)
    if output.with_suffix('.json.sha256').exists():
        return read(output)
    if not chat:  # The chat JSONL transport reserves stdout for request replies.
        emit(phase='memory_process', scope=scope, operation=request['operation'], turns=len(rows))
    with (folder / (request['operation'] + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'memory', '--run', str(run),
                                  '--request', str(request_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    if process.returncode:
        raise RuntimeError('Memory phase failed; see ' + str(folder / (request['operation'] + '.log')))
    return read(output)


def sync_and_reopen(run, arm_root, actor, rows, label, query=None):
    folder = arm_root / label
    scope = f"{actor['case_id']}/memory/{label}"
    installed = memory_phase(run, folder, actor, rows, scope)
    reopened = memory_phase(run, folder, actor, rows, scope, query=query, ingest=folder/'ingest.json')
    return reopened


def execute_action(workspace, action):
    kind = action.get('action')
    if kind == 'write_many':
        files = action.get('artifacts')
        if not isinstance(files, dict) or not 1 <= len(files) <= 8:
            raise ValueError('write_many requires one to eight text files')
        for name, text in files.items():
            if not isinstance(text, str):
                raise ValueError('File contents must be text')
            safe_path(workspace, name)
        for name, text in files.items():
            path = safe_path(workspace, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8', newline='')
        return {'written': list(files)}
    if kind == 'write':
        return execute_action(workspace, {'action': 'write_many', 'artifacts': {action['path']: action['content']}})
    if kind == 'read':
        path = safe_path(workspace, action['path'])
        start, count = action.get('start_line', 1), action.get('line_count', 120)
        if type(start) is not int or start < 1 or type(count) is not int or not 1 <= count <= 120:
            raise ValueError('Invalid line window')
        lines = path.read_text(encoding='utf-8').splitlines()
        return {'text': '\n'.join(f'{i+1}: {line}' for i, line in enumerate(lines) if start-1 <= i < start-1+count)}
    if kind == 'list':
        return {'files': [p.relative_to(workspace).as_posix() for p in sorted(workspace.rglob('*')) if p.is_file()]}
    if kind == 'test':
        return execute_tests(workspace)
    raise ValueError('Unknown action')


def messages_for(actor, evidence, live, web=False):
    if web:
        from tools.engineering_research_web import browsing_actor
        actor = browsing_actor(actor)
    messages = battery.actor_messages(actor, evidence)
    messages[0]['content'] += ACTOR_TOOLS
    if web:
        from tools.engineering_research_web import WEB_TOOLS
        messages[0]['content'] = messages[0]['content'].replace(
            'Only workspace files are accessible; no network, credentials, installs or external\nprocesses are available.',
            'Candidate code can access workspace files only; it has no network, credentials,\ninstalls or external processes. Public browsing uses the separate web actions below.')
        messages[0]['content'] += WEB_TOOLS
    if live:
        messages[1]['content'] += '\n\nWork during the current task (actions and observations):\n' + battery.render_history(live)
    return messages


def run_arm(run, record, arm):
    complete = run / 'cases' / record['id'] / arm / 'result.json'
    if complete.exists():
        return read(complete)
    # Sealed historical plans retain their original lifecycle. Every newly
    # prepared plan uses the public chat interface, including the control arm.
    plan = read(run / 'run-plan.json')
    if not plan.get('chat_ingestion'):
        return _run_arm(run, record, arm)
    from tools.engineering_research_chat import open_chat
    actor = battery.read_binding(record['actor'])
    with open_chat(run, run / 'cases' / record['id'] / arm, actor, arm,
                   batch_exchanges=plan.get('chat_batch_exchanges', 0)) as chat:
        return _run_arm(run, record, arm, chat=chat)


def _run_arm(run, record, arm, *, chat=None, reader_gateway=None,
             inline_memory=False, recall_each_action=False):
    arm_root = run / 'cases' / record['id'] / arm
    complete = arm_root / 'result.json'
    if complete.exists():
        return read(complete)
    workspace = arm_root / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    actor = battery.read_binding(record['actor'])
    web = read(run / 'run-plan.json').get('web', {}).get('enabled', False)
    historical = list(actor['history'])
    gateway = reader_gateway or Gateway(run)
    current = dict(turn_id=actor['current_turn_id'], source_id=actor['source']['family'], role='user', text=actor['original_request'])
    if inline_memory:
        # Capture the adapted engineering request that the actor actually sees.
        current['text'] = source_query(actor)
    if chat is not None:
        from dataclasses import asdict
        from memory_condense.application.chat_io import ChatIO
        from tools.engineering_research_chat import event_from_row
        io = ChatIO(chat)
        def admit(row, timestamp=None):
            if row['turn_id'].startswith('_chat:'):
                # Recall's transaction already captured this exact tool result.
                if not any(e.event_id == row['turn_id'] and e.text == row['text'] for e in chat.events()):
                    raise ValueError('Saved recall event is missing from the chat stream')
                return
            chat.ingest(event_from_row(row, actor['source']['family'], timestamp))
        with chat.capture_exchange():
            chat.ingest_many([event_from_row(r, actor['source']['family'], actor['source']['export_timestamp']) for r in historical])
            chat.flush()
        chat.ingest(event_from_row(current, actor['source']['family']))
        def recalled_packet(query, **kwargs):
            value = chat.recall(query, **kwargs)
            return dict(asdict(value), context_text=value.context_text)
    if arm == 'memory' and chat is not None:
        packet = recalled_packet(source_query(actor), packet_id='initial', input_event_id=current['turn_id'])
        evidence = packet['context_text']
    elif arm == 'memory':
        packet = sync_and_reopen(run, arm_root, actor, historical, 'initial', source_query(actor))
        evidence = packet['text']
    else:
        evidence = battery.render_history(historical)
    live, events, finished, error = [], [], False, None
    # Replay completed controller actions from disk, never reissue generations.
    # Existing event records are append-only. File writes are already materialized.
    prior = sorted((arm_root / 'actions').glob('*/event.json'))
    for p in prior:
        event = read(p)
        if chat is not None:
            for row in event['rows']:
                admit(row)
        events.extend(event['rows'])
        finished = event['finished']
    live = list(events)
    # After a resumed run, rebuild only when earlier working events exceed the cap.
    responses = []
    inline_failures = invalid_actions = 0
    started = time.perf_counter()
    for index in range(len(prior), 24):
        if finished:
            break
        if (run/'STOP').exists():
            raise RuntimeError('Run stopped')
        folder = arm_root / 'actions' / f'{index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        if count_tokens(battery.render_history(live)) > WORKING_CAP:
            if arm == 'memory':
                packet = (recalled_packet(source_query(actor), packet_id=f'flush-{index:03d}', input_event_id=current['turn_id'])
                          if chat is not None else sync_and_reopen(run, arm_root, actor, historical + [current] + events, f'flush-{index:03d}', source_query(actor)))
                evidence = packet.get('context_text', packet['text'])
            # Full context's old work stays in its evidence block; memory uses retrieval.
            elif events:
                evidence = battery.render_history(historical + events)
            live = []
        if recall_each_action and arm == 'memory' and index > 0:
            packet = recalled_packet(source_query(actor), packet_id=f'action-{index:03d}',
                                     input_event_id=current['turn_id'])
            evidence = packet['context_text']
        messages = messages_for(actor, evidence, live, web=web)
        cap = 24576 if arm == 'memory' else 131072
        tokens = count_chat_prompt_token_proxy(messages)
        if tokens > cap:
            error = f'prompt_overflow:{tokens}>{cap}'
            break
        save(folder/'prompt.json', dict(messages=messages, prompt_tokens_proxy=tokens,
             memory_packet_sha256=identity_sha256(packet) if arm == 'memory' else None,
             history_sha256=identity_sha256(historical), current_work_sha256=identity_sha256(events)))
        request_id = f"{record['id']}:{arm}:A{index:03d}"
        def generate():
            if inline_memory:
                from memory_condense.application.inline_memory import generate_inline
                def serve(kind, served, **kwargs):
                    actual_tokens = count_chat_prompt_token_proxy(served)
                    if actual_tokens > cap:
                        raise ValueError(f'Inline prompt exceeds arm budget: {actual_tokens}>{cap}')
                    save(folder/'served-prompt.json', dict(messages=served,prompt_tokens_proxy=actual_tokens))
                    return gateway.call(kind, served, **kwargs)
                return generate_inline(serve, messages, user_text=current['text'],
                                       scope=f"{record['id']}/{arm}/{index:03d}", max_tokens=4096)
            return gateway.call('actor', messages, scope=f"{record['id']}/{arm}/{index:03d}")
        if chat is not None:
            response = io.invoke(request_id=request_id, reader=generate,
                                 input_event_id=current['turn_id'],
                                 packet_id=packet['packet_id'] if arm == 'memory' else None,
                                 packet_ids=[r['metadata']['_chat']['packet_id'] for r in live
                                             if r.get('metadata', {}).get('_chat', {}).get('kind') == 'recall'])
        else:
            response = generate()
        save(folder/'response.json', response)
        responses.append(response)
        if inline_memory and chat is not None:
            status = chat.event(request_id + ':assistant').metadata['inline_generation']['status']
            inline_failures = inline_failures + 1 if status == 'fallback' else 0
            if inline_failures >= 5:
                raise RuntimeError('Five consecutive rejected inline summary pairs')
        response_row = dict(turn_id=request_id, source_id=actor['source']['family'],
                            role='assistant', text=response['content'])
        if chat is not None and response['content'].strip():
            response_row = next(e.row(actor['source']['family']) for e in chat.events()
                                if e.event_id == request_id + ':assistant')
        action = {}
        recall_event = None
        try:
            if not response['content'].strip():
                error = 'empty_gateway_response'
                raise ValueError('Gateway returned no actor content; stopping this arm without retry')
            action = parse_json(response['content'])
            if not isinstance(action, dict):
                raise ValueError('Action must be a JSON object')
            if action.get('action') == 'finish':
                finished = True
                observation = {'finished': True, 'message': action.get('message', '')}
            elif action.get('action') == 'recall':
                query = action.get('query')
                if not isinstance(query, str) or not query.strip():
                    raise ValueError('Recall needs a nonempty query')
                if arm == 'memory':
                    if chat is not None:
                        recalled = recalled_packet(query, packet_id=f'recall-{index:03d}', input_event_id=response_row['turn_id'])
                        recall_event = next(e.row(actor['source']['family']) for e in chat.events()
                                            if e.event_id == f'_chat:recall:recall-{index:03d}')
                        observation = dict(evidence=recalled['context_text'], packet_id=recalled['packet_id'],
                                           references=recalled['references'], input_event_id=recalled['input_event_id'])
                    else:
                        recalled = sync_and_reopen(run, arm_root, actor, historical + [current] + events + [response_row],
                                                   f'recall-{index:03d}', query)
                        observation = {'evidence': recalled['text']}
                else:
                    observation = {'note': 'All earlier source and work are present in the supplied full context.'}
            elif action.get('action') in ('web_search', 'web_open'):
                from tools.engineering_research_web import browse
                observation = browse(run, action, f"{record['id']}/{arm}/{index:03d}")
            else:
                observation = execute_action(workspace, action)
        except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            observation = {'tool_error': type(exc).__name__ + ': ' + str(exc)}
        text = json.dumps(observation, ensure_ascii=False)
        if len(text) > 16000:
            text = text[:16000] + '\n[Tool output limit reached; narrow the request.]'
        tool_row = dict(turn_id=f"{record['id']}:{arm}:O{index:03d}", source_id=actor['source']['family'], role='tool', text=text)
        if recall_event is not None:
            tool_row = recall_event
        elif chat is not None:
            io.tool_result(event_id=tool_row['turn_id'], text=tool_row['text'], call_event_id=response_row['turn_id'])
            tool_row = next(e.row(actor['source']['family']) for e in chat.events() if e.event_id == tool_row['turn_id'])
        # Empty gateway envelopes remain in response.json; they contain no
        # conversation content to ingest. Preserve the explicit error observation.
        rows = ([response_row] if response['content'].strip() else []) + [tool_row]
        save(folder/'event.json', dict(rows=rows, observation=observation, finished=finished))
        events.extend(rows)
        live.extend(rows)
        if inline_memory:
            invalid_actions = invalid_actions + 1 if 'tool_error' in observation else 0
            if invalid_actions >= 2:
                raise RuntimeError('Two consecutive malformed or unusable tool actions')
        emit(phase='actor_action', case=record['id'], arm=arm, action=index, kind=action.get('action') if isinstance(action,dict) else 'invalid',
             elapsed_s=response['elapsed_s'], prompt_tokens=tokens)
        if error == 'empty_gateway_response':
            break
    artifacts = {name: safe_path(workspace, name).read_text(encoding='utf-8') for name in actor['deliverables'] if safe_path(workspace,name).is_file()}
    result = dict(case_id=record['id'], arm=arm, artifacts=artifacts, finished=finished,
                  error=error or (None if finished else 'action_limit'), actor_calls=len(prior)+len(responses),
                  actor_loop_wall_s=time.perf_counter()-started, workspace=str(workspace))
    try:
        result['structural'] = battery.validate_result(result, actor)
    except (ValueError, TypeError) as exc:
        result['structural'] = dict(structurally_complete=False, error=str(exc), quality_scored=False)
    if actor['task_kind'] == 'python_component':
        result['unit_tests'] = execute_tests(workspace)
        from tools.engineering_research_checks import MODULES
        if record['id'] in MODULES:
            result['behavioral_checks'] = execute_tests(workspace, acceptance_case=record['id'])
    if arm == 'memory':
        try:
            if chat is not None:
                result['chat'] = chat.flush()
                result['final_reopen'] = chat.backend.last_reopen
            else:
                result['final_reopen'] = sync_and_reopen(run, arm_root, actor, historical + [current] + events, 'final')
        except (RuntimeError, ValueError) as exc:
            result['lifecycle_error'] = str(exc)
    save(complete, result)
    emit(phase='arm_complete', case=record['id'], arm=arm, actor_calls=result['actor_calls'], finished=finished,
         structural=result['structural']['structurally_complete'], checks=result.get('behavioral_checks',{}).get('exit_code'))
    return result


def run_arm_recording_failure(run, record, arm):
    """Keep an exhausted case in the denominator and continue independent cases."""
    try:
        return run_arm(run, record, arm)
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        folder = run / 'cases' / record['id'] / arm
        actor = battery.read_binding(record['actor'])
        workspace = folder / 'workspace'
        artifacts = {name: safe_path(workspace, name).read_text(encoding='utf-8')
                     for name in actor['deliverables'] if safe_path(workspace, name).is_file()}
        result = dict(case_id=record['id'], arm=arm, artifacts=artifacts, finished=False,
                      error=type(exc).__name__ + ': ' + str(exc),
                      actor_calls=len(list((folder/'actions').glob('*/response.json'))),
                      workspace=str(workspace), structural=dict(structurally_complete=False,
                      quality_scored=False, error='Arm failed before normal completion'))
        if arm == 'memory':
            result['lifecycle_error'] = result['error']
        save(folder/'result.json', result)
        emit(phase='arm_failed',case=record['id'],arm=arm,error=result['error'])
        return result


def validate_review(value, rubric, candidates, actor):
    source_ids = {r['turn_id'] for r in actor['history']} | {actor['current_turn_id']}
    expected = {r['id'] for r in rubric['criteria']}
    if set(value.get('candidates', {})) != {'A', 'B'}:
        raise ValueError('Review requires both anonymous candidates')
    for label, result in value['candidates'].items():
        rows = result.get('criteria', [])
        if len(rows) != len(expected) or {r['id'] for r in rows} != expected:
            raise ValueError('Review criterion population changed')
        for r in rows:
            if r['score'] is not None and (type(r['score']) is not int or r['score'] not in (0,1,2)):
                raise ValueError('Invalid criterion score')
            if not isinstance(r.get('reason'), str) or not r['reason'].strip():
                raise ValueError('Missing review justification')
            if not r.get('source_turn_ids') or not set(r['source_turn_ids']) <= source_ids:
                raise ValueError('Review cited nonexistent source')
            quote = r.get('artifact_quote')
            if quote and quote not in candidates[label].get(r.get('artifact_file'), ''):
                raise ValueError('Review fabricated artifact quote')
    return value


def grade_pair(run, record, results, *, gateway=None):
    folder = run / 'cases' / record['id'] / 'grading'
    if (folder/'reviews.json').exists():
        return
    actor = battery.read_binding(record['actor'])
    rubric = battery.read_binding(record['rubric'])
    plan = read(run/'run-plan.json')
    judge_system = JUDGE
    if plan.get('web', {}).get('enabled'):
        from tools.engineering_research_web import browsing_actor
        actor = browsing_actor(actor)
        judge_system += '\nPublic web tools are allowed equally for both candidates. Assess the unchanged historical criteria; web references cannot substitute for missing private session facts. External URL claims require separate verification and are not automatically certified by this review.\n'
    order = ['memory','full_context'] if int(hashlib.sha256(record['id'].encode()).hexdigest(),16)%2 else ['full_context','memory']
    reviews = []
    for iteration in range(2):
        if iteration == 1 and record['id'] not in PILOT and reviews and reviews[0]['valid'] and all(
            r['score'] == 2 for v in reviews[0]['review']['candidates'].values() for r in v['criteria']):
            break
        mapping = dict(zip(('A','B'), order if iteration == 0 else list(reversed(order))))
        candidates = {k: results[arm]['artifacts'] for k,arm in mapping.items()}
        payload = dict(
            task={k:actor[k] for k in ('case_id','original_request','current_turn_id','task','deliverables','public_checks')},
            source=actor['history'], rubric=rubric, candidates=candidates)
        if plan.get('web', {}).get('enabled'):
            # Preserve every source character without JSON-escaping the large
            # history. Escaped backslashes inflated R04 beyond the prompt cap.
            payload.pop('source')
            user_content = 'Complete historical source:\n' + battery.render_history(actor['history'])
            user_content += '\n\nTask, rubric and anonymous candidates:\n' + json.dumps(payload, ensure_ascii=False)
        else:
            user_content = json.dumps(payload, ensure_ascii=False)
        messages = [{'role':'system','content':judge_system}, {'role':'user','content':user_content}]
        response = (gateway or Gateway(run)).call('judge', messages, scope=f"{record['id']}/review/{iteration}")
        result = dict(mapping=mapping, response=response)
        try:
            result.update(valid=True, review=validate_review(parse_json(response['content']),rubric,candidates,actor))
        except (ValueError, TypeError, KeyError) as exc:
            result.update(valid=False, error=str(exc))
        save(folder/f'review-{iteration}.json',result)
        reviews.append(result)
    save(folder/'reviews.json',dict(reviews=reviews))


def summarize(run):
    plan = read(run/'run-plan.json')
    rows = []
    for record in plan['cases']:
        folder = run/'cases'/record['id']
        path = folder/'grading'/'reviews.json'
        if not path.exists():
            continue
        reviews = read(path)['reviews']
        arms = {}
        for arm in ('memory','full_context'):
            result = read(folder/arm/'result.json')
            scores = []
            for criterion in battery.read_binding(record['rubric'])['criteria']:
                values = []
                for review in reviews:
                    if review['valid']:
                        label = next(k for k,v in review['mapping'].items() if v == arm)
                        values.append(next(r['score'] for r in review['review']['candidates'][label]['criteria'] if r['id']==criterion['id']))
                scores.append(values[0] if values and all(v==values[0] for v in values) else None)
            structural = result['structural']['structurally_complete']
            mechanical = all(result[k]['exit_code']==0 for k in ('unit_tests','behavioral_checks') if k in result)
            status = 'unresolved' if None in scores else ('pass' if all(s==2 for s in scores) else 'fail')
            if not structural or not mechanical or not result['finished'] or (arm=='memory' and not result.get('final_reopen')):
                status = 'fail'
            arms[arm] = dict(status=status, criterion_scores=scores, structural=structural, mechanical=mechanical,
                             actor_calls=result['actor_calls'], final_reopen_verified=bool(result.get('final_reopen')) if arm=='memory' else None)
        rows.append(dict(case_id=record['id'],domain=record['domain'],family=record['family'],arms=arms))
    usage = Counter()
    timing = {}
    calls = Counter()
    for p in (run/'gateway').glob('*.response.json'):
        response = read(p)
        job = read(p.with_name(p.name.replace('.response.json','.request.json')))
        kind = job['kind']
        calls[kind] += 1
        if response.get('usage'):
            for key in ('prompt_tokens','completion_tokens'):
                usage[kind+':'+key] += response['usage'].get(key,0)
        if kind == 'actor' and response.get('elapsed_s') is not None:
            arm = job['scope'].split('/')[1]
            timing.setdefault(arm,[]).append(response['elapsed_s'])
    report = dict(status='complete' if len(rows)==len(plan['cases']) else 'partial', completed_pairs=len(rows), planned_pairs=len(plan['cases']),
        rows=rows, provider_calls=dict(calls), provider_usage=dict(usage),
        per_arm={arm:dict(Counter(r['arms'][arm]['status'] for r in rows)) for arm in ('memory','full_context')},
        actor_response_latency={arm:dict(mean_s=statistics.mean(v),median_s=statistics.median(v),responses=len(v)) for arm,v in timing.items()},
        note='Task quality and mechanical outcomes; independent paired checkpoints, not full-session or 1M stress score.')
    destination = run/f'report-{len(rows):02d}.json'
    # Reports only seal once the corresponding completed population is stable.
    if not destination.exists():
        save(destination,report)
    emit(phase='report',completed_pairs=len(rows),per_arm=report['per_arm'],provider_calls=dict(calls))
    return report


def run_cases(run, stage):
    plan = read(run/'run-plan.json')
    pilots = plan.get('pilot', list(PILOT))
    selected = [r for r in plan['cases'] if (r['id'] in pilots) == (stage=='pilot')]
    if stage == 'remaining' and any(not (run/'cases'/i/'grading'/'reviews.json').exists() for i in pilots):
        raise ValueError('Complete the paired pilot before validation')
    for record in selected:
        order = ['memory','full_context'] if int(record['id'][1:]) % 2 else ['full_context','memory']
        results = {arm:run_arm_recording_failure(run,record,arm) for arm in order}
        grade_pair(run,record,results)
        emit(phase='pair_complete',case=record['id'])
    summarize(run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','gateway','memory','pilot','remaining','report'])
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--bundle',type=Path,default=battery.DEFAULT_OUTPUT)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--reuse-preparation',type=Path)
    parser.add_argument('--web-baseline',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    if args.phase=='prepare':
        prepare(run,args.bundle,args.reuse_preparation,args.web_baseline)
    elif args.phase=='gateway':
        from tools.engineering_research_gateway import worker
        worker(run)
    elif args.phase=='memory':
        from tools.engineering_research_memory import run as memory_run
        memory_run(run,args.request)
    elif args.phase=='report':
        summarize(run)
    else:
        run_cases(run,args.phase)


if __name__=='__main__':
    main()
