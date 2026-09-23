"""Audit a finished bounded engineering replay and report its actual outcomes."""
import argparse
import difflib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens, count_chat_prompt_token_proxy
from tools import native_spine_engineering_session as session


def stats(values):
    ordered = sorted(values)
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': ordered[max(0, int(len(ordered) * .95 + .9999) - 1)], 'max': max(values)}


def audit_gateway(root):
    """Include failed attempts in cost/timing and verify any recovery chain."""
    jobs = {}
    for path in (root / 'gateway').glob('*.request.json'):
        request = session.old.load(path)
        key = identity_sha256(request.payload)
        if path.name != key + '.request.json':
            raise ValueError('gateway request identity mismatch')
        response = session.old.load(path.with_name(key + '.response.json'))
        if response.payload['request_sha256'] != request.sha256:
            raise ValueError('gateway response binding mismatch')
        jobs[key] = (request, response)
    recovered = []
    for key, (request, response) in jobs.items():
        if not response.payload.get('error_type'):
            continue
        seen = set()
        current = key
        while jobs[current][1].payload.get('error_type'):
            if current in seen:
                raise ValueError('cyclic transport retry chain')
            seen.add(current)
            failed_request, failed_response = jobs[current]
            receipt = session.payload(root / 'transport-retries' / f'{current}.json')
            if (receipt['failed_request_sha256'] != failed_request.sha256
                    or receipt['failed_response_sha256'] != failed_response.sha256
                    or receipt['error_type'] != failed_response.payload['error_type']):
                raise ValueError('transport retry does not bind its original failure')
            next_key = receipt['retry_job_sha256']
            next_request, _ = jobs[next_key]
            before = {k: v for k, v in failed_request.payload.items() if k != 'nonce'}
            after = {k: v for k, v in next_request.payload.items() if k != 'nonce'}
            expected_nonce = {'transport_retry': receipt['retry_number'],
                              'original_nonce': jobs[receipt['original_job_sha256']][0].payload['nonce'],
                              'original_job_sha256': receipt['original_job_sha256']}
            if before != after or next_request.payload['nonce'] != expected_nonce:
                raise ValueError('transport retry changed the generation payload')
            current = next_key
        recovered.append({'failed_job_sha256': key, 'successful_job_sha256': current})
    totals = {}
    for request, response in jobs.values():
        kind = request.payload['kind']
        row = totals.setdefault(kind, {'count': 0, 'successful_count': 0, 'failed_count': 0,
                                       'total_s': 0., 'total_prompt_token_proxy': 0})
        row['count'] += 1
        row['failed_count' if response.payload.get('error_type') else 'successful_count'] += 1
        row['total_s'] += response.payload['elapsed_s']
        row['total_prompt_token_proxy'] += count_chat_prompt_token_proxy(request.payload['messages'])
    return totals, recovered


def report(root, *, incomplete=False, output_name=None):
    if output_name is not None and (not output_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in output_name)):
        raise ValueError('output name must be a plain lowercase artifact stem')
    report_name = output_name or ('partial-report' if incomplete else 'report')
    complete = session.payload(root / 'complete.json') if (root / 'complete.json').exists() else None
    if complete is None and not (incomplete and (root / 'STOP').exists()):
        raise ValueError('report requires a completed session or an explicitly stopped incomplete session')
    if incomplete and complete is not None:
        raise ValueError('completed sessions must use the full validation report')
    plan = session.payload(root / 'plan.json')
    adapter = session.old.load(root / 'read-tool-adapter.json') if (root / 'read-tool-adapter.json').exists() else None
    cycles = {a.sha256: a for p in root.glob('tool-cycle-adapter*.json') for a in (session.old.load(p),)}
    working = session.old.load(root / 'working-state-adapter.json') if (root / 'working-state-adapter.json').exists() else None
    working_retrieval = session.old.load(root / 'working-retrieval-adapter.json') if (root / 'working-retrieval-adapter.json').exists() else None
    live_adapter = session.old.load(root / 'live-session-adapter.json') if (root / 'live-session-adapter.json').exists() else None
    quote_adapter = session.old.load(root / 'quote-bound-adapter.json') if (root / 'quote-bound-adapter.json').exists() else None
    quote_recovery = session.payload(root / 'quote-bound-recovery.json') if (root / 'quote-bound-recovery.json').exists() else None
    dedup_adapter = session.payload(root / 'raw-dedup-adapter.json') if (root / 'raw-dedup-adapter.json').exists() else None
    dedup_recovery = session.payload(root / 'raw-dedup-recovery.json') if (root / 'raw-dedup-recovery.json').exists() else None
    compression_adapter = session.payload(root / 'compression-budget-adapter.json') if (root / 'compression-budget-adapter.json').exists() else None
    compression_recovery = session.payload(root / 'compression-budget-recovery.json') if (root / 'compression-budget-recovery.json').exists() else None
    final_failure = session.payload(root / 'final-memory-failure.json') if (root / 'final-memory-failure.json').exists() else None
    recoveries = [r for r in (quote_recovery, dedup_recovery, compression_recovery) if r is not None]
    failed_ingestion_s = sum(r['failed_ingestion_wall_estimate_s'] for r in recoveries)
    failed_ingestion_s += final_failure['failed_recovery_wall_estimate_s'] if final_failure else 0.
    for name, sha in plan['acceptance_sha256s'].items():
        if session.evaluation.digest(root / 'acceptance' / name) != sha:
            raise ValueError('frozen acceptance suite changed')
    events = [session.payload(p) for p in sorted((root / 'events').glob('*.json'))]
    rows = [dict(r, kind='seed') for r in plan['seed']] + events
    if complete is not None and identity_sha256(rows) != complete['history_sha256']:
        raise ValueError('final event transcript differs from completed memory')
    final_ingestion = session.payload(root / 'final-memory/ingest.json') if complete is not None else None
    if final_ingestion is not None and (
            final_ingestion['history_sha256'] != complete['history_sha256']
            or final_ingestion['history_turns'] != len(rows)
            or final_ingestion['snapshot'] != complete['snapshot']):
        raise ValueError('completed transcript differs from final ingestion')
    prefix_counts = {identity_sha256(rows[:i]): i for i in range(len(plan['seed']), len(rows) + 1)}
    tokens, retrieval, ingest, model_times, full_prefix, steps = [], [], [], [], [], []
    packet_count = span_count = 0
    incomplete_actions = []
    live_actions = []
    last_ingested_count = 0
    for step, prompt in enumerate(plan['prompts']):
        folder = root / 'steps' / f'{step:02d}'
        finished = session.payload(folder / 'complete.json') if (folder / 'complete.json').exists() else None
        if finished is None and not incomplete:
            raise ValueError('completed session is missing a completed prompt')
        if not any(row.get('kind') == 'prompt' and row.get('step') == step for row in rows):
            continue
        prompt_position = next(i for i, row in enumerate(rows)
                               if row.get('kind') == 'prompt' and row.get('step') == step)
        actions = []
        for request_path in sorted((folder / 'actions').glob('*/request.json')):
            action_folder = request_path.parent
            request = session.payload(request_path)
            if not (action_folder / 'tool.json').exists():
                if not incomplete:
                    raise ValueError('completed session contains an unfinished action')
                incomplete_actions.append({'step': step, 'action': request['action'],
                    'response_saved': (action_folder / 'response.json').exists()})
                continue
            context = session.payload(action_folder / 'context.json')
            ingestion = session.payload(action_folder / 'ingest.json')
            response = session.payload(action_folder / 'response.json')
            tool = session.payload(action_folder / 'tool.json')
            count = prefix_counts[request['history_sha256']]
            if count != prompt_position + 1 + plan.get('events_per_tool_action', 2) * request['action']:
                raise ValueError('a prior generated action or tool observation was omitted from memory')
            prefix = rows[:count]
            memory_hash = request.get('memory_history_sha256', request['history_sha256'])
            memory_count = prefix_counts[memory_hash]
            if not prompt_position + 1 <= memory_count <= count:
                raise ValueError('completed prior user turns or the current prompt were omitted from memory')
            raw = {r['turn_id']: r for r in rows[:memory_count]}
            generated = rows[count]
            if tool['action'] == 'finish':
                expected = json.loads(response['content'])['message']
                if generated['kind'] != 'final' or generated['text'] != expected:
                    raise ValueError('generated final reply was not retained exactly')
            else:
                if generated['kind'] != 'action' or generated['text'] != response['content']:
                    raise ValueError('generated action was not retained exactly')
                observation = rows[count + 1]
                if observation['kind'] != 'tool' or observation['role'] != 'system':
                    raise ValueError('tool observation lost its distinct role')
                if ('stored_observation_sha256' in tool and
                        quote_sha256(observation['text']) != tool['stored_observation_sha256']):
                    raise ValueError('tool observation changed before ingestion')
                if plan.get('events_per_tool_action') == 3:
                    activity = rows[count + 2]
                    action = {'action': 'protocol_error'} if tool['action'] == 'protocol_error' else json.loads(response['content'])
                    expected_activity = session.activity_receipt(action, tool['result'],
                        action_turn_id=generated['turn_id'], tool_turn_id=observation['turn_id'], ordinal=request['action'])
                    if activity['kind'] != 'activity' or activity['text'] != expected_activity:
                        raise ValueError('recent-work receipt differs from its actual tool execution')
            if (context['history_sha256'] != memory_hash
                    or ingestion['history_sha256'] != memory_hash
                    or ingestion['history_turns'] != memory_count
                    or ingestion['snapshot']['turn_count'] != memory_count):
                raise ValueError('answer did not use the fully ingested chronological prefix')
            last_ingested_count = max(last_ingested_count, memory_count)
            if any(r.get('step', -1) > step for r in prefix):
                raise ValueError('future replay events entered the answer context')
            actual = count_chat_prompt_token_proxy(request['messages'])
            if actual != request['prompt_token_proxy'] or actual > plan['max_prompt_tokens']:
                raise ValueError('request token budget was misstated or exceeded')
            expected_system = plan['system']
            if request.get('tool_adapter_sha256'):
                if adapter is None or request['tool_adapter_sha256'] != adapter.sha256:
                    raise ValueError('unbound tool adapter in answer request')
                expected_system = adapter.payload['system']
            expected_count = 2
            if request.get('live_session_adapter_sha256'):
                from tools.native_spine_engineering_live_session import working_pairs
                if live_adapter is None or request['live_session_adapter_sha256'] != live_adapter.sha256:
                    raise ValueError('unbound live-session policy')
                live_messages, selected = working_pairs(prefix, budget=live_adapter.payload['working_context_tokens'])
                expected_system = live_adapter.payload['system']
                expected_count = 2 + len(live_messages)
                expected_base = [{'role': 'system', 'content': expected_system}, {'role': 'user',
                    'content': context['text'] + '\n\nCurrent user prompt:\n' + prompt['text']}]
                if request['working_pairs'] != selected or request['messages'] != expected_base + live_messages:
                    raise ValueError('working conversation differs from actual generated action/tool pairs')
                if selected and selected[0]['start_index'] > memory_count:
                    raise ValueError('working context was evicted before it entered memory')
                live_actions.append({'step': step, 'action': request['action'], 'prompt_tokens': actual,
                    'ingest_s': ingestion['elapsed_s'], 'retrieval_s': context['elapsed_s'],
                    'model_s': response['elapsed_s'], 'memory_turns': memory_count,
                    'working_pairs': len(selected), 'kind': tool['action']})
            if request.get('working_state_adapter_sha256'):
                if working is None or request['working_state_adapter_sha256'] != working.sha256:
                    raise ValueError('unbound working-state adapter')
                expected_system = working.payload['system']
                if request['action']:
                    from tools.native_spine_engineering_working_state import readable_observation
                    from tools.native_spine_engineering_tool_cycle import tool_cycle_messages
                    expected_count = 4
                    expected = tool_cycle_messages(request['messages'][:2], prefix[-3]['text'],
                                                   readable_observation(prefix[-2]['text']))
                    if request['messages'] != expected:
                        raise ValueError('readable live tool result differs from stored observation')
            if request.get('tool_cycle_adapter_sha256'):
                if request['tool_cycle_adapter_sha256'] not in cycles:
                    raise ValueError('unbound live tool cycle')
                if request['action']:
                    expected_count = 4
                    if (request['messages'][2] != {'role': 'assistant', 'content': prefix[-3]['text']}
                            or prefix[-3]['kind'] != 'action' or prefix[-2]['kind'] != 'tool'
                            or not request['messages'][3]['content'].startswith('Tool result:\n')):
                        raise ValueError('live assistant action is not the immediately preceding one')
            if len(request['messages']) != expected_count or request['messages'][0]['content'] != expected_system:
                raise ValueError('extra conversation history or system changes entered a request')
            if context['routing']['raw_reads_during_routing'] or context['routing']['query_qwen_passes']:
                raise ValueError('routing boundary violated')
            if request.get('working_retrieval_adapter_sha256'):
                from memory_condense.search.native_spine_parent_user_routing import route_from_payload
                from memory_condense.search.section_working_context import prioritize_unseen_direct_routes
                if working_retrieval is None or request['working_retrieval_adapter_sha256'] != working_retrieval.sha256:
                    raise ValueError('unbound working retrieval adapter')
                serving_policy = context['serving_policy']
                visible = sorted({sec['section']['section_id'] for name, packet in context['packets'].items()
                                  if name != 'semantic' for sec in packet['sections']})
                excluded = sorted({rows[prompt_position]['turn_id'], *(r['turn_id'] for r in prefix
                                   if r.get('kind') in ('action', 'activity'))})
                latest = next((r for r in reversed(prefix) if r.get('kind') == 'tool' and r.get('step') == step), None)
                deferred = [latest['turn_id']] if latest else []
                if (serving_policy['visible_section_ids'] != visible
                        or serving_policy['excluded_turn_ids'] != excluded
                        or serving_policy['deferred_turn_ids'] != deferred
                        or serving_policy['semantic_tokens'] != working_retrieval.payload['semantic_tokens']):
                    raise ValueError('working context exclusions or budget differ from the permitted prefix')
                route = route_from_payload(context['routing'])
                rebuilt = prioritize_unseen_direct_routes(route.baseline, route.expanded,
                    visible_section_ids=visible, excluded_turn_ids=excluded, deferred_turn_ids=deferred)
                if context['packets']['semantic']['plan'] != rebuilt.identity_payload():
                    raise ValueError('served working evidence differs from the bound summary-selected plan')
            for packet in context['packets'].values():
                packet_count += 1
                for section in packet['sections']:
                    for item in section['evidence']:
                        span = item['span']
                        original = raw[span['turn_id']]
                        text = original['text']
                        if (original['role'] != span['role'] or quote_sha256(text) != span['turn_text_sha256']
                                or text[span['start_char']:span['end_char']] != item['text']
                                or quote_sha256(item['text']) != span['span_text_sha256']):
                            raise ValueError('memory evidence is not the exact permitted raw section')
                        span_count += 1
            tokens.append(actual)
            retrieval.append(context['elapsed_s'])
            ingest.append(ingestion['elapsed_s'])
            model_times.append(response['elapsed_s'])
            full_prefix.append(sum(count_tokens(r['text']) for r in prefix))
            actions.append({'ordinal': request['action'], 'kind': tool['action'],
                            'prompt_tokens': actual, 'model_s': response['elapsed_s'],
                            'retrieval_s': context['elapsed_s'], 'ingest_s': ingestion['elapsed_s']})
        steps.append({'step': step, 'original_turn_index': prompt['original_turn_index'],
                      'prompt': prompt['text'], 'complete': finished is not None,
                      'reply': finished['message'] if finished else None, 'actions': actions})
    workspace = root / 'workspace'
    archive = subprocess.check_output(['git', 'archive', plan['starting_revision']])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        original = {m.name: stream.extractfile(m).read() for m in stream.getmembers() if m.isfile()}
    current = {p.relative_to(workspace).as_posix(): p.read_bytes() for p in workspace.rglob('*')
               if p.is_file() and not any(part in ('__pycache__', '.pytest_cache') for part in p.parts)}
    changed = sorted(name for name in original.keys() | current.keys() if original.get(name) != current.get(name))
    diff = []
    for name in changed:
        before = original.get(name, b'').decode('utf-8').splitlines(keepends=True)
        after = current.get(name, b'').decode('utf-8').splitlines(keepends=True)
        diff.extend(difflib.unified_diff(before, after, fromfile='a/' + name, tofile='b/' + name))
    patch_name = report_name + '.patch' if output_name else ('partial-candidate.patch' if incomplete else 'candidate.patch')
    (root / patch_name).write_bytes(''.join(diff).encode('utf-8'))
    gateway, recovered = audit_gateway(root)
    validation = (json.loads((root / 'validation-report.json').read_text(encoding='utf-8'))
                  if (root / 'validation-report.json').exists() else {'status': 'not_run_for_incomplete_episode'})
    supplemental = None
    if (root / 'additional-validation/result.json').exists():
        supplemental = json.loads((root / 'additional-validation/result.json').read_text(encoding='utf-8'))
        supplemental['summary'] = (root / 'additional-validation/pytest.log').read_text(encoding='utf-8').strip().splitlines()[-1]
    checkpoint = None
    if (root / 'checkpoint-after-turn-fix/result.json').exists():
        checkpoint = {
            'artifact': 'checkpoint-after-turn-fix/result.json',
            'sha256': session.evaluation.digest(root / 'checkpoint-after-turn-fix/result.json'),
            'summary': (root / 'checkpoint-after-turn-fix/pytest.log').read_text(encoding='utf-8').strip().splitlines()[-1],
            'actor_received_results': False}
    complete_prompts = sum(step['complete'] for step in steps)
    summary = {'format': 'bounded-engineering-session-result-v1',
        'status': ('complete' if complete is not None else
                   'actor_complete_memory_save_failed' if final_failure and complete_prompts == len(plan['prompts']) else 'incomplete'),
        'complete_prompts': complete_prompts, 'attempted_prompts': len(steps),
        'planned_prompts': len(plan['prompts']), 'incomplete_actions': incomplete_actions,
        'history_sha256': identity_sha256(rows),
        'action_count': len(tokens), 'stored_turns': len(rows), 'new_events': len(events),
        'stored_tool_observations': sum(e['kind'] == 'tool' for e in events),
        'seed_raw_tokens': sum(count_tokens(r['text']) for r in plan['seed']),
        'final_raw_tokens': sum(count_tokens(r['text']) for r in rows),
        'request_tokens': stats(tokens), 'retrieval_seconds': stats(retrieval),
        'ingest_seconds': stats(ingest), 'answer_generation_seconds': stats(model_times),
        'noncached_retrieval_seconds': stats([value for value in retrieval if value > 0]),
        'noncached_ingest_seconds': stats([value for value in ingest if value > 0]),
        'final_ingest_seconds': final_ingestion['elapsed_s'] if final_ingestion else None,
        'gateway_calls': gateway, 'recovered_transport_failures': recovered,
        'full_prefix_raw_token_estimate': stats(full_prefix),
        'full_context_generation_calls': 0, 'audited_packets': packet_count, 'audited_exact_spans': span_count,
        'all_actions_use_complete_chronological_memory': live_adapter is None,
        'all_actions_use_memory_plus_bounded_current_work': live_adapter is not None,
        'max_prompt_tokens': plan['max_prompt_tokens'],
        'all_generated_tool_history_in_memory': complete is not None, 'raw_inputs_to_qwen': False,
        'last_answer_ingested_turn_count': last_ingested_count,
        'events_after_last_answer_ingestion': len(rows) - last_ingested_count,
        'changed_files': changed, 'candidate_patch_sha256': session.evaluation.digest(root / patch_name),
        'continuation': plan.get('continuation'),
        'read_tool_adapter': adapter.payload if adapter else None,
        'tool_cycle_adapters': [a.payload for a in cycles.values()],
        'working_state_adapter': working.payload if working else None,
        'working_retrieval_adapter': working_retrieval.payload if working_retrieval else None,
        'live_session_adapter': live_adapter.payload if live_adapter else None,
        'live_session_actions': live_actions,
        'live_session_request_tokens': stats([a['prompt_tokens'] for a in live_actions]) if live_actions else None,
        'quote_bound_adapter': quote_adapter.payload if quote_adapter else None,
        'quote_bound_recovery': quote_recovery,
        'raw_dedup_adapter': dedup_adapter, 'raw_dedup_recovery': dedup_recovery,
        'compression_budget_adapter': compression_adapter,
        'compression_budget_recovery': compression_recovery,
        'final_memory_failure': final_failure,
        'failed_ingestion_wall_estimate_seconds': failed_ingestion_s,
        'recorded_ingestion_seconds_including_recovered_failure': sum(ingest)
            + (final_ingestion['elapsed_s'] if final_ingestion else 0.)
            + failed_ingestion_s,
        'live_session_wall_seconds_from_preparation': (
            (root / ('complete.json' if complete is not None else 'STOP')).stat().st_mtime
            - (root / 'live-session-adapter.json').stat().st_mtime if live_adapter else None),
        'raw_support_repairs': len(list((root / 'raw-support-repairs').glob('*.json'))),
        'validation': validation, 'supplemental_validation': supplemental,
        'independent_checkpoint': checkpoint, 'steps': steps,
        'limits': ['One eight-prompt feature episode, not a complete project or a 1M-token coding benchmark.',
                   'Same prior user/assistant seed as baseline; original pre-episode tool dumps absent.',
                   ('Current prompt and bounded current-user working conversation remain live; filesystem persists.' if live_adapter else
                    'Current prompt and a bounded immediate assistant/tool pair remain live; filesystem persists.'),
                   'Recent eight user leads and last completed reply are reserved through exact memory hydration.',
                   'Six compact recent-work receipts are also reserved; they are deterministic records of actual tool operations, stored and hydrated through memory.',
                   'No matched full-context control; no causal claim that attention is responsible.',
                   'Staged GPU embedding, ingestion and snapshot reopening are included separately in timing; cached actions have zero ingestion/retrieval time.',
                   'Bounded answer context does not cap the number of ingestion summary calls; new summary input tokens are reported separately. Reused seed-cache compilation is excluded.',
                   'Acceptance checks were frozen before this run but informed by the prior failed replay.',
                   'The coding runner exposes a fixed set of nine offline test modules. Benchmark tests were rejected by that tool and independently checked after the actor finished; no benchmark dataset experiment was executed.',
                   'Provider usage counters may be unreliable; token figures are local prompt estimates.']}
    if plan.get('continuation'):
        summary['limits'].append('The current-tool slot and total cap were increased during the episode; completed prompts and all prior tool events were retained, not replayed.')
    if working:
        summary['limits'].append('A later declared continuation adds an actor-written working note within the latest live action and decodes nested tool-result JSON for display; no human-authored solution note was supplied.')
    if working_retrieval:
        summary['limits'].append('Working retrieval was repaired during development: direct matches precede hierarchy context, already supplied evidence/action metadata is excluded, the current live observation is deferred, and semantic evidence has 6144 tokens within the unchanged total cap.')
    if live_adapter:
        summary['limits'].append('The live-session continuation keeps a bounded current-user working conversation. All completed user turns enter memory, and working context is ingested before eviction. It is not a per-tool ingestion experiment; inherited failed attempts remain in the totals and live-session actions are reported separately.')
    if quote_adapter:
        summary['limits'].append('An ingestion validation failure was repaired by shortening only oversized model-selected literal support quotes to source-exact prefixes. Summaries, actor context and generated candidate code were unchanged. Failed-ingestion elapsed time is reported separately and included in the ingestion total; it is not hidden in the successful-action timing distribution.')
    if dedup_adapter:
        summary['limits'].append('A duplicate raw-summary cache publication failure was repaired by compiling each pending cache key once. Existing summaries and every original source occurrence were retained. No generated candidate code or actor prompt was changed by this repair.')
    if final_failure:
        summary['limits'].append('All eight coding prompts and final validation finished, but final memory installation failed. Qwen exceeded the 128-token merge budget on three standard attempts and also failed two stricter requests. The last reopened snapshot contains 481 of 524 journaled turns; 43 final work events remain outside application memory. This is a failed lifecycle result, not a fully persisted completed session.')
    session.save(root / f'{report_name}.json', summary)
    lines = ['# Bounded engineering session replay', '',
             f"Completed {complete_prompts}/{len(plan['prompts'])} original prompts; attempted {len(steps)}; {len(tokens)} executed model responses.", '',
             'Status: ' + summary['status'] + '.', '',
             f"Request tokens: mean {statistics.mean(tokens):.1f}, max {max(tokens)}; fixed cap {plan['max_prompt_tokens']}.",
             f"Final stored raw history: {summary['final_raw_tokens']:,} tokens; {summary['stored_tool_observations']} new tool observations.",
             f"Exact evidence audit: {packet_count} packets and {span_count} spans passed.", '',
             '## Validation', '']
    for name, result in validation.items():
        if isinstance(result, dict):
            lines.append(f"- {name}: {result.get('summary')} (exit {result['exit_code']}).")
    if supplemental:
        lines.append(f"- Additional benchmark/related tests and current-API clock diagnostic: {supplemental['summary']} (exit {supplemental['exit_code']}); frozen grade unchanged.")
    if checkpoint:
        lines.append(f"- Independent checkpoint after the initial turn-based implementation: {checkpoint['summary']}; not shown to the actor.")
    lines.extend(['', '## Prompt-by-prompt replies', ''])
    for step in steps:
        lines.extend([f"### {step['step'] + 1}. {step['prompt']}", '', step['reply'] or 'Not completed.', ''])
    lines.extend(['## Limits', ''] + ['- ' + item for item in summary['limits']])
    (root / f'{report_name}.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    session.emit(phase='report_complete', complete_prompts=complete_prompts, model_responses=len(tokens),
                 max_prompt_tokens=max(tokens), raw_tokens=summary['final_raw_tokens'], changed_files=len(changed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--incomplete', action='store_true')
    parser.add_argument('--output-name')
    args = parser.parse_args()
    report(args.root.resolve(), incomplete=args.incomplete, output_name=args.output_name)
