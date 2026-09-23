"""Repair only escaped apostrophes inside otherwise valid summary JSON strings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search.spine_summary import SpineSummaryRequest, parse_spine_summary
from tools.build_spine_corpus_hierarchy import restore_request
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.restore_spine_parent_hierarchy_local import LocalJournal
from tools.reuse_local_spine_parent_summaries import INPUT_KEYS, verify_implementation
from tools.run_local_spine_parent_batch4 import BatchFourQwen


POLICY = {'format': 'memory-condense-summary-json-encoding-v1',
    'repair': 'remove only invalid apostrophe escapes inside JSON strings',
    'valid_json_unchanged': True, 'strict_original_summary_contract': True,
    'preserve_original_response': True, 'new_model_calls_for_encoding_repair': 0}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('encoding repair cannot resolve duplicate JSON keys')
        result[key] = value
    return result


def repair_response(response, request):
    if type(request) is not SpineSummaryRequest or type(response) is not str:
        raise TypeError('encoding repair requires a typed summary request and text')
    try:
        json.loads(response)
        return response, None
    except json.JSONDecodeError:
        pass
    output, positions = [], []
    inside, i = False, 0
    while i < len(response):
        char = response[i]
        if inside and char == '\\' and i + 1 < len(response):
            following = response[i + 1]
            if following == "'":
                output.append(following)
                positions.append(i)
            else:
                output.extend((char, following))
            i += 2
            continue
        if char == '"':
            inside = not inside
        output.append(char)
        i += 1
    if not positions:
        return response, None
    candidate = ''.join(output)
    try:
        json.loads(candidate, object_pairs_hook=unique_object)
        parse_spine_summary(candidate, request)
    except (ValueError, TypeError):
        return response, None
    return candidate, {'policy_sha256': identity_sha256(POLICY),
        'original_response_sha256': quote_sha256(response),
        'repaired_response_sha256': quote_sha256(candidate),
        'removed_backslash_positions': positions}


def repaired_row(row, job):
    result = dict(row)
    if row['job_sha256'] != job.prompt_sha256:
        raise ValueError('encoding repair changed job attribution')
    if row['stopped'] is not True:
        return result
    response, receipt = repair_response(row['response'], job)
    if receipt is not None:
        result.update(response=response, original_response=row['response'], encoding_repair=receipt)
    return result


def verify_repaired_row(row, job):
    if 'encoding_repair' not in row:
        if 'original_response' in row:
            raise ValueError('original response lacks its encoding repair receipt')
        return
    base = {k: v for k, v in row.items() if k not in ('original_response', 'encoding_repair')}
    expected = repaired_row({**base, 'response': row['original_response']}, job)
    if (row['stopped'] is not True or expected.get('encoding_repair') != row['encoding_repair']
            or expected['response'] != row['response']):
        raise ValueError('saved encoding repair does not reproduce')


class JsonBatchFourQwen(BatchFourQwen):
    def __init__(self, *args):
        super().__init__(*args)
        self.base_backend_sha256 = self.identity_sha256
        self.identity.update(encoding_policy=POLICY,
            encoding_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            base_backend_sha256=self.base_backend_sha256)
        self.identity_sha256 = identity_sha256(self.identity)

    def generate(self, jobs, attempt):
        jobs = tuple(jobs)
        result = super().generate(jobs, attempt)
        result['rows'] = [repaired_row(row, job) for row, job in zip(result['rows'], jobs, strict=True)]
        for row, job in zip(result['rows'], jobs, strict=True):
            verify_repaired_row(row, job)
        return result


def repair_snapshot(snapshot_root, root):
    if root.resolve() == snapshot_root.resolve():
        raise ValueError('encoding recovery requires a separate snapshot')
    prior = read_sealed_json(snapshot_root / 'preflight.json')
    verify_implementation(prior)
    original = read_sealed_json(snapshot_root / 'input-cache.json')
    if (original.sha256 != prior.payload['input_cache_sha256']
            or original.payload['atoms_sha256'] != prior.payload['atoms_sha256']
            or prior.payload['raw_inputs_to_qwen'] is not False):
        raise ValueError('summary snapshot binding changed')
    if original.payload['incomplete_requests_excluded']:
        raise ValueError('unacknowledged local requests cannot be implicitly retried')
    values = dict(original.payload['summaries'])
    repairs, latest = [], {}
    local_root = Path(prior.payload['local_root'])
    if prior.payload['local_preflight_sha256'] is not None:
        local = read_sealed_json(local_root / 'preflight.json')
        if local.sha256 != prior.payload['local_preflight_sha256']:
            raise ValueError('source local preflight changed')
        journal = LocalJournal(local_root, local,
            SimpleNamespace(identity_sha256=local.payload['backend_sha256']), 0)
        expected = {(r['request_sha256'], r['response_sha256'])
            for r in original.payload['completed_local_responses']}
        observed = set()
        for path in sorted((local_root / 'requests').glob('*.json')):
            request = read_sealed_json(path)
            response = read_sealed_json(local_root / 'responses' / (request.sha256 + '.json'))
            journal.accept(request, response)
            observed.add((request.sha256, response.sha256))
            for body, row in zip(request.payload['jobs'], response.payload['rows'], strict=True):
                job = restore_request(body)
                if job.prompt_sha256 in values:
                    continue
                attempt = request.payload['attempt']
                old = latest.get(job.prompt_sha256)
                if old is not None and old[0] == attempt:
                    raise ValueError('duplicate completed attempts for one missing summary')
                if old is None or attempt > old[0]:
                    latest[job.prompt_sha256] = (attempt, request, response, job, row)
        if observed != expected:
            raise ValueError('completed response population changed after snapshot')
        for sha, (attempt, request, response, job, row) in sorted(latest.items()):
            normalized = repaired_row(row, job)
            verify_repaired_row(normalized, job)
            if 'encoding_repair' not in normalized:
                raise ValueError('unresolved completed summary needs a separate recovery decision')
            values[sha] = parse_spine_summary(normalized['response'], job)
            repairs.append({'job_sha256': sha, 'attempt': attempt,
                'request_sha256': request.sha256, 'response_sha256': response.sha256,
                'row': normalized, 'summary_sha256': quote_sha256(values[sha])})
    cache, _ = publish_sealed_json(root / 'input-cache.json', {
        'atoms_sha256': prior.payload['atoms_sha256'], 'summaries': values,
        'original_cache_sha256': original.sha256, 'original_preflight_sha256': prior.sha256,
        'encoding_repairs': repairs, 'policy': POLICY, 'new_calls': 0, 'raw_inputs_to_qwen': False})
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-json-recovered-summary-snapshot-v1',
        **{k: prior.payload[k] for k in INPUT_KEYS}, 'input_cache_sha256': cache.sha256,
        'source_snapshot_root': str(snapshot_root.resolve()), 'source_snapshot_sha256': prior.sha256,
        'raw_inputs_to_qwen': False, 'implementation': {**prior.payload['implementation'],
            'tools/local_spine_json_recovery.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    return {'output_root': str(root.resolve()), 'preflight_sha256': preflight.sha256,
        'cached_jobs': len(values), 'encoding_repaired_jobs': len(repairs)}


def repair_population(source_population, root):
    source = read_sealed_json(source_population)
    if [r['offset'] for r in source.payload['records']] != list(range(0, 100, 10)):
        raise ValueError('encoding recovery requires all ten bound memories')
    records = []
    for row in source.payload['records']:
        source_root = Path(row['output_root'])
        if read_sealed_json(source_root / 'preflight.json').sha256 != row['preflight_sha256']:
            raise ValueError('source snapshot population changed')
        records.append({'offset': row['offset'],
            **repair_snapshot(source_root, root / f"offset-{row['offset']:03}")})
    artifact, _ = publish_sealed_json(root / 'population.json', {
        'format': 'memory-condense-json-recovered-summary-population-v1',
        'source_population_sha256': source.sha256, 'records': records,
        'new_calls': 0, 'raw_inputs_to_qwen': False, 'complete_namespace_population': False})
    return artifact
