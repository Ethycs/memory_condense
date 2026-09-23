"""Authenticate reusable merge journals once, without replaying body compilers."""
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from tools import compile_native_spine_exchanges as original
from tools import compile_reused_native_spine_exchanges as reused
from tools import compile_recovered_native_spine_exchanges as recovered
from tools import compile_expanding_native_spine_exchanges as expanding
from tools import compile_bounded_native_spine_exchanges as bounded_compiler
from tools import native_spine_bounded_journal as bounded
from tools import native_recovered_merge_seed as recovery_seed
from tools import recover_native_summary_lengths as recovery
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import read_sealed_json


def merge_values(target, values):
    for key, value in values.items():
        if key in target and target[key] != value:
            raise ValueError('conflicting response-backed exchange summaries')
        target[key] = value


class NativeExchangeJournalCache:
    """Read each immutable ancestor once; model calls and body reconstruction are absent.

    Own journals are verified by the original accept/replay boundary. Historical
    ancestry receipts and recovery projections must reproduce their recorded
    cache hashes. Completed body artifacts remain a separate admission boundary.
    A cache instance belongs to one compilation process and one source bank.
    """

    def __init__(self, backend, sources_sha256):
        self.backend = backend
        self.sources_sha256 = sources_sha256
        self.loaded = {}
        self.active = set()

    def ancestors(self, bindings):
        values, attempted, receipts = {}, set(), []
        seen = set()
        for binding in bindings:
            source = Path(binding['root']).resolve()
            if source in seen:
                raise ValueError('duplicate exchange journal ancestor')
            seen.add(source)
            cached, tried, receipt = self.load(source)
            merge_values(values, cached)
            attempted.update(tried)
            receipts.append(receipt)
        if receipts != bindings:
            raise ValueError('response-backed exchange ancestry changed')
        return values, attempted, receipts

    def recovered_seed(self, root, inputs_sha):
        """Reuse the original projection validator with journal-only ancestry."""
        root = Path(root).resolve()
        plan = read_sealed_json(root/'preflight.json')
        result = read_sealed_json(root/'result.json')
        p, r = plan.payload, result.payload
        if (p['format'] != recovery.FORMAT or p['implementation_sha256'] != digest(recovery.__file__)
                or p['backend_sha256'] != self.backend.identity_sha256 or p['raw_inputs_to_qwen'] is not False
                or r['preflight_sha256'] != plan.sha256 or r['all_original_limits_met'] is not True
                or r['original_journals_unchanged'] is not True or r['raw_inputs_to_qwen'] is not False
                or r['remote_provider_calls'] != 0 or r['new_local_jobs'] != len(p['jobs'])
                or len(r['projections']) != len(p['jobs'])):
            raise ValueError('length recovery provenance changed')
        prior_root, source = Path(p['previous_root']), Path(p['source_root']).resolve()
        prior = read_sealed_json(prior_root/'preflight.json')
        failure = read_sealed_json(prior_root/'failure.json')
        source_plan = read_sealed_json(source/'preflight.json')
        inputs = read_sealed_json(source/'inputs.json')
        previous, s = prior.payload, source_plan.payload
        if (prior.sha256 != p['previous_preflight_sha256'] or failure.sha256 != p['previous_failure_sha256']
                or failure.payload['preflight_sha256'] != prior.sha256
                or source_plan.sha256 != p['source_preflight_sha256']
                or previous['source_preflight_sha256'] != source_plan.sha256
                or Path(previous['source_root']).resolve() != source
                or inputs.sha256 != inputs_sha or s['inputs_sha256'] != inputs_sha
                or inputs.payload['sources_sha256'] != self.sources_sha256
                or s['producer_format'] != reused.FORMAT or s['producer_implementation'] != reused.implementation()
                or s['backend_sha256'] != self.backend.identity_sha256
                or previous['backend_sha256'] != self.backend.identity_sha256
                or set(previous['jobs']) != set(p['jobs'])):
            raise ValueError('recovery no longer belongs to the original exchange inputs')
        files = {str(path.resolve()): read_sealed_json(path).sha256
            for directory in ('requests', 'responses') for path in (source/directory).glob('*.json')}
        if files != previous['original_files']:
            raise ValueError('stopped recovery source journal changed')
        values, attempted, _ = self.ancestors(s['reuse_roots'])
        if identity_sha256(values) != s['reused_merge_cache_sha256']:
            raise ValueError('partial exchange ancestor cache changed')
        journal = original.NeutralJournal(source, source_plan, self.backend, 0)
        journal.replay()
        merge_values(values, journal.cache.values)
        attempted.update(journal.attempted)
        projections = {row['original_merge_key']: row for row in r['projections']}
        if set(projections) != set(p['jobs']) or len(projections) != len(r['projections']):
            raise ValueError('recovery omitted or duplicated an original request')
        for key, binding in p['jobs'].items():
            if (binding['request'] != previous['jobs'][key]['original_request'] or key in values
                    or not all((key, attempt) in journal.attempted for attempt in range(3))):
                raise ValueError('recovery changed or replaced its exhausted original request')
            values[key] = recovery_seed.validate_projection(key, binding, projections[key],
                read_sealed_json(root/'requests'/f'{key}.json'),
                read_sealed_json(root/'responses'/f'{key}.json'), plan.sha256, self.backend.identity_sha256)
        if any(key not in values for key, _ in journal.attempted):
            raise ValueError('recovery journal has another unresolved job')
        receipt = {'root': str(root), 'preflight_sha256': plan.sha256, 'result_sha256': result.sha256,
            'source_preflight_sha256': source_plan.sha256, 'original_journal_sha256': identity_sha256(files),
            'accepted_merge_count': len(values), 'merge_cache_sha256': identity_sha256(values)}
        return values, attempted, receipt

    def load(self, root, *, complete=True):
        root = Path(root).resolve()
        key = (root, complete)
        if root in self.active:
            raise ValueError('cyclic exchange journal ancestry')
        if key in self.loaded:
            values, attempted, receipt = self.loaded[key]
            return dict(values), set(attempted), dict(receipt)
        self.active.add(root)
        try:
            plan = read_sealed_json(root/'preflight.json')
            inputs = read_sealed_json(root/'inputs.json')
            p, i = plan.payload, inputs.payload
            if (p['inputs_sha256'] != inputs.sha256 or p['implementation'] != original.implementation()
                    or i['implementation'] != original.implementation()
                    or i['sources_sha256'] != self.sources_sha256
                    or i['raw_text_included'] is not False or i['question_or_gold_inputs'] is not False
                    or p['backend_sha256'] != self.backend.identity_sha256 or p['raw_inputs_to_qwen'] is not False):
                raise ValueError('exchange journal source or implementation changed')
            producers = {reused.FORMAT: reused, recovered.FORMAT: recovered,
                         expanding.FORMAT: expanding, bounded_compiler.FORMAT: bounded_compiler}
            flavor = p.get('producer_format')
            if flavor is not None and (flavor not in producers or p['producer_implementation'] != producers[flavor].implementation()):
                raise ValueError('exchange journal producer changed')
            values, attempted, receipts = {}, set(), []
            if flavor == recovered.FORMAT:
                values, attempted, receipt = self.recovered_seed(p['recovery']['root'], inputs.sha256)
                if receipt != p['recovery']:
                    raise ValueError('exchange recovery receipt changed')
            elif flavor is not None:
                values, attempted, receipts = self.ancestors(p['reuse_roots'])
            if p.get('previous_journal'):
                cached, tried, receipt = bounded_compiler.previous_journal(
                    p['previous_journal']['root'], self.backend, inputs.sha256, receipts)
                if receipt != p['previous_journal']:
                    raise ValueError('retired exchange journal changed')
                merge_values(values, cached)
                attempted.update(tried)
            if flavor is not None and (len(values) != p['reused_merge_keys']
                    or identity_sha256(values) != p['reused_merge_cache_sha256']):
                raise ValueError('exchange seed cache differs from its recorded identity')
            journal = bounded.BoundedJournal(root, plan, self.backend, 0)
            journal.replay()
            merge_values(values, journal.cache.values)
            attempted.update(journal.attempted)
            result = read_sealed_json(root/'result.json') if complete else None
            if result is not None and (result.payload['preflight_sha256'] != plan.sha256
                    or result.payload['complete_available_body_exchanges'] is not True):
                raise ValueError('exchange ancestor is incomplete')
            receipt = {'root': str(root), 'preflight_sha256': plan.sha256,
                'result_sha256': result.sha256 if result else None,
                'accepted_merge_count': len(values), 'merge_cache_sha256': identity_sha256(values)}
            self.loaded[key] = (dict(values), set(attempted), dict(receipt))
            return values, attempted, receipt
        finally:
            self.active.remove(root)
