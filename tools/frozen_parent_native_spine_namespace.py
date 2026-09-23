"""Read combined immutable parent caches without copying or eagerly loading all trees."""
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import sqlite3

from tools import compile_remaining_native_spine_hierarchies as compiler
from tools import prepare_native_spine_frozen_corpus as preparation
from tools.assemble_native_spine_json_recovered import JsonRecoveredSummaryBodies
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import read_sealed_json
from tools.parent_budget_native_spine_namespace import ParentBudgetNativeSpineCorpus, validate_parent_policy
from tools.prepare_native_spine_design_slice import bound


def validate_complete_result(report, plan, scope):
    r, p, s = report.payload, plan.payload, scope.payload
    validate_parent_policy(p)
    if (r['producer_format'] != compiler.FORMAT or p['producer_format'] != compiler.FORMAT
            or p['producer_implementation'] != compiler.implementation()
            or p['implementation'] != compiler.parent.implementation()
            or r['preflight_sha256'] != plan.sha256 or bound(p['scope']).sha256 != scope.sha256
            or bound(r['scope']).sha256 != scope.sha256
            or s['format'] != preparation.FORMAT or s['implementation_sha256'] != digest(preparation.__file__)
            or r['complete_available_body_hierarchies'] is not True or r['complete_native_hierarchies'] is not True
            or r['complete_source_compilation'] is not True or r['raw_inputs_to_qwen'] is not False
            or r['original_atomic_addresses_preserved'] is not True
            or r['body_count'] != r['prepared_body_count'] or r['body_count'] != s['body_count']
            or len(r['templates']) != r['body_count']
            or {t['body_sha256'] for t in r['templates']} != {b['body_sha256'] for b in s['bodies']}):
        raise ValueError('frozen corpus requires complete matching parent templates')
    expected = {b['body_sha256']: b for b in s['bodies']}
    for template in r['templates']:
        original = expected[template['body_sha256']]
        if original['parent'] is not None:
            if (template['artifact'] != original['parent']
                    or template['parent_preflight_sha256'] != original['parent_preflight_sha256']):
                raise ValueError('frozen corpus replaced an existing parent tree')
        else:
            path = Path(template['artifact']['path']).resolve()
            path.relative_to((plan.path.parent/'hierarchies').resolve())
            if template['parent_preflight_sha256'] != plan.sha256 or path.stem != template['body_sha256']:
                raise ValueError('new frozen corpus parent has a foreign producer')


class LazyParentTemplates(Mapping):
    """Bounded resident tree cache; each miss verifies the exact referenced artifact."""

    def __init__(self, rows, *, capacity=1024):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('parent cache capacity must be a positive integer')
        self.bindings = {r['body_sha256']: r for r in rows}
        if len(self.bindings) != len(rows):
            raise ValueError('duplicate frozen parent body')
        self.capacity, self.cache = capacity, OrderedDict()

    def __len__(self):
        return len(self.bindings)

    def __iter__(self):
        return iter(self.bindings)

    def __contains__(self, key):
        return key in self.bindings

    def __getitem__(self, key):
        if key not in self.cache:
            row = self.bindings[key]
            artifact = bound(row['artifact'])
            p = artifact.payload
            if (p['body_sha256'] != key or p['preflight_sha256'] != row['parent_preflight_sha256']
                    or p['raw_inputs_to_qwen'] is not False or p['original_atomic_addresses_preserved'] is not True
                    or any(p[k] != row[k] for k in ('atomic_count', 'leaf_count', 'parent_count'))):
                raise ValueError('frozen parent artifact changed its identity')
            self.cache[key] = p
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


class FrozenParentNativeSpineCorpus(ParentBudgetNativeSpineCorpus):
    """Retain the established occurrence rebinding and exact namespace materializer."""

    def __init__(self, source_root, store_root, hierarchy_report):
        self.store = self.bank = None
        self.extension = None
        self.source_root = Path(source_root).resolve()
        self.sources = read_sealed_json(self.source_root/'sources.json')
        self.report = read_sealed_json(hierarchy_report)
        plan = read_sealed_json(self.report.path.parent/'preflight.json')
        scope = bound(plan.payload['scope'])
        validate_complete_result(self.report, plan, scope)
        attention = compiler.FrozenAttention(Path(plan.payload['attention_root']))
        if (attention.result.sha256 != plan.payload['attention_result_sha256']
                or attention.method.sha256 != plan.payload['attention_method_sha256']):
            raise ValueError('frozen corpus attention producer changed')
        inputs = bound(scope.payload['inputs'])
        if (self.sources.sha256 != inputs.payload['sources_sha256']
                or self.sources.payload['question_inputs'] is not False or self.sources.payload['gold_inputs'] is not False
                or self.sources.payload['raw_inputs_to_qwen'] is not False
                or self.sources.payload['body_count'] != self.report.payload['body_count']):
            raise ValueError('frozen corpus source bank changed')
        self.templates = LazyParentTemplates(self.report.payload['templates'])
        bank_path = (self.source_root/self.sources.payload['body_bank_path']).resolve()
        bank_path.relative_to(self.source_root)
        if digest(bank_path) != self.sources.payload['body_bank_sha256']:
            raise ValueError('frozen corpus raw body bank changed')
        try:
            self.store = JsonRecoveredSummaryBodies(Path(store_root))
            if (self.store.manifest.sha256 != inputs.payload['summary_body_store_sha256']
                    or self.store.manifest.payload['sources_sha256'] != self.sources.sha256
                    or self.store.manifest.payload['complete_source_compilation'] is not True
                    or self.store.manifest.payload['all_prepared_bodies_admitted'] is not True):
                raise ValueError('frozen corpus summary store changed or is incomplete')
            self.body_ids = frozenset(row[0] for row in self.store.connection.execute('SELECT body_sha256 FROM bodies'))
            if len(self.body_ids) != self.sources.payload['body_count'] or set(self.templates) != self.body_ids:
                raise ValueError('frozen corpus summary or hierarchy body population changed')
            self.bank = sqlite3.connect(bank_path.as_uri()+'?mode=ro', uri=True)
            self.namespaces = {n['namespace_id']: n for n in self.sources.payload['namespaces']}
            if len(self.namespaces) != len(self.sources.payload['namespaces']):
                raise ValueError('duplicate frozen corpus namespace')
            self.frozen_binding = {'preflight_sha256': plan.sha256, 'scope_sha256': scope.sha256,
                'adapter_sha256': digest(__file__), 'producer_format': compiler.FORMAT}
        except Exception:
            self.close()
            raise

    def load_namespace(self, namespace_id, *, allow_partial=False):
        namespace = super().load_namespace(namespace_id, allow_partial=allow_partial)
        namespace.audit['frozen_corpus_producer'] = dict(self.frozen_binding)
        namespace.audit['hierarchy_producer_format'] = compiler.FORMAT
        return namespace
