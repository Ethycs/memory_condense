from types import SimpleNamespace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.search.native_spine_merges import neutral_key
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from tests.test_native_spine_exchanges import Backend, prepared, request
from tests.test_reused_native_spine_exchanges import copy_inputs
from tools import compile_bounded_native_spine_exchanges as compiler
from tools import compile_expanding_native_spine_exchanges as legacy
from tools import native_spine_bounded_journal as bounded
from tools import run_native_spine_full_corpus as pipeline
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


class OverlongBackend(Backend):
    def __init__(self, *, bounded_failure=None):
        super().__init__()
        self.bounded_calls = 0
        self.bounded_failure = bounded_failure
        self.model = None

    def generate(self, jobs, attempt):
        result = super().generate(jobs, attempt)
        for job, row in zip(jobs, result['rows'], strict=True):
            row['response'] = canonical_json({'summary': 'word ' * (job.max_output_tokens + 2)})
        return result

    def generate_bounded(self, jobs, variant):
        self.calls += 1
        self.bounded_calls += 1
        job, = jobs
        wire = canonical_json(bounded.messages(job, variant))
        assert 'RAW_CANARY' not in wire and 'transcript_date' not in wire
        assert '2026-01-02' not in wire
        if self.bounded_failure == 'interrupted':
            raise RuntimeError('interrupted bounded generation')
        stopped = self.bounded_failure != 'no_eos' and not (
            self.bounded_failure == 'first_no_eos' and variant == 3)
        summary = 'User discusses furniture.' if job.kind == 'user_spine' else 'Assistant suggests furniture options.'
        if self.bounded_failure == 'overlong':
            summary = 'word ' * (job.max_output_tokens + 2)
        return {'backend_sha256': self.identity_sha256,
            'rows': [{'merge_key': neutral_key(job), 'response': canonical_json({'summary': summary}),
                      'stopped': stopped}], 'elapsed_s': 0, 'raw_inputs_to_qwen': False,
            'remote_provider_calls': 0, 'timestamp_metadata_in_model_inputs': False,
            'recovery_format': bounded.FORMAT, 'variant': variant, 'max_new_tokens': 128}


def journal(root, backend, budget):
    plan, _ = publish_sealed_json(root/'preflight.json', {'backend_sha256': backend.identity_sha256})
    return bounded.BoundedJournal(root, plan, backend, budget)


@pytest.mark.parametrize('cap', [128, 512])
def test_overlong_exchange_and_parent_summaries_recover_and_replay_without_calls(tmp_path, cap):
    backend = OverlongBackend()
    job = request(role='assistant', cap=cap)
    writer = journal(tmp_path, backend, 4)
    assert writer.resolve({neutral_key(job): job})
    assert backend.calls == writer.jobs == 4 and backend.bounded_calls == 1
    assert writer.cache(job) == 'Assistant suggests furniture options.'
    replay = journal(tmp_path, backend, 0)
    replay.replay()
    assert replay.resolve({neutral_key(job): job})
    assert replay.cache(job) == writer.cache(job) and backend.calls == 4
    recovery = next(read_sealed_json(p) for p in (tmp_path/'requests').glob('*.json')
                    if 'recovery_format' in read_sealed_json(p).payload)
    assert recovery.payload['jobs'][0]['max_output_tokens'] == cap
    assert recovery.payload['messages'][0][1] == bounded.messages(job, 3)[1]


def test_recovery_respects_the_remaining_invocation_budget(tmp_path):
    backend = OverlongBackend()
    job = request()
    first = journal(tmp_path, backend, 3)
    assert first.resolve({neutral_key(job): job}) is False
    assert backend.calls == 3 and backend.bounded_calls == 0
    next_call = journal(tmp_path, backend, 1)
    next_call.replay()
    assert next_call.resolve({neutral_key(job): job})
    assert backend.calls == 4 and next_call.jobs == 1


@pytest.mark.parametrize('failure', ['no_eos', 'overlong'])
def test_invalid_bounded_responses_never_enter_the_cache(tmp_path, failure):
    backend = OverlongBackend(bounded_failure=failure)
    job = request()
    writer = journal(tmp_path, backend, 5)
    with pytest.raises(ValueError, match='bounded native abstraction exhausted'):
        writer.resolve({neutral_key(job): job})
    assert neutral_key(job) not in writer.cache.values
    assert backend.calls == 5 and backend.bounded_calls == 2
    replay = journal(tmp_path, backend, 5)
    replay.replay()
    with pytest.raises(ValueError, match='bounded native abstraction exhausted'):
        replay.resolve({neutral_key(job): job})
    assert backend.calls == 5


def test_unstopped_first_recovery_uses_only_the_second_declared_variant(tmp_path):
    backend = OverlongBackend(bounded_failure='first_no_eos')
    job = request()
    writer = journal(tmp_path, backend, 5)
    assert writer.resolve({neutral_key(job): job})
    assert backend.bounded_calls == 2 and writer.jobs == 5


def test_interrupted_recovery_cannot_be_implicitly_retried(tmp_path):
    backend = OverlongBackend(bounded_failure='interrupted')
    job = request()
    writer = journal(tmp_path, backend, 5)
    with pytest.raises(RuntimeError, match='interrupted bounded'):
        writer.resolve({neutral_key(job): job})
    calls = backend.calls
    with pytest.raises(ValueError, match='refusing an implicit retry'):
        journal(tmp_path, backend, 5).replay()
    assert backend.calls == calls


@pytest.mark.parametrize('field', ['messages', 'variant', 'max_new_tokens', 'raw_inputs_to_qwen'])
def test_bounded_response_policy_and_inputs_are_authenticated(tmp_path, field):
    backend = OverlongBackend()
    job = request()
    writer = journal(tmp_path, backend, 4)
    writer.resolve({neutral_key(job): job})
    saved = next(read_sealed_json(p) for p in (tmp_path/'requests').glob('*.json')
                 if 'recovery_format' in read_sealed_json(p).payload)
    response = read_sealed_json(tmp_path/'responses'/f'{saved.sha256}.json')
    if field == 'messages':
        saved.payload['messages'][0][0]['content'] = 'changed input'
    elif field == 'variant':
        response.payload['variant'] = 4
    elif field == 'max_new_tokens':
        response.payload['max_new_tokens'] = 256
    else:
        response.payload['raw_inputs_to_qwen'] = True
    with pytest.raises(ValueError, match='provenance changed'):
        journal(tmp_path, backend, 0).accept(saved, response)


def test_retired_partial_journal_is_reused_and_all_raw_spans_survive(prepared, monkeypatch):
    root, history = prepared
    source = copy_inputs(root, root/'old-run'/'exchanges')
    backend = OverlongBackend()
    with pytest.raises(ValueError, match='recovery exhausted'):
        legacy.execute(source, backend, 128)
    publish_sealed_json(source.parent/'stages/exchanges.started.json',
        {'pid': 42, 'create_time': 1.0, 'policy_sha256': 'old-policy'})
    publish_sealed_json(source.parent/'stages/exchanges.exit.json',
        {'returncode': 1, 'policy_sha256': 'old-policy'})
    monkeypatch.setattr(compiler.psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda: 2.0))
    frozen = {p: p.read_bytes() for p in source.rglob('*.json')}
    target = root/'new'
    assert compiler.copy_inputs(source, target).sha256 == read_sealed_json(source/'inputs.json').sha256
    calls = backend.calls
    initial = compiler.execute(target, backend, 0, previous_root=source)
    assert not initial.payload['complete_available_body_exchanges'] and backend.calls == calls
    result = compiler.execute(target, backend, 128)
    assert result.payload['complete_available_body_exchanges']
    assert backend.calls == calls+1 and backend.bounded_calls == 1
    assert all(p.read_bytes() == data for p, data in frozen.items())
    saved_calls = backend.calls
    assert compiler.execute(target, backend, 0).sha256 == result.sha256
    assert backend.calls == saved_calls
    body = read_sealed_json(target/result.payload['compiled_bodies'][0]['path'])
    sections = [SectionSummary.from_dict(e['section']) for e in body.payload['exchanges']]
    assert tuple(s for section in sections for s in section.spans) == tuple(s for a in history.atoms for s in a.spans)
    plan = SectionSummaryIndex(sections).route('furniture', max_sections=len(sections))
    packet = hydrate_section_plan(plan, load_turn=history.get_turn, max_raw_spans=128, max_context_tokens=4096)
    assert not packet.diagnostics
    evidence = [e for section in packet.sections for e in section.evidence]
    assert len(evidence) == len(history.atoms)
    for row in evidence:
        raw = history.get_turn(row.span.turn_id)
        assert row.text == raw.text[row.span.start_char:row.span.end_char]


def test_pipeline_counts_bounded_calls_toward_the_same_stage_limit(tmp_path):
    backend = OverlongBackend()
    zero, _ = publish_sealed_json(tmp_path/'zero.json', {'done': False, 'body_count': 0, 'complete_source_compilation': True})
    final, _ = publish_sealed_json(tmp_path/'final.json', {'done': True, 'body_count': 1, 'complete_source_compilation': True})
    def execute(budget):
        if not budget:
            return zero
        backend.generate_bounded([request()], 3)
        return final
    assert pipeline.complete_merges(tmp_path, backend, execute, limit=1, complete_key='done').sha256 == final.sha256
    progress = read_sealed_json(tmp_path/'pipeline-progress/0001.json')
    assert progress.payload['new_local_jobs'] == 1 and progress.payload['new_local_batches'] == 1


@pytest.mark.parametrize('changed', [False, True])
def test_bounded_parent_reuse_keeps_legacy_ancestry_and_rejects_changed_receipts(tmp_path, monkeypatch, changed):
    from tests import test_expanded_native_parent_reuse as fixture
    from tools import compile_bounded_native_spine_hierarchy as parents
    from tools import compile_expanding_native_spine_hierarchy as old_parents
    original = fixture.source_cache(tmp_path/'original', Backend())
    backend = Backend(fail=True)
    expected, receipts = parents.reusable_parents([original], backend, 'source-corpus', 'fixed-method')
    monkeypatch.setattr(fixture, 'expanded', old_parents)
    old_root = fixture.source_cache(tmp_path/'old', backend, ancestors=receipts)
    values, receipts = parents.reusable_parents([old_root], backend, 'source-corpus', 'fixed-method')
    assert values == expected
    monkeypatch.setattr(fixture, 'expanded', parents)
    if changed:
        receipts = [dict(receipts[0], merge_cache_sha256='changed')]
    new_root = fixture.source_cache(tmp_path/'bounded', backend, ancestors=receipts)
    frozen = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    if changed:
        with pytest.raises(ValueError, match='ancestry changed'):
            parents.reusable_parents([new_root], backend, 'source-corpus', 'fixed-method')
    else:
        values, receipts = parents.reusable_parents([new_root], backend, 'source-corpus', 'fixed-method')
        assert values == expected and receipts[0]['accepted_merge_count'] == 1
    assert backend.calls == 0 and all(p.read_bytes() == data for p, data in frozen.items())
