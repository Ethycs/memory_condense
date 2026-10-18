"""Paired statistics must count regressions and retain the full denominator."""
import pytest

from tools.evaluate_chronos_packet_time import paired


def test_paired_accuracy_counts_both_directions_and_ties():
    rows = ([dict(control_correct=False, correct=True)]*4
            + [dict(control_correct=True, correct=False)]*2
            + [dict(control_correct=True, correct=True)]*3
            + [dict(control_correct=False, correct=False)])
    result = paired(rows)
    assert result['questions'] == 10
    assert result['control_correct'] == 5
    assert result['dated_correct'] == 7
    assert result['delta_pp'] == 20
    assert result['gains'] == 4 and result['losses'] == 2
    assert result['mcnemar_exact_two_sided_p'] == pytest.approx(0.6875)
    assert result['paired_normal_95ci_pp'][0] < 0 < result['paired_normal_95ci_pp'][1]


def test_paired_significance_uses_discordant_pairs_only():
    rows = [dict(control_correct=False, correct=True)]*4
    assert paired(rows)['mcnemar_exact_two_sided_p'] == 0.125
    rows.extend([dict(control_correct=True, correct=True)]*96)
    assert paired(rows)['mcnemar_exact_two_sided_p'] == 0.125
    assert paired(rows)['delta_pp'] == 4


def test_empty_or_tied_comparisons_do_not_claim_significance():
    assert paired([]) == dict(questions=0)
    result = paired([dict(control_correct=True, correct=True)])
    assert result['delta_pp'] == 0 and result['mcnemar_exact_two_sided_p'] == 1


def test_http_transport_preserves_original_wire_options(tmp_path):
    import httpx
    import json
    from tools.packet_gateway_runtime import PacketGatewayRuntime
    messages = [dict(role='user', content='What happened?')]
    def receive(request):
        assert json.loads(request.content) == dict(model='codex_sdk/gpt-5.6-sol',messages=messages,
            max_tokens=256,stream=True,stream_options={'include_usage':True},reasoning_effort='none')
        chunks = [dict(model='codex_sdk/gpt-5.6-sol',choices=[dict(index=0,delta=dict(content='Tuesday.'),finish_reason=None)]),
            dict(model='codex_sdk/gpt-5.6-sol',choices=[dict(index=0,delta={},finish_reason='stop')],
                 usage=dict(prompt_tokens=20,completion_tokens=3,total_tokens=23))]
        body = ''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)+'data: [DONE]\n\n'
        return httpx.Response(200,text=body,headers={'content-type':'text/event-stream'})
    runtime = PacketGatewayRuntime(tmp_path,gateway='https://central-dev.zt:4000/v1',
        reader_model='codex_sdk/gpt-5.6-sol',judge_model='codex_sdk/gpt-5.6-sol')
    runtime.remote = httpx.Client(base_url='https://central-dev.zt:4000/v1/',transport=httpx.MockTransport(receive))
    try:
        result = runtime.call('actor',messages,scope='test',max_tokens=256)
        assert result['content'] == 'Tuesday.' and result['finish_reason'] == 'stop'
        assert result['usage']['prompt_tokens'] == 20
    finally:
        runtime.close()
