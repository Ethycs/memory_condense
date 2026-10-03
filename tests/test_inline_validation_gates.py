from tools.evaluate_chat_io_local100 import obvious_problem
from tools.run_inline_validation_battery import qa_gate


def row(**changes):
    return dict(prediction='answer',rendered={'text':'evidence'},
                inline_memory_status='accepted',backlog_completed_exchanges=2,**changes)


def test_single_fallback_does_not_abort_but_sustained_failure_does():
    good=row()
    failed=dict(good,inline_memory_status='fallback')
    assert obvious_problem([good,failed]) is None
    assert 'Five consecutive' in obvious_problem([good]+[failed]*5)


def test_empty_evidence_and_stalled_ingestion_stop():
    assert obvious_problem([dict(row(),rendered={'text':''})])
    assert obvious_problem([dict(row(),backlog_completed_exchanges=13)]*5)


def test_ordinary_misses_do_not_stop_campaign():
    result=dict(invalid_grades=0,support_complete=100,correct=88,inline_memory={'accepted':99})
    assert qa_gate(result) is None
    assert qa_gate(dict(result,correct=74))
    assert qa_gate(dict(result,support_complete=89))


def test_sealed_source_is_used_for_live_grades_and_audit():
    # Both operations must use the run's selected source rather than history01.
    import inspect
    from tools import evaluate_chat_io_local100 as runner
    for method in (runner.live,runner.audit):
        source=inspect.getsource(method)
        assert "Path(plan['source'])" in source
        assert "SOURCE/" not in source
