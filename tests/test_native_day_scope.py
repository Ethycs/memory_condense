from tools.audit_native_day_scope import same_day_or_earlier


def test_inclusive_day_keeps_later_same_day_and_excludes_next_day():
    asked = "2023-05-21T00:01:00+00:00"
    assert same_day_or_earlier("2023-05-21T23:59:00+00:00", asked)
    assert same_day_or_earlier("2023-05-20T23:59:00+00:00", asked)
    assert not same_day_or_earlier("2023-05-22T00:00:00+00:00", asked)
