import io
import json

import pytest

from tools.assess_native_longmemeval import source_identity, stream_records


@pytest.mark.parametrize("chunk", [1, 2, 7, 32, 10000])
def test_stream_handles_nested_records_escaped_delimiters_and_unicode(chunk):
    rows = [{"question_id": "first", "history": [{"content": 'é [ { , \\" } ] 😀'}]},
            {"question_id": "last", "history": [{"content": "x" * 200}]}]
    text = " \n " + json.dumps(rows, ensure_ascii=False) + "\n\t"
    assert list(stream_records(io.StringIO(text), chunk)) == rows


@pytest.mark.parametrize("text", ['[{"id":1}', '[{"id":1},]', '[{"id":1} {"id":2}]',
                                  '[{"id":1}] {}', '{}', '[3]', '['])
def test_stream_rejects_incomplete_or_invalid_record_arrays(text):
    with pytest.raises(ValueError):
        list(stream_records(io.StringIO(text), 2))


def test_empty_array_and_source_date_identity():
    assert list(stream_records(io.StringIO(" [ ] "), 1)) == []
    turns = [{"role": "user", "text": "bought it yesterday", "created_at": "2026-09-09"}]
    original = source_identity("session", turns)
    assert original != source_identity("session", [{**turns[0], "created_at": "2026-09-10"}])
    assert original != source_identity("session", [{**turns[0], "role": "assistant"}])
