from copy import deepcopy

from tools.assess_native_history_union import added_sources


def turns(text, date):
    return [{"role": "system", "text": f"Session at {date}", "created_at": date},
            {"role": "user", "text": text, "created_at": date}]


def test_same_source_alternate_sampling_does_not_duplicate_an_event_or_rewrite_m():
    m = {"trip": turns("I went camping yesterday.", "2023-05-21")}
    s = {"trip": turns("I went camping yesterday.", "2023-05-23"),
         "extra": turns("I bought a tent.", "2023-05-19")}
    before_m, before_s = deepcopy(m), deepcopy(s)
    added, duplicated, conflicts = added_sources(m, s)
    assert duplicated == ["trip"] and conflicts == [] and list(added) == ["extra"]
    assert added["extra"][0]["created_at"] == "2023-05-19"
    assert m == before_m and s == before_s


def test_same_identifier_with_different_facts_is_a_conflict_not_a_replacement():
    m = {"meeting": turns("coffee shop", "2023-05-21")}
    s = {"meeting": turns("grocery store", "2023-05-23")}
    added, duplicated, conflicts = added_sources(m, s)
    assert added == {} and duplicated == [] and conflicts == ["meeting"]
    assert m["meeting"][1]["text"] == "coffee shop"


def test_distinct_source_ids_with_identical_words_remain_distinct_occurrences():
    m = {"trip_a": turns("I went camping yesterday.", "2023-05-21")}
    s = {"trip_b": turns("I went camping yesterday.", "2023-05-23")}
    added, duplicated, conflicts = added_sources(m, s)
    assert list(added) == ["trip_b"] and duplicated == [] and conflicts == []
