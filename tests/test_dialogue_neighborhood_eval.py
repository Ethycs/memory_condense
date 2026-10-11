from tools.grade_dialogue_neighborhood import citation_labels


def test_archive_cutoff_is_not_treated_as_an_evidence_citation():
    assert citation_labels('The archive ends at T480. The fix was reported [T450].')=={'T450'}
    assert citation_labels('Earlier [T3, T51]; later [T386; T450].')=={'T3','T51','T386','T450'}
