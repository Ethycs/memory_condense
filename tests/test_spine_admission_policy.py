from tools.spine_admission_policy import method_sha256


def test_method_identity_ignores_population_receipts_but_preserves_admission_rules():
    policy = {"format": "v4", "corpus_preflight_sha256": "a", "execution_preflight_sha256": "b",
        "summary_budget_repairs_sha256": "c", "summary_use": "routing only", "summary_text_changes_allowed": "bounded compaction",
        "implementation": {"admit": "same-code"}}
    other = {**policy, "corpus_preflight_sha256": "other", "execution_preflight_sha256": "other",
             "summary_budget_repairs_sha256": "other"}
    assert method_sha256(other) == method_sha256(policy)
    for change in ({"summary_use": "answer evidence"}, {"summary_text_changes_allowed": True},
                   {"implementation": {"admit": "other-code"}}, {"format": "v3"}):
        assert method_sha256({**policy, **change}) != method_sha256(policy)
