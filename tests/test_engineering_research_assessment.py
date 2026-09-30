from tools.assess_engineering_research_results import accounted_usage, exact_format_equivalent


def test_missing_gateway_usage_does_not_make_ingestion_free():
    result = accounted_usage({'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
                              'prompt_tokens_proxy': 321, 'content': 'A nonempty summary.'})
    assert result['prompt_tokens'] == 321
    assert result['completion_tokens'] > 0
    assert result['prompt_estimated'] and result['completion_estimated']


def test_measured_usage_is_preserved():
    result = accounted_usage({'usage': {'prompt_tokens': 300, 'completion_tokens': 17},
                              'prompt_tokens_proxy': 321, 'content': 'Summary'})
    assert result['prompt_tokens'] == 300 and result['completion_tokens'] == 17
    assert not result['prompt_estimated'] and not result['completion_estimated']


def test_format_recovery_keeps_exact_span_without_paraphrase_or_number_changes():
    original = '3. Preserve **57.23%**.\n4. Do not claim success.'
    recovered = exact_format_equivalent('Preserve 57.23%. Do not claim success.', original)
    assert recovered in original
    assert exact_format_equivalent('Preserve 57.24%.', original) is None
    assert exact_format_equivalent('Preserve 57.23%. Claim success.', original) is None


def test_math_delimiter_recovery_preserves_expression_and_exact_offsets():
    original = r'Create a leaf report \(r_v\) for every vertex.'
    assert exact_format_equivalent('Create a leaf report r_v for every vertex.', original) == original
    assert exact_format_equivalent(r'Create a leaf report \\(r_v\\) for every vertex.', original) == original
    assert exact_format_equivalent('Create a leaf report r_u for every vertex.', original) is None
