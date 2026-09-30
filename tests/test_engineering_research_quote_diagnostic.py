from tools.diagnose_engineering_research_quote_repair import recover_exact_span


def test_line_marker_and_emphasis_mapping_returns_actual_source():
    text='[L12] Full-depth composition **passed** in 4/5;\r\n[L13] extrapolation passed in 0/5.'
    quote='Full-depth composition passed in 4/5; extrapolation passed in 0/5.'
    recovered=recover_exact_span(quote,text)
    assert recovered in text and '[L13]' in recovered and '**passed**' in recovered


def test_diagnostic_does_not_repair_changed_values_or_signs():
    assert recover_exact_span('passed in 5/5','passed in 4/5') is None
    assert recover_exact_span('value 32','value -32') is None


def test_ambiguous_normalized_span_is_not_selected():
    assert recover_exact_span('A result.','**A result.**\n`A result.`') is None
