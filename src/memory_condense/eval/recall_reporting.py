"""Human-readable reporting for recall measurement contracts."""

from __future__ import annotations

from memory_condense.eval.recall_models import RecallReport

def print_recall_report(report: RecallReport) -> None:
    """Human-readable summary. No API calls were made to produce any of this."""
    print()
    print("=" * 72)
    print("ANSWER REACHABILITY (offline — no API calls)")
    print(f"  benchmark: {report.benchmark or '(unnamed)'}")
    print(f"  mode     : {report.mode}  k={report.k}")
    print(f"  questions: {report.n_questions}")
    print("=" * 72)
    print(f"{'answer present anywhere in haystack':<34}{report.haystack_recall:>8.1%}")
    print(f"{'answer present in context':<34}{report.recall:>8.1%}")
    print(f"{'mean best token-F1':<34}{report.mean_best_f1:>8.3f}")
    if report.mode == "memory":
        print(f"{'  ...via the memory header':<34}{report.header_recall:>8.1%}")
        print(f"{'  ...via verbatim expansions':<34}{report.expansion_recall:>8.1%}")
    print()
    print(f"{'mean context tokens':<34}{report.mean_context_tokens:>8.0f}")
    print(f"{'recall per 1k tokens':<34}{report.recall_per_1k_tokens:>8.2f}")
    print("  (condensation wins by costing less, not only by recalling more —")
    print("   compare arms on this row as well as the one above)")

    if report.evidence_source_recall is not None:
        print()
        print(f"{'mean evidence-source coverage':<34}{report.evidence_source_recall:>8.1%}")
        print(f"{'questions with any evidence':<34}{report.evidence_any_source_recall:>8.1%}")
        print(f"{'questions with all evidence':<34}{report.evidence_all_source_recall:>8.1%}")
        if report.raw_evidence_source_recall is not None:
            print(
                f"{'raw graph source coverage':<34}"
                f"{report.raw_evidence_source_recall:>8.1%}"
            )
            print(
                f"{'raw graph with all evidence':<34}"
                f"{report.raw_evidence_all_source_recall:>8.1%}"
            )

    if report.answer_value_component_recall is not None:
        print()
        print(
            f"{'packed answer-value coverage':<34}"
            f"{report.answer_value_component_recall:>8.1%}"
        )
        print(
            f"{'questions with all answer values':<34}"
            f"{report.answer_value_all_component_recall:>8.1%}"
        )
        print(
            f"{'answer-value questions scored':<34}"
            f"{report.answer_value_scored_questions:>8}"
        )

    if report.survival_by_horizon:
        print()
        print("Answer still held by a non-COLD memory item, by turns ahead:")
        for turns, frac in sorted(report.survival_by_horizon.items()):
            label = "now" if turns == 0 else f"+{turns}t"
            print(f"  {label:>5} {frac:>7.1%}")

    selector_calls = [
        question
        for question in report.questions
        if (
            question.coverage_selector_inspected > 0
            or question.coverage_selector_frontier_candidates > 0
            or bool(question.coverage_selector_operator)
            or bool(question.coverage_selector_status)
            or bool(question.coverage_selector_bypass_reason)
            or bool(question.coverage_selector_fallback_reason)
            or bool(question.coverage_selector_score_provider_fallback)
        )
    ]
    if report.coverage_selector_calls:
        print()
        print(
            f"{'coverage-selector calls':<34}"
            f"{report.coverage_selector_calls:>8}"
        )
        if selector_calls:
            print(
                f"{'mean selector latency (s)':<34}"
                f"{sum(q.coverage_selector_elapsed_s for q in selector_calls) / len(selector_calls):>8.2f}"
            )
        print(
            f"{'selector bypasses':<34}"
            f"{report.coverage_selector_bypasses:>8}"
        )
        print(
            f"{'selector fallbacks':<34}"
            f"{report.coverage_selector_fallbacks:>8}"
        )
        print(
            f"{'score-provider fallbacks':<34}"
            f"{report.coverage_score_provider_fallbacks:>8}"
        )
        print(
            f"{'degraded/fallback calls':<34}"
            f"{report.coverage_degraded_calls:>8}"
        )
        if report.coverage_routed_frontier_audited_calls:
            print(
                f"{'routed frontier exhaustive':<34}"
                f"{report.coverage_routed_frontier_exhaustive_calls:>4}/"
                f"{report.coverage_routed_frontier_audited_calls:<3}"
            )
        if report.coverage_active_partition_audited_calls:
            print(
                f"{'active partition exhaustive':<34}"
                f"{report.coverage_active_partition_exhaustive_calls:>4}/"
                f"{report.coverage_active_partition_audited_calls:<3}"
            )
        print(
            f"{'selected scope structurally complete':<34}"
            f"{report.coverage_selected_scope_structurally_complete_calls:>8}"
        )
        print(
            f"{'global semantic completeness':<34}"
            f"{report.coverage_global_semantic_complete_calls:>8}"
        )
        print(
            f"{'post-coverage closures':<34}"
            f"{report.coverage_closure_calls:>8}"
        )
        if report.coverage_closure_calls:
            print(
                f"{'  selected-scope policy':<34}"
                f"{report.coverage_selected_scope_policy_closure_calls:>8}"
            )
            print(
                f"{'  globally recall-guaranteed':<34}"
                f"{report.coverage_global_recall_guaranteed_closure_calls:>8}"
            )
        if report.coverage_cardinality_deficit_calls:
            print(
                f"{'cardinality deficit calls':<34}"
                f"{report.coverage_cardinality_deficit_calls:>8}"
            )
            print(
                f"{'cardinality deficit total':<34}"
                f"{report.coverage_cardinality_deficit_total:>8}"
            )

    if report.by_category:
        print()
        print(f"{'category':<40}{'recall':>8}")
        print("-" * 48)
        for category, frac in report.by_category.items():
            print(f"{category[:40]:<40}{frac:>8.1%}")
    print()
