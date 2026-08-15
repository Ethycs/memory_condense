"""Tests for the pure-function analysis layer over saved eval results.

No litellm, no network — these operate on in-memory EvalRunResult objects.
"""

from memory_condense.eval.analysis import (
    ascii_curve,
    binned_scores,
    compare_runs,
    config_label,
    load_run,
    to_csv,
)
from memory_condense.eval.report import save_run_result
from memory_condense.eval.schemas import (
    ChunkerConfig,
    ConversationResult,
    EvalConfig,
    EvalRunResult,
    RetrievalConfig,
    TurnResult,
    UsageStats,
)


def _turn(index: int, score: int, tokens: int = 100, context_tokens: int = 500):
    return TurnResult(
        turn_index=index,
        user_text="u",
        actual_response="a",
        generated_response="g",
        retrieved_chunks=[],
        score=score,
        judge_reasoning="r",
        responder_usage=UsageStats(
            input_tokens=tokens, output_tokens=tokens // 10, elapsed_s=0.1, calls=1
        ),
        judge_usage=UsageStats(
            input_tokens=tokens // 2, output_tokens=5, elapsed_s=0.2, calls=1
        ),
        retrieval_s=0.01,
        context_tokens=context_tokens,
    )


def _make_run(
    convos: dict[str, list[int]],
    k: int = 10,
    tokens: int = 100,
    context_tokens: int = 500,
) -> EvalRunResult:
    conversations: list[ConversationResult] = []
    total_usage = UsageStats()
    all_scores: list[int] = []

    for filename, scores in convos.items():
        turn_results = [
            _turn(i * 2, s, tokens=tokens, context_tokens=context_tokens)
            for i, s in enumerate(scores)
        ]
        usage = UsageStats()
        for tr in turn_results:
            usage = usage + tr.responder_usage + tr.judge_usage
        total_usage = total_usage + usage
        all_scores.extend(scores)
        conversations.append(
            ConversationResult(
                filename=filename,
                num_turns=len(scores) * 2,
                turn_results=turn_results,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                scores_by_position=[float(s) for s in scores],
                usage=usage,
            )
        )

    n = len(all_scores)
    return EvalRunResult(
        config=EvalConfig(
            chunker=ChunkerConfig(min_tokens=120, max_tokens=250),
            retrieval=RetrievalConfig(k=k, ef_search=50),
        ),
        conversations=conversations,
        aggregate_mean_score=sum(all_scores) / n if n else 0.0,
        aggregate_recall_at_4=sum(1 for s in all_scores if s >= 4) / n if n else 0.0,
        run_timestamp="2026-08-14T00:00:00+00:00",
        usage=total_usage,
        total_elapsed_s=12.5,
        mean_context_tokens=float(context_tokens),
        tokens_per_scored_turn=total_usage.total_tokens / n if n else 0.0,
    )


# --- binned_scores -----------------------------------------------------------


def test_binned_scores_known_list():
    run = _make_run({"a.txt": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
    bins = binned_scores(run, bins=5)

    assert len(bins) == 5
    assert [b.bin_index for b in bins] == [0, 1, 2, 3, 4]
    assert [b.n for b in bins] == [2, 2, 2, 2, 2]
    assert [b.mean_score for b in bins] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_binned_scores_aggregates_across_conversations():
    # Both conversations are binned relative to their own length, so a short
    # and a long conversation each contribute to every bin.
    run = _make_run({"a.txt": [1, 2, 3, 4], "b.txt": [5, 5, 5, 5]})
    bins = binned_scores(run, bins=4)

    assert [b.n for b in bins] == [2, 2, 2, 2]
    assert [b.mean_score for b in bins] == [3.0, 3.5, 4.0, 4.5]


def test_binned_scores_empty_bins():
    run = _make_run({"a.txt": [3, 4]})
    bins = binned_scores(run, bins=5)

    assert len(bins) == 5
    assert sum(b.n for b in bins) == 2
    empty = [b for b in bins if b.n == 0]
    assert all(b.mean_score == 0.0 for b in empty)


def test_binned_scores_no_conversations():
    run = _make_run({})
    bins = binned_scores(run, bins=3)
    assert [b.n for b in bins] == [0, 0, 0]


# --- compare_runs ------------------------------------------------------------


def test_compare_runs_deltas():
    baseline = _make_run({"a.txt": [1, 1, 2, 2]}, k=0, tokens=100, context_tokens=200)
    treatment = _make_run({"a.txt": [3, 3, 5, 5]}, k=10, tokens=200, context_tokens=800)

    report = compare_runs(baseline, treatment, bins=2)

    assert report.baseline_mean_score == 1.5
    assert report.treatment_mean_score == 4.0
    assert report.delta_mean_score == 2.5

    assert report.baseline_recall_at_4 == 0.0
    assert report.treatment_recall_at_4 == 0.5
    assert report.delta_recall_at_4 == 0.5

    # Per-bin deltas
    assert len(report.bin_deltas) == 2
    assert report.bin_deltas[0].baseline_mean == 1.0
    assert report.bin_deltas[0].treatment_mean == 3.0
    assert report.bin_deltas[0].delta == 2.0
    assert report.bin_deltas[1].delta == 3.0

    # Token / latency deltas
    assert report.delta_total_tokens == (
        treatment.usage.total_tokens - baseline.usage.total_tokens
    )
    assert report.delta_total_tokens > 0
    assert report.delta_mean_context_tokens == 600.0
    assert report.delta_tokens_per_scored_turn > 0

    assert "k=0" in report.baseline_label
    assert "k=10" in report.treatment_label


def test_compare_runs_by_conversation():
    baseline = _make_run({"a.txt": [2, 2], "b.txt": [4, 4]}, k=0)
    treatment = _make_run({"a.txt": [4, 4], "b.txt": [3, 3]}, k=10)

    report = compare_runs(baseline, treatment, bins=2)

    assert len(report.by_conversation) == 2
    by_name = {cd.filename: cd for cd in report.by_conversation}
    assert by_name["a.txt"].delta == 2.0
    assert by_name["b.txt"].delta == -1.0
    assert by_name["a.txt"].baseline_n == 2
    assert by_name["a.txt"].treatment_n == 2


def test_compare_runs_handles_missing_conversation():
    baseline = _make_run({"a.txt": [3, 3]}, k=0)
    treatment = _make_run({"a.txt": [4, 4], "b.txt": [5, 5]}, k=10)

    report = compare_runs(baseline, treatment)

    names = [cd.filename for cd in report.by_conversation]
    assert names == ["a.txt", "b.txt"]
    missing = [cd for cd in report.by_conversation if cd.filename == "b.txt"][0]
    assert missing.baseline_n == 0
    assert missing.baseline_mean == 0.0
    assert missing.delta == 5.0


# --- ascii_curve -------------------------------------------------------------


def test_ascii_curve_output_shape():
    curve = ascii_curve([1.0, 2.0, 3.0, 4.0, 5.0], width=60, height=12)
    lines = curve.split("\n")

    assert len(lines) == 12
    assert all(len(line) == 60 for line in lines)
    assert "*" in curve


def test_ascii_curve_custom_dimensions():
    curve = ascii_curve([1.0, 5.0, 3.0], width=20, height=5)
    lines = curve.split("\n")
    assert len(lines) == 5
    assert all(len(line) == 20 for line in lines)


def test_ascii_curve_empty_values():
    curve = ascii_curve([], width=10, height=3)
    lines = curve.split("\n")
    assert len(lines) == 3
    assert all(line == " " * 10 for line in lines)


def test_ascii_curve_flat_series():
    curve = ascii_curve([2.0, 2.0, 2.0], width=12, height=5)
    lines = curve.split("\n")
    assert len(lines) == 5
    assert all(len(line) == 12 for line in lines)
    # A flat series lands entirely on one row
    plotted_rows = [i for i, line in enumerate(lines) if "*" in line]
    assert len(plotted_rows) == 1


def test_ascii_curve_ascending_goes_up():
    curve = ascii_curve([1.0, 5.0], width=2, height=5)
    lines = curve.split("\n")
    left_row = next(i for i, line in enumerate(lines) if line[0] == "*")
    right_row = next(i for i, line in enumerate(lines) if line[1] == "*")
    # Row 0 is the top of the plot, so a higher value has a *lower* row index.
    assert right_row < left_row


# --- to_csv ------------------------------------------------------------------


def test_to_csv_row_count():
    run = _make_run({"a.txt": [1, 2, 3], "b.txt": [4, 5]})
    csv_text = to_csv(run)
    lines = csv_text.strip().split("\n")

    assert len(lines) == 1 + 5  # header + one row per scored turn
    assert lines[0].startswith("conversation,position,turn_index,score")
    assert lines[1].startswith("a.txt,0,")
    assert lines[4].startswith("b.txt,0,")


def test_to_csv_includes_tokens_and_elapsed():
    run = _make_run({"a.txt": [4]}, tokens=100, context_tokens=777)
    csv_text = to_csv(run)
    header, row = csv_text.strip().split("\n")

    cols = header.split(",")
    values = dict(zip(cols, row.split(",")))
    assert values["score"] == "4"
    assert values["context_tokens"] == "777"
    # responder 100 in / 10 out + judge 50 in / 5 out
    assert values["input_tokens"] == "150"
    assert values["output_tokens"] == "15"
    assert values["total_tokens"] == "165"
    assert float(values["elapsed_s"]) > 0.0


def test_to_csv_empty_run():
    run = _make_run({})
    lines = to_csv(run).strip().split("\n")
    assert len(lines) == 1  # header only


# --- load_run ----------------------------------------------------------------


def test_load_run_round_trip(tmp_path):
    run = _make_run({"a.txt": [3, 4, 5]})
    path = save_run_result(run, tmp_path)

    loaded = load_run(path)

    assert loaded.aggregate_mean_score == run.aggregate_mean_score
    assert loaded.usage.total_tokens == run.usage.total_tokens
    assert len(loaded.conversations) == 1
    assert loaded.conversations[0].scores_by_position == [3.0, 4.0, 5.0]
    assert loaded.conversations[0].turn_results[0].context_tokens == 500
    assert binned_scores(loaded, bins=3) == binned_scores(run, bins=3)


def test_config_label():
    cfg = EvalConfig(
        chunker=ChunkerConfig(min_tokens=80, max_tokens=200),
        retrieval=RetrievalConfig(k=0, ef_search=50),
    )
    assert config_label(cfg) == "chunk(80-200) k=0 ef=50"
