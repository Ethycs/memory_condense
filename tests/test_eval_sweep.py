from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.eval.sweep import generate_configs


def test_generate_configs_default_grids():
    base = EvalConfig(conversation_dir="/tmp/test")
    configs = generate_configs(base)
    # min_tokens [80, 120, 180] x max_tokens [200, 300, 400]
    # All 9 combos valid (180 < 200)
    # x retrieval [k=5,10,15] x [ef=50,100] = 6
    # Total: 9 * 6 = 54
    assert len(configs) == 54

    # All should have min < max
    for c in configs:
        assert c.chunker.min_tokens < c.chunker.max_tokens


def test_generate_configs_custom_grids():
    base = EvalConfig(conversation_dir="/tmp/test")
    configs = generate_configs(
        base,
        chunker_grid={"min_tokens": [100], "max_tokens": [200]},
        retrieval_grid={"k": [5], "ef_search": [50]},
    )
    assert len(configs) == 1
    assert configs[0].chunker.min_tokens == 100
    assert configs[0].chunker.max_tokens == 200
    assert configs[0].retrieval.k == 5


def test_generate_configs_filters_invalid():
    base = EvalConfig(conversation_dir="/tmp/test")
    configs = generate_configs(
        base,
        chunker_grid={"min_tokens": [300], "max_tokens": [200]},
        retrieval_grid={"k": [10], "ef_search": [50]},
    )
    # min_tokens 300 >= max_tokens 200 -> filtered out
    assert len(configs) == 0


def test_generate_configs_inherits_base():
    base = EvalConfig(
        conversation_dir="/data",
        judge_model="claude-3-5-sonnet",
        max_conversations=5,
    )
    configs = generate_configs(
        base,
        chunker_grid={"min_tokens": [100], "max_tokens": [200]},
        retrieval_grid={"k": [10], "ef_search": [50]},
    )
    assert configs[0].judge_model == "claude-3-5-sonnet"
    assert configs[0].max_conversations == 5
    assert configs[0].conversation_dir == "/data"
