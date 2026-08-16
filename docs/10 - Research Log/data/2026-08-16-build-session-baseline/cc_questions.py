"""Probe questions over the build-session corpus.

Hand-authored by the assistant that lived the session — a stated bias:
selection favours memorable, distinctive strings, so recall here is an
UPPER estimate vs organic questions. Guards applied programmatically:
  * answer must appear verbatim (SQuAD-normalized) in >=1 ingested turn
  * answers appearing in >25 turns are dropped as ubiquitous
Phases: A early-session, B repo-review, C decay-coordinate, D measurement,
E frames/requirements. Positions spread across the timeline.
"""
QA = [
    # phase A — early session
    ("What was the commit message of cd9f423?", "Good compressor", "A"),
    ("Which merge commit brought the memory layer to main?", "f3edc91", "A"),
    ("Which commit added supersede over MCP?", "7717c3a", "A"),
    # phase B — layer_context_seg review
    ("Which repo was reviewed for useful pieces?", "layer_context_seg", "B"),
    ("Which submodule is the reference implementation?", "claude-forest-atlas", "B"),
    ("What does DHS stand for?", "Direct Hierarchical Summarization", "B"),
    ("What does HSC stand for?", "Hierarchical Semantic Contraction", "B"),
    ("Which library does the SOM strategy import?", "minisom", "B"),
    ("What dishonest literal did HSC hardcode?", "coverage_percentage=100.0", "B"),
    ("Which segmenter function enriches code blocks with prose?", "_enrich_with_adjacent_context", "B"),
    ("How does DHS pick its anchor nodes?", "PageRank", "B"),
    # phase C — decay coordinate
    ("What replaced the 300-second refractory window?", "REHEAT_ONCE_PER_TURN", "C"),
    ("What SQL expression does current_turn use?", "COALESCE(MAX(ordinal), 0)", "C"),
    ("Why is the clock MAX(ordinal) instead of COUNT(*)?", "renumber the clock backwards", "C"),
    ("What is the v4 backfill function called?", "_backfill_turn_ordinals", "C"),
    ("What index was added on turns in v4?", "idx_turns_ordinal", "C"),
    ("Which commit moved decay onto turns?", "9aea4cd", "C"),
    ("Which commit documented the coordinate finding?", "249e4bb", "C"),
    ("What is the turn half-life constant called?", "DEFAULT_HALF_LIFE_TURNS", "C"),
    ("What are the new survival horizons in turns?", "(0, 15, 30, 45)", "C"),
    ("What text filled the 60 smoke-test turns?", "unrelated chatter", "C"),
    ("How long did an important item need to reach COLD under wall-clock?", "11.75 days", "C"),
    # phase D — measurement
    ("What was the whole-transcript ceiling on conv-26?", "33.2%", "D"),
    ("What recall did span x4 get on conv-26?", "23.1%", "D"),
    ("What recall did hybrid k=10 get?", "13.1%", "D"),
    ("How much recall did dense add to the oracle union?", "0.0pp", "D"),
    ("What was the router's oracle ceiling over span plus hybrid?", "+2.0pp", "D"),
    ("How many ms does embedding a query cost?", "155.3", "D"),
    ("What recall did random k=40 get?", "5.5%", "D"),
    ("Which baseline scored worse than random?", "worse than random", "D"),
    ("What did span replicate to across 4 samples?", "23.4%", "D"),
    ("What are the two cosine similarity means showing length bias?", "0.678 vs 0.602", "D"),
    ("What are the default span token levels?", "(110, 220)", "D"),
    ("What is the chunker's cross-turn defect?", "never merges across turns", "D"),
    ("What was LoCoMo's median chunk size in tokens?", "27 tok", "D"),
    ("What was the median item age in the heat probe?", "p50=220", "D"),
    ("What line of the sweep output names the energy reorder test?", "top-5 order changed by energy", "D"),
    ("How many tokens of transcript does conv-26 have?", "13,455 tokens", "D"),
    ("What was the memory arm's mean context in tokens?", "391", "D"),
    # phase E — frames / requirements
    ("What operating envelope did the operator set?", "1-1M tokens", "E"),
    ("What per-turn cost goal did the operator state?", "linear token expense per turn", "E"),
    ("What comment marks the span-cache invalidation?", "spans are derived; new chunks change them", "E"),
    ("Which line stops the packer reaching span?", "self.search_hybrid if hybrid else self.search", "E"),
    ("What is the total ContextBudget in tokens?", "6,200", "E"),
    ("What does mean-pooling do when a span straddles topics?", "washes out once a span straddles topics", "E"),
    ("What sheaf condition does mean-pooling fail?", "fails gluing", "E"),
    ("What fraction of gold answers appear nowhere verbatim?", "two-thirds", "E"),
    ("What did the operator permit from time to time?", "expensive operations", "E"),
]
