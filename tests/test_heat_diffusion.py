from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from memory_condense.association_store import AssociationArtifact
from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget, ContextPacker
from memory_condense.heat_diffusion import (
    diffuse_association_heat,
    expand_heat_diffusion_results,
)
from memory_condense.schemas import Chunk, RetrievalResult


class TinyEmbedder:
    @property
    def dim(self) -> int:
        return 16

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


def _artifact() -> AssociationArtifact:
    return AssociationArtifact.create(
        model_id="Qwen/Qwen3-8B",
        checkpoint_id="heat-test",
        prefix_layers=7,
        head_layer=1,
        cav_layer=5,
        concept_names=("context", "binding"),
        head_count=2,
    )


@pytest.fixture
def heat_store(tmp_path):
    condenser = MemoryCondenser(data_dir=tmp_path / "store", embedder=TinyEmbedder())
    results: dict[str, RetrievalResult] = {}
    for name, text in {
        "a": "alpha project anchor",
        "b": "beta project anchor",
        "c": "shared corroborated answer",
        "d": "one additional related memory",
    }.items():
        turn, chunks = condenser.ingest("user", text)
        results[name] = RetrievalResult(chunk=chunks[0], turn=turn, score=1.0)
    artifact = condenser.associations.register_artifact(_artifact())
    yield condenser, artifact, results
    condenser.close()


def test_heat_is_conserved_and_multiple_paths_accumulate(heat_store):
    condenser, artifact, results = heat_store
    for source in ("a", "b"):
        condenser.associations.upsert_edge(
            results[source].chunk.chunk_id,
            results["c"].chunk.chunk_id,
            artifact.artifact_id,
            [0.9, 0.1],
            qk_score=0.9,
        )

    diffusion = diffuse_association_heat(
        [results["a"], results["b"]],
        artifact.artifact_id,
        store=condenser.associations,
        now_turn=condenser._db.current_turn(),
        hops=1,
        neighbors_per_node=2,
        max_nodes=4,
        restart_probability=0.0,
    )

    assert diffusion.total_heat == pytest.approx(1.0)
    assert len(diffusion.nodes) == 1
    assert diffusion.nodes[0].chunk_id == results["c"].chunk.chunk_id
    assert diffusion.nodes[0].heat == pytest.approx(1.0)
    assert diffusion.nodes[0].supporting_transitions == 2


def test_restart_bounds_a_cycle_without_creating_heat(heat_store):
    condenser, artifact, results = heat_store
    condenser.associations.upsert_edge(
        results["a"].chunk.chunk_id,
        results["b"].chunk.chunk_id,
        artifact.artifact_id,
        [0.8, 0.2],
        qk_score=0.8,
    )
    condenser.associations.upsert_edge(
        results["b"].chunk.chunk_id,
        results["a"].chunk.chunk_id,
        artifact.artifact_id,
        [0.7, 0.3],
        qk_score=0.7,
    )

    diffusion = diffuse_association_heat(
        [results["a"]],
        artifact.artifact_id,
        store=condenser.associations,
        now_turn=condenser._db.current_turn(),
        hops=6,
        neighbors_per_node=2,
        max_nodes=2,
        restart_probability=0.25,
    )

    assert diffusion.total_heat == pytest.approx(1.0)
    assert {node.chunk_id for node in diffusion.nodes} <= {
        results["a"].chunk.chunk_id,
        results["b"].chunk.chunk_id,
    }
    assert all(len(node.best_path) <= 7 for node in diffusion.nodes)


def test_heat_expansion_keeps_the_safe_lexical_guard(heat_store):
    condenser, artifact, results = heat_store
    condenser.associations.upsert_edge(
        results["a"].chunk.chunk_id,
        results["c"].chunk.chunk_id,
        artifact.artifact_id,
        [0.9, 0.1],
        qk_score=0.9,
    )
    protected = results["a"].model_copy(update={"lexical_score": 0.99})

    expanded = expand_heat_diffusion_results(
        [protected],
        artifact.artifact_id,
        store=condenser.associations,
        hydrate=condenser._retriever.hydrate_chunk,
        now_turn=condenser._db.current_turn(),
        k=1,
        association_slots=1,
        qk_reserve=1,
        diffusion_hops=1,
        lexical_protection_threshold=0.9,
        max_prompt_token_increase=0,
        touch=False,
    )

    assert [result.chunk.chunk_id for result in expanded] == [
        protected.chunk.chunk_id
    ]


def test_heat_expansion_can_reserve_the_ranked_qk_exploitation_slot(heat_store):
    condenser, artifact, results = heat_store
    condenser.associations.upsert_edge(
        results["a"].chunk.chunk_id,
        results["c"].chunk.chunk_id,
        artifact.artifact_id,
        [0.95, 0.1],
        qk_score=0.95,
    )
    condenser.associations.upsert_edge(
        results["a"].chunk.chunk_id,
        results["d"].chunk.chunk_id,
        artifact.artifact_id,
        [0.2, 0.1],
        qk_score=0.2,
    )

    expanded = expand_heat_diffusion_results(
        [results["a"], results["b"]],
        artifact.artifact_id,
        store=condenser.associations,
        hydrate=condenser._retriever.hydrate_chunk,
        now_turn=condenser._db.current_turn(),
        k=2,
        association_slots=1,
        qk_reserve=1,
        ranked_qk_reserve=1,
        diffusion_hops=1,
        lexical_protection_threshold=None,
        max_prompt_token_increase=None,
        touch=False,
    )

    selected = next(result for result in expanded if result.route == "qk")
    assert selected.chunk.chunk_id == results["c"].chunk.chunk_id
    assert selected.association_path == (
        results["a"].chunk.chunk_id,
        results["c"].chunk.chunk_id,
    )


def _heated_result(
    text: str,
    *,
    chunk_id: str,
    source_id: str,
    source_heat: float,
    diffusion_heat: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            turn_id=source_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=len(text.split()),
        ),
        score=diffusion_heat,
        route="heat",
        memory_source_id=source_id,
        source_heat=source_heat,
        diffusion_heat=diffusion_heat,
    )


def test_context_packer_turns_source_heat_into_weighted_exposure():
    expansions = [
        _heated_result(
            "hot " * 30,
            chunk_id="h1",
            source_id="hot",
            source_heat=0.8,
            diffusion_heat=0.5,
        ),
        _heated_result(
            "cold " * 30,
            chunk_id="c1",
            source_id="cold",
            source_heat=0.2,
            diffusion_heat=0.2,
        ),
        _heated_result(
            "hotter " * 30,
            chunk_id="h2",
            source_id="hot",
            source_heat=0.8,
            diffusion_heat=0.3,
        ),
        _heated_result(
            "colder " * 30,
            chunk_id="c2",
            source_id="cold",
            source_heat=0.2,
            diffusion_heat=0.1,
        ),
    ]
    packed = ContextPacker(
        ContextBudget(
            expansion_tokens=105,
            max_expansions=3,
            max_expansion_tokens=30,
            heat_weighted_expansions=True,
        )
    ).pack(expansions=expansions)

    assert packed.token_counts["expansions"] <= 105
    assert packed.expansion_source_token_counts["hot"] > (
        packed.expansion_source_token_counts["cold"]
    )
    assert len(packed.expansions) == 3
