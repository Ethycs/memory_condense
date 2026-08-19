from __future__ import annotations

import re
import sys
import zlib

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import DiscourseArtifact
from memory_condense.domain.schemas import Chunk
from memory_condense.search.episodes import (
    EpisodeBuilder,
    EpisodeRetrievalPolicy,
    FixedIntervalBoundaryDetector,
)


class _DeterministicEmbedder:
    def __init__(self, dimension: int = 48) -> None:
        self._dimension = dimension

    @property
    def dim(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            vector[zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
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


def _artifact() -> DiscourseArtifact:
    return DiscourseArtifact.create(
        kind="provider-free-diffuse-fixture",
        implementation_sha256="d" * 64,
        policy={
            "episode_boundary": "fixed_interval_4",
            "linker": "explicit-cue-v1",
            "coverage": "all_current_chunks",
        },
        metadata={
            "boundary_policy_id": "fixed_interval_4",
            "scorer_id": "none",
        },
    )


def test_long_noisy_default_stack_closes_diffuse_recommendation_without_em_llm(
    tmp_path,
) -> None:
    facts = (
        "Atlas retrieval objective target is 95 percent factual accuracy.",
        "Atlas retrieval currently reaches 82 percent judged accuracy.",
        "Atlas retrieval must keep the prompt below 8000 tokens.",
        "Atlas retrieval decision selected dense top-k only.",
        "Atlas retrieval failure: dense top-k missed diffuse dependencies.",
        "Atlas retrieval depends on exact source-span provenance.",
        "Atlas retrieval measured result showed source coverage reached 100 percent.",
        "Atlas retrieval instead revised the decision to hybrid evidence closure.",
        "Atlas retrieval unresolved open issue is calibrating semantic linking.",
    )
    noise = tuple(
        f"Unrelated discussion item {index} covers catering logistics."
        for index in range(27)
    )
    positions = (0, 4, 8, 12, 16, 20, 24, 30, 35)
    rows = list(noise)
    for position, fact in zip(positions, facts, strict=True):
        rows.insert(position, fact)

    artifact = _artifact()
    with MemoryCondenser(
        data_dir=tmp_path / "diffuse-e2e",
        embedder=_DeterministicEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=False,
    ) as condenser:
        ingested = [
            condenser.ingest("user", text, source_id="engineering-thread")
            for text in rows
        ]
        chunks = tuple(item[1][0] for item in ingested)
        chunks_by_text = {chunk.text: chunk for chunk in chunks}
        required_chunk_ids = {
            chunks_by_text[text].chunk_id for text in facts
        }

        condenser.build_and_publish_discourse_episodes(
            artifact,
            tuple(chunk.chunk_id for chunk in chunks),
            builder=EpisodeBuilder(
                min_size=1,
                max_size=4,
                detector=FixedIntervalBoundaryDetector(interval=4),
            ),
            representative_limit=1,
        )
        condenser.link_and_publish_discourse(
            artifact,
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        )
        coverage = condenser.finalize_discourse_coverage(artifact.artifact_id)
        assert coverage.chunk_count == len(chunks)

        query = "How should we improve the Atlas retrieval system?"
        direct = tuple(condenser.search_hybrid(query, k=1))
        assert len(direct) == 1
        assert not required_chunk_ids <= {item.chunk.chunk_id for item in direct}

        plan = condenser.retrieve_close_discourse_evidence(
            direct,
            artifact_id=artifact.artifact_id,
            query=query,
            episode_policy=EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                previous_episodes=1,
                next_episodes=1,
            ),
        )
        replay = condenser.retrieve_close_discourse_evidence(
            direct,
            artifact_id=artifact.artifact_id,
            query=query,
            episode_policy=EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                previous_episodes=1,
                next_episodes=1,
            ),
        )

        assert plan.plan_sha256 == replay.plan_sha256
        assert plan.artifact_id == artifact.artifact_id
        assert plan.complete_claimed is True, (
            [(item.obligation_id, item.status, item.reason) for item in plan.obligation_results],
            [
                (item.kind, item.subject_id, item.exhaustive, item.detail)
                for item in plan.scope_witnesses
                if not item.exhaustive
            ],
        )
        assert {item.status for item in plan.obligation_results} == {"satisfied"}
        assert required_chunk_ids <= {item.span.chunk_id for item in plan.atoms}
        assert plan.expansion_receipt_sha256 is not None
        assert all(item.exhaustive for item in plan.scope_witnesses)

        unframed = condenser.pack_discourse_evidence(
            plan,
            max_context_tokens=4096,
        )
        base_messages = (
            {"role": "system", "content": "Answer only from cited evidence."},
            {"role": "user", "content": query},
        )
        prefix = "Evidence follows:\n"
        suffix = "\nPropose the next action and cite the evidence."
        reserve = 128
        exact_prompt = count_chat_prompt_token_proxy(
            (
                *base_messages,
                {
                    "role": "user",
                    "content": prefix + unframed.context + suffix,
                },
            )
        )
        packet = condenser.pack_discourse_evidence(
            plan,
            max_context_tokens=4096,
            base_messages=base_messages,
            evidence_prefix=prefix,
            evidence_suffix=suffix,
            max_prompt_tokens=exact_prompt + reserve,
            output_token_reserve=reserve,
        )

        assert packet.context == unframed.context
        assert required_chunk_ids <= {item.span.chunk_id for item in packet.atoms}
        assert all(fact in packet.context for fact in facts)
        assert "role=user" in packet.context
        assert "date=" in packet.context
        assert packet.receipt.prompt_token_proxy == exact_prompt
        assert packet.receipt.prompt_workspace_token_proxy == exact_prompt + reserve
        assert packet.receipt.retained_request_token_state_bytes == 0
        assert condenser.discourse.stats()["retained_request_token_state_bytes"] == 0
        assert not {
            name
            for name in sys.modules
            if name == "em_llm" or name.startswith("em_llm.")
        }
