from __future__ import annotations

from memory_condense.db import Database
from memory_condense.association_store import AssociationArtifact, AssociationStore
from memory_condense.eval.transition_replay import (
    TransitionReplayExample,
    load_compiled_transition_examples,
    run_transition_replay,
)
from memory_condense.schemas import Chunk
from memory_condense.transcript_store import TranscriptStore
from memory_condense.transition_policy import CausalTransitionPolicy, TransitionCandidate


def _candidate(destination: str, score: float, head: int) -> TransitionCandidate:
    attention = [0.0, 0.0]
    attention[head] = 1.0
    return TransitionCandidate(
        destination_id=destination,
        base_score=score,
        head_attention=tuple(attention),
        head_cav_deltas=((1.0,), (1.0,)),
    )


def test_causal_replay_learns_a_repeating_head_pattern() -> None:
    examples = []
    for group in ("train", "test"):
        for index in range(3):
            examples.append(
                TransitionReplayExample(
                    source_group=group,
                    source_turn_id=f"{group}-{index}",
                    from_role="user",
                    next_role="assistant",
                    source_cav=(0.0,),
                    cav_velocity=(1.0,),
                    next_cav=(1.0,),
                    candidates=(
                        _candidate("baseline", 0.6, 1),
                        _candidate("target", 0.5, 0),
                    ),
                    actual_destination_id="target",
                )
            )

    report = run_transition_replay(
        examples,
        train_source_groups=["train"],
        policy=CausalTransitionPolicy(transition_weight=1.0, prior_mass=0.1),
    )

    assert report.training_transitions == 3
    assert report.evaluation_transitions == 3
    assert report.baseline_recall_at_1 == 0.0
    assert report.learned_recall_at_1 == 1.0
    assert report.improved is True


def test_compiled_loader_never_uses_future_chunks_as_current_candidates(tmp_path) -> None:
    database_path = tmp_path / "memory.db"
    with Database(database_path) as database:
        transcript = TranscriptStore(database)
        turns = [transcript.append("assistant", f"turn {index}") for index in range(4)]
        chunks = []
        for index, turn in enumerate(turns):
            chunk = Chunk(
                turn_id=turn.turn_id,
                text=f"chunk {index}",
                start_char=0,
                end_char=7,
                token_count=2,
                embedding=[1.0, 0.0],
                hnsw_label=index,
            )
            chunks.append(chunk)
            database.execute(
                "INSERT INTO chunks (chunk_id, turn_id, text, start_char, end_char, "
                "token_count, embedding, hnsw_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.turn_id,
                    chunk.text,
                    0,
                    7,
                    2,
                    b"embedding",
                    index,
                ),
            )
        artifact = AssociationArtifact.create(
            model_id="teacher",
            checkpoint_id="test",
            prefix_layers=2,
            head_layer=1,
            cav_layer=1,
            concept_names=("topic",),
            head_count=2,
        )
        associations = AssociationStore(database)
        associations.register_artifact(artifact)
        for index, chunk in enumerate(chunks):
            associations.put_signature(chunk.chunk_id, artifact.artifact_id, (float(index),))
        # Turn 2 ranks old chunk 0 over old chunk 1. Turn 3 later reveals that
        # its own strongest history target was chunk 1.
        associations.upsert_edge(
            chunks[2].chunk_id,
            chunks[0].chunk_id,
            artifact.artifact_id,
            (0.2, 0.8),
            qk_score=0.8,
        )
        associations.upsert_edge(
            chunks[2].chunk_id,
            chunks[1].chunk_id,
            artifact.artifact_id,
            (0.8, 0.2),
            qk_score=0.7,
        )
        associations.upsert_edge(
            chunks[3].chunk_id,
            chunks[1].chunk_id,
            artifact.artifact_id,
            (0.9, 0.1),
            qk_score=0.9,
        )
        # A deliberately impossible future edge must be filtered from turn 2.
        associations.upsert_edge(
            chunks[2].chunk_id,
            chunks[3].chunk_id,
            artifact.artifact_id,
            (1.0, 0.0),
            qk_score=1.0,
        )
        database.commit()

    loaded_artifact, examples = load_compiled_transition_examples(
        database_path,
        legacy_source_blocks=[("conversation", 4)],
    )

    assert loaded_artifact == artifact.artifact_id
    assert len(examples) == 1
    assert examples[0].actual_destination_id == chunks[1].chunk_id
    assert chunks[3].chunk_id not in {
        candidate.destination_id for candidate in examples[0].candidates
    }
