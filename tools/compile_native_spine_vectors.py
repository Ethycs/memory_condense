"""Reusable BGE vectors over admitted native summaries, independent of dates."""
from contextlib import closing
from pathlib import Path

import numpy as np

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from tools.assemble_native_spine_admitted import AdmittedSummaryBodies, implementation as admission_implementation
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock


def implementation():
    return {**admission_implementation(), **{name: digest(name) for name in (
        "tools/compile_native_spine_vectors.py", "src/memory_condense/modeling/embedding.py",
        "src/memory_condense/modeling/checkpoint_identity.py",
        "src/memory_condense/search/summary_semantic_index.py", "src/memory_condense/search/hot_retrieval.py",
    )}}


def prepare(store_root, root, encoder, *, reuse_roots=()):
    identity = summary_embedding_identity(encoder)
    reused = [NativeSummaryVectors(path) for path in reuse_roots]
    if any(cache.embedding_identity != identity for cache in reused):
        raise ValueError("native vector reuse requires the same encoder execution")
    with closing(AdmittedSummaryBodies(store_root)) as store:
        texts = sorted({atom["summary"] for row in store.connection.execute("SELECT body_sha256 FROM bodies")
                        for atom in store.load(row[0])}, key=lambda text: (quote_sha256(text), text))
        preflight, _ = publish_sealed_json(Path(root)/"preflight.json", {
            "summary_store_sha256": store.manifest.sha256, "sources_sha256": store.manifest.payload["sources_sha256"],
            "complete_source_compilation": store.manifest.payload["complete_source_compilation"],
            "embedding_identity": identity, "texts": texts, "unique_summary_count": len(texts),
            "maximum_summaries_per_checkpoint": 128, "summary_only_inputs": True,
            "occurrence_dates_in_model_inputs": False, "question_or_gold_inputs": False,
            "reuse_roots": [{"root": str(Path(path).resolve()), "result_sha256": cache.result.sha256}
                           for path, cache in zip(reuse_roots, reused, strict=True)],
            "implementation": implementation(),
        })
    return preflight


def checked_matrix(path, expected_sha, rows, dimension=None):
    if digest(path) != expected_sha:
        raise ValueError("native summary vector file changed")
    matrix = np.load(path, allow_pickle=False)
    if (matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != rows
            or matrix.shape[1] < 1 or (dimension is not None and matrix.shape[1] != dimension)
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, rtol=2e-5, atol=2e-5)):
        raise ValueError("native summary vectors have invalid dimensions or normalization")
    matrix.setflags(write=False)
    return matrix


class NativeSummaryVectors:
    """Read a completed immutable vector snapshot without loading an encoder."""

    def __init__(self, root):
        root = Path(root)
        self.preflight = read_sealed_json(root/"preflight.json")
        self.result = read_sealed_json(root/"result.json")
        p, r = self.preflight.payload, self.result.payload
        if (r["preflight_sha256"] != self.preflight.sha256 or p["implementation"] != implementation()
                or r["complete_prepared_vectors"] is not True or p["summary_only_inputs"] is not True):
            raise ValueError("native vector compilation binding changed")
        self.embedding_identity = p["embedding_identity"]
        self.values, observed = {}, []
        dimension = r["dimension"]
        for binding in r["batches"]:
            manifest_path = (root/binding["path"]).resolve()
            manifest_path.relative_to((root/"batches").resolve())
            manifest = read_sealed_json(manifest_path)
            b = manifest.payload
            if manifest.sha256 != binding["sha256"] or b["preflight_sha256"] != self.preflight.sha256:
                raise ValueError("native vector batch binding changed")
            start, stop = b["start"], b["stop"]
            texts = p["texts"][start:stop]
            if start != len(observed) or b["summary_sha256s"] != [quote_sha256(t) for t in texts]:
                raise ValueError("native vector batch summary population changed")
            matrix = checked_matrix(manifest_path.with_suffix(".npy"), b["matrix_sha256"], len(texts), dimension)
            for text, vector in zip(texts, matrix, strict=True):
                if text in self.values:
                    raise ValueError("native vector summaries are duplicated")
                self.values[text] = vector
            observed.extend(texts)
        if observed != p["texts"] or len(observed) != p["unique_summary_count"]:
            raise ValueError("native vector population is incomplete")

    def semantic_index(self, atomic_index):
        if any(s.child_section_ids or len(s.spans) != 1 for s in atomic_index.sections):
            raise ValueError("native vectors address original single-span summaries")
        try:
            matrix = np.stack([self.values[s.summary] for s in atomic_index.sections])
        except KeyError as error:
            raise ValueError("native index summary has no authenticated embedding") from error
        return SemanticSectionIndex(atomic_index, matrix, embedding_identity=self.embedding_identity)


def execute(root, encoder):
    root = Path(root)
    with _phase_lock(root, "native-summary-vectors"):
        preflight = read_sealed_json(root/"preflight.json")
        p = preflight.payload
        if p["implementation"] != implementation() or summary_embedding_identity(encoder) != p["embedding_identity"]:
            raise ValueError("native vector implementation or encoder changed")
        if (root/"result.json").exists():
            return NativeSummaryVectors(root).result
        reusable = {}
        for binding in p["reuse_roots"]:
            cache = NativeSummaryVectors(binding["root"])
            if cache.result.sha256 != binding["result_sha256"] or cache.embedding_identity != p["embedding_identity"]:
                raise ValueError("native vector reuse source changed")
            reusable.update(cache.values)
        texts, batches, dimension = p["texts"], [], None
        new_rows = reused_rows = 0
        for start in range(0, len(texts), p["maximum_summaries_per_checkpoint"]):
            selected = texts[start:start+p["maximum_summaries_per_checkpoint"]]
            path = root/"batches"/f"{start:06d}.json"
            if path.exists():
                batch = read_sealed_json(path)
                b = batch.payload
                if (b["preflight_sha256"] != preflight.sha256 or b["start"] != start
                        or b["stop"] != start+len(selected)
                        or b["summary_sha256s"] != [quote_sha256(t) for t in selected]):
                    raise ValueError("native vector resume summary population changed")
                matrix = checked_matrix(path.with_suffix(".npy"), b["matrix_sha256"], len(selected), dimension)
            else:
                missing = [t for t in selected if t not in reusable]
                generated = {}
                if missing:
                    matrix = np.array(encoder.embed_queries(missing), dtype=np.float32, copy=True)
                    if (summary_embedding_identity(encoder) != p["embedding_identity"] or matrix.ndim != 2
                            or matrix.shape[0] != len(missing) or not np.isfinite(matrix).all()):
                        raise ValueError("native summary encoder returned invalid rows or changed identity")
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    if np.any(norms == 0) or not np.isfinite(norms).all():
                        raise ValueError("native summary encoder returned zero or invalid norms")
                    matrix /= norms
                    generated = dict(zip(missing, matrix, strict=True))
                matrix = np.stack([generated[t] if t in generated else reusable[t] for t in selected])
                if dimension is not None and matrix.shape[1] != dimension:
                    raise ValueError("native summary encoder dimension changed")
                path.parent.mkdir(parents=True, exist_ok=True)
                # An orphan file is preserved for explicit recovery; never
                # overwrite output from an interrupted local checkpoint.
                with path.with_suffix(".npy").open("xb") as stream:
                    np.save(stream, matrix, allow_pickle=False)
                batch, _ = publish_sealed_json(path, {
                    "preflight_sha256": preflight.sha256, "start": start, "stop": start+len(selected),
                    "summary_sha256s": [quote_sha256(t) for t in selected],
                    "matrix_sha256": digest(path.with_suffix(".npy")),
                    "new_embedding_rows": len(missing), "reused_embedding_rows": len(selected)-len(missing),
                })
                new_rows += len(missing)
                reused_rows += len(selected)-len(missing)
            dimension = matrix.shape[1]
            batches.append({"path": str(path.relative_to(root)), "sha256": batch.sha256})
            print({"native_vector_rows_complete": start+len(selected), "prepared": len(texts)}, flush=True)
        result, _ = publish_sealed_json(root/"result.json", {
            "preflight_sha256": preflight.sha256, "batches": batches, "dimension": dimension,
            "complete_prepared_vectors": True, "complete_source_compilation": p["complete_source_compilation"],
            "unique_summary_count": len(texts), "raw_inputs_to_models": False,
            "remote_provider_calls": 0, "full100_target_passed": False,
        })
        print({"vector_result_sha256": result.sha256, "new_embedding_rows": new_rows,
               "reused_embedding_rows": reused_rows}, flush=True)
        return NativeSummaryVectors(root).result
