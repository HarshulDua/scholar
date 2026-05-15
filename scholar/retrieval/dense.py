"""Dense retrieval using numpy brute-force cosine similarity.

Uses pre-computed embeddings.npy (200K × 384 float32) loaded at startup.
Avoids FAISS HNSW which segfaults on Windows + Python 3.13 + asyncio due
to OMP threading conflicts. Numpy matmul for 200K×384 takes ~10ms on CPU.

The FAISS index is still loaded for backward compatibility (reconstruct()
used by recsys routes), but all ANN searches use numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self) -> None:
        self._index = None          # FAISS index (for reconstruct() only)
        self._paper_ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # (N, 384) float32
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from scholar.config import settings
            # Always run on CPU: Phi-3 owns the GPU for generation.
            # Query encoding is short (1 sentence) so CPU latency is negligible.
            self._model = SentenceTransformer(
                EMBED_MODEL,
                cache_folder=str(settings.hf_home),
                device="cpu",
            )

    def load(self) -> None:
        import faiss

        from scholar.config import settings

        index_path = settings.faiss_index_path
        ids_path = str(index_path) + ".ids.json"
        emb_path = settings.embeddings_path

        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not Path(ids_path).exists():
            raise FileNotFoundError(f"Paper IDs not found at {ids_path}")

        # Load FAISS index (used only for reconstruct() in recsys routes).
        # Set OMP threads=1 as safety measure; actual search uses numpy below.
        faiss.omp_set_num_threads(1)
        self._index = faiss.read_index(str(index_path))

        with open(ids_path, "r") as f:
            self._paper_ids = json.load(f)

        # Load pre-computed embeddings for numpy-based ANN search.
        # 200K × 384 float32 = ~293 MB. Faster and more stable than FAISS HNSW
        # on Windows + Python 3.13 + asyncio (HNSW OMP threads cause segfault).
        if Path(emb_path).exists():
            print(f"Loading embeddings from {emb_path} ...")
            raw = np.load(str(emb_path)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            self._embeddings = raw / (norms + 1e-9)
            print(f"Embeddings loaded: {self._embeddings.shape}")
        else:
            print(f"Warning: embeddings.npy not found at {emb_path}; falling back to FAISS.")
            self._embeddings = None

        print("Dense retriever ready.")

    def _encode_and_normalize(self, text: str) -> np.ndarray:
        self._ensure_model()
        assert self._model is not None  # set by _ensure_model
        vec = self._model.encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _numpy_search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Brute-force cosine similarity search via numpy matmul."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded.")
        scores = self._embeddings @ query_vec          # (N,) cosine similarity
        k = min(top_k, len(self._paper_ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self._paper_ids[i], float(scores[i])) for i in top_indices]

    def _faiss_search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Fallback FAISS search (used only if embeddings.npy not available)."""
        assert self._index is not None  # only called when index is loaded
        vec_2d = query_vec.reshape(1, -1)
        k = min(top_k, len(self._paper_ids))
        distances, indices = self._index.search(vec_2d, k)
        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._paper_ids):
                continue
            results.append((self._paper_ids[idx], float(dist)))
        return results

    def query(self, query_text: str, top_k: int = 200) -> list[tuple[str, float]]:
        if self._embeddings is None and self._index is None:
            raise RuntimeError("DenseRetriever not loaded. Call load() first.")
        vec = self._encode_and_normalize(query_text)
        if self._embeddings is not None:
            return self._numpy_search(vec, top_k)
        return self._faiss_search(vec, top_k)

    def query_by_vector(self, vec: np.ndarray, top_k: int = 200) -> list[tuple[str, float]]:
        """Query using a pre-computed embedding vector (e.g. user embedding)."""
        if self._embeddings is None and self._index is None:
            raise RuntimeError("DenseRetriever not loaded. Call load() first.")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec.astype(np.float32)
        if self._embeddings is not None:
            return self._numpy_search(vec, top_k)
        return self._faiss_search(vec, top_k)
