"""FAISS embedding index management for similarity search."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..models import TokenTracker

log = logging.getLogger(__name__)

# Dimension for text-embedding-3-small
DEFAULT_DIM = 1536


class EmbeddingIndex:
    """Wraps a FAISS index with OpenAI embedding generation.

    Supports adding texts, searching by text, batch search,
    and saving/loading from disk with hash-based cache keys.
    """

    def __init__(self, client: Any = None, model: str = "text-embedding-3-small", dim: int = DEFAULT_DIM) -> None:
        self.client = client
        self.model = model
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)  # cosine similarity (normalized vectors)
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Call OpenAI embeddings API and return normalized vectors."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Batch in chunks of 2048 (OpenAI limit)
        all_vectors = []
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            response = self.client.embeddings.create(input=batch, model=self.model)
            TokenTracker.instance().add_embedding(response.usage)
            vectors = [item.embedding for item in response.data]
            all_vectors.extend(vectors)

        arr = np.array(all_vectors, dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(arr)
        return arr

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Embed texts and add to the FAISS index."""
        if not texts:
            return

        vectors = self._embed(texts)
        self.index.add(vectors)

        if ids is None:
            ids = [str(i) for i in range(len(self.ids), len(self.ids) + len(texts))]
        if metadata is None:
            metadata = [{} for _ in texts]

        self.ids.extend(ids)
        self.metadata.extend(metadata)
        self.texts.extend(texts)

        log.info("Added %d vectors to index (total: %d)", len(texts), self.index.ntotal)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Search for the top-k nearest neighbors of a query text.

        Returns list of (similarity, id, metadata) tuples.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self._embed([query])
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((
                float(score),
                self.ids[idx] if idx < len(self.ids) else str(idx),
                self.metadata[idx] if idx < len(self.metadata) else {},
            ))
        return results

    def search_batch(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[float, str, Dict[str, Any]]]]:
        """Batch search — embed all queries in one API call and search."""
        if self.index.ntotal == 0 or not queries:
            return [[] for _ in queries]

        query_vecs = self._embed(queries)
        k = min(top_k, self.index.ntotal)
        scores_batch, indices_batch = self.index.search(query_vecs, k)

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                results.append((
                    float(score),
                    self.ids[idx] if idx < len(self.ids) else str(idx),
                    self.metadata[idx] if idx < len(self.metadata) else {},
                ))
            all_results.append(results)
        return all_results

    def save(self, cache_dir: Path, hash_key: str) -> None:
        """Save the index and metadata to disk."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        index_path = cache_dir / f"{hash_key}.faiss"
        meta_path = cache_dir / f"{hash_key}.meta.pkl"

        faiss.write_index(self.index, str(index_path))
        with meta_path.open("wb") as f:
            pickle.dump({
                "ids": self.ids,
                "metadata": self.metadata,
                "texts": self.texts,
                "dim": self.dim,
                "model": self.model,
            }, f)

        log.info("Saved index to %s (%d vectors)", index_path, self.index.ntotal)

    @classmethod
    def load_cached(
        cls,
        cache_dir: Path,
        hash_key: str,
        client: Any,
        model: str,
    ) -> Optional["EmbeddingIndex"]:
        """Load a cached index from disk if it exists and matches the hash."""
        index_path = cache_dir / f"{hash_key}.faiss"
        meta_path = cache_dir / f"{hash_key}.meta.pkl"

        if not index_path.exists() or not meta_path.exists():
            return None

        try:
            faiss_index = faiss.read_index(str(index_path))
            with meta_path.open("rb") as f:
                meta = pickle.load(f)

            obj = cls(client=client, model=model, dim=meta.get("dim", DEFAULT_DIM))
            obj.index = faiss_index
            obj.ids = meta.get("ids", [])
            obj.metadata = meta.get("metadata", [])
            obj.texts = meta.get("texts", [])

            log.info("Loaded cached index from %s (%d vectors)", index_path, obj.index.ntotal)
            return obj
        except Exception as exc:
            log.warning("Failed to load cached index: %s", exc)
            return None
