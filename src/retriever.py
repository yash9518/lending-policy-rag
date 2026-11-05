"""Hybrid retrieval: BM25 keyword search + dense embeddings, fused with RRF.

Policy questions mix exact terms ("CIBIL", "LTV", "DPD", "FOIR") with
paraphrases ("credit score", "how much can I borrow against my house").
Keyword search handles the first, embeddings the second; Reciprocal Rank
Fusion combines both rankings without having to calibrate their scores.

The dense half is optional. If the vector index or embedding model is not
available (offline machine, no index built yet), the retriever keeps working
in BM25-only mode and reports that in ``status``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src import config
from src.ingest import Chunk, load_chunks

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "for", "to", "in", "on", "and", "or", "what",
    "which", "how", "much", "can", "i", "my", "do", "does", "with", "at", "be", "by", "it",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class RetrievedClause:
    chunk: Chunk
    score: float
    retrievers: tuple[str, ...]  # which retrievers ranked this clause

    def to_dict(self) -> dict:
        return {**self.chunk.to_dict(), "score": round(self.score, 4), "matched_by": list(self.retrievers)}


class HybridRetriever:
    def __init__(self, chunks: list[Chunk] | None = None, use_dense: bool = True):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in self.chunks])
        self.collection = None
        self.embedder = None
        self.status = "Retrieval: BM25 only"
        if use_dense:
            self._init_dense()

    def _init_dense(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            names = [getattr(c, "name", c) for c in client.list_collections()]
            if config.COLLECTION_NAME not in names:
                from src.ingest import build_vector_store

                build_vector_store()  # fresh cloud checkout: index is git-ignored
            collection = client.get_collection(config.COLLECTION_NAME)
            if collection.count() != len(self.chunks):
                from src.ingest import build_vector_store

                build_vector_store()  # policy files changed since the last build
                collection = client.get_collection(config.COLLECTION_NAME)
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
            self.collection = collection
            self.status = "Retrieval: hybrid (BM25 + dense embeddings, RRF)"
        except Exception as exc:  # pragma: no cover - depends on environment
            self.status = f"Retrieval: BM25 only (dense index unavailable: {type(exc).__name__})"

    # --- individual rankers -------------------------------------------------
    def bm25_ranking(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].id for i in ranked[:k] if scores[i] > 0]

    def dense_ranking(self, query: str, k: int) -> list[str]:
        if self.collection is None:
            return []
        vector = self.embedder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=vector, n_results=min(k, len(self.chunks)))
        return [i for i in result["ids"][0] if i in self.by_id]

    # --- fused search -------------------------------------------------------
    def search(self, query: str, k: int = config.DEFAULT_TOP_K, mode: str = "hybrid") -> list[RetrievedClause]:
        """Return the top-k clauses. ``mode`` is "hybrid", "bm25" or "dense"."""
        pool = max(k * 3, 10)
        rankings: dict[str, list[str]] = {}
        if mode in ("hybrid", "bm25"):
            rankings["bm25"] = self.bm25_ranking(query, pool)
        if mode in ("hybrid", "dense"):
            rankings["dense"] = self.dense_ranking(query, pool)

        scores: dict[str, float] = {}
        hits: dict[str, list[str]] = {}
        for name, ids in rankings.items():
            for rank, chunk_id in enumerate(ids):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank + 1)
                hits.setdefault(chunk_id, []).append(name)

        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [RetrievedClause(self.by_id[i], scores[i], tuple(hits[i])) for i in best]
