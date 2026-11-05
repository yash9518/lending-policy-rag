import pytest

from src.ingest import load_chunks
from src.retriever import HybridRetriever


@pytest.fixture(scope="session")
def retriever():
    # BM25-only keeps tests fast and offline (no embedding model download).
    return HybridRetriever(load_chunks(), use_dense=False)
