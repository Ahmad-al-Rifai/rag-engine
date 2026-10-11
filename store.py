"""Shared, lazily created embedding model and ChromaDB collection.

Nothing heavy loads at import time; the first call creates the object and
later calls reuse it. Tests can pass their own collection/embedder instead.
"""
from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_or_create_collection(settings.collection_name)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured model."""
    return get_model().encode(texts, batch_size=32).tolist()
