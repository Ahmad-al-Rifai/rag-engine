"""Retrieval: question -> ranked chunks from the vault index.

Kept separate from generation so it can be inspected and tested on its own
(see query.py and eval_retrieval.py).
"""
from dataclasses import dataclass

from config import settings

MAX_TOP_K = 20
MAX_QUESTION_CHARS = 1000


@dataclass(frozen=True)
class Hit:
    source: str      # vault-relative path, e.g. "00_System/Rules/master-rules.md"
    title: str
    folder: str
    chunk: int
    distance: float  # lower = more similar
    text: str


def validate(question: str, top_k: int | None) -> tuple[str, int]:
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
    top_k = settings.default_top_k if top_k is None else top_k
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    return question, top_k


def retrieve(question: str, top_k: int | None = None, collection=None, embed=None) -> list[Hit]:
    """Return up to top_k chunks, most similar first. Raises ValueError on bad input."""
    question, top_k = validate(question, top_k)
    if collection is None or embed is None:
        from store import embed as default_embed, get_collection
        collection = collection if collection is not None else get_collection()
        embed = embed or default_embed

    available = collection.count()
    if available == 0:
        return []
    res = collection.query(
        query_embeddings=embed([question]),
        n_results=min(top_k, available),
        include=["documents", "metadatas", "distances"],
    )
    return [
        Hit(
            source=meta.get("source", ""),
            title=meta.get("title", ""),
            folder=meta.get("folder", ""),
            chunk=int(meta.get("chunk", 0)),
            distance=round(float(dist), 4),
            text=doc,
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
