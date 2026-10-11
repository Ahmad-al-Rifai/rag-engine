from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag
from config import settings
from llm import LLMError, get_provider
from retrieval import MAX_QUESTION_CHARS, MAX_TOP_K, retrieve
from store import get_collection, get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # load the embedding model once at startup, not on the first request
    yield


app = FastAPI(title="Vault RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=MAX_TOP_K)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=settings.rag_top_k, ge=1, le=MAX_TOP_K)


class SourceItem(BaseModel):
    n: int  # the [n] number used in the answer
    source: str
    title: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    cited: list[SourceItem]      # notes the answer actually cites
    retrieved: list[SourceItem]  # all notes the model was given


class ResultItem(BaseModel):
    source: str  # vault-relative path
    title: str
    folder: str
    chunk: int
    distance: float
    content: str


class QueryResponse(BaseModel):
    question: str
    results: list[ResultItem]


@app.get("/health")
def health():
    collection = get_collection()
    sources = {m.get("source") for m in collection.get(include=["metadatas"])["metadatas"]}
    return {
        "status": "ok",
        "chunks_indexed": collection.count(),
        "notes_indexed": len(sources),
        "llm": get_provider().health(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # Sync endpoint: FastAPI runs it in a worker thread, so the blocking
    # embedding + ChromaDB calls don't stall the event loop.
    try:
        hits = retrieve(req.question, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return QueryResponse(
        question=req.question.strip(),
        results=[
            ResultItem(source=h.source, title=h.title, folder=h.folder, chunk=h.chunk,
                       distance=h.distance, content=h.text[:500])
            for h in hits
        ],
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Sync endpoint (worker thread): the LLM call can take tens of seconds.
    try:
        result = rag.answer(req.question, req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    return AskResponse(
        question=result.question,
        answer=result.answer,
        cited=[SourceItem(**vars(s)) for s in result.cited],
        retrieved=[SourceItem(**vars(s)) for s in result.retrieved],
    )
