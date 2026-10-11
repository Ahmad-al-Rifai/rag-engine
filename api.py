from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

DEFAULT_TOP_K = settings.default_top_k

app = FastAPI(title="Vault RAG API")

model = SentenceTransformer(settings.embed_model)
client = chromadb.PersistentClient(path=str(settings.chroma_path))
collection = client.get_or_create_collection(settings.collection_name)


class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class ResultItem(BaseModel):
    source: str
    filename: str
    folder: str
    distance: float
    content: str


class QueryResponse(BaseModel):
    question: str
    results: list[ResultItem]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_in_vault": collection.count()
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"]
    )

    items = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        items.append(
            ResultItem(
                source=meta.get("source", ""),
                filename=meta.get("filename", ""),
                folder=meta.get("folder", ""),
                distance=round(float(dist), 4),
                content=doc[:500]
            )
        )

    return QueryResponse(question=question, results=items)

