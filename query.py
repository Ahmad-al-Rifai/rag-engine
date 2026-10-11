import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

TOP_K = settings.default_top_k

model = SentenceTransformer(settings.embed_model)
client = chromadb.PersistentClient(path=str(settings.chroma_path))
collection = client.get_or_create_collection(settings.collection_name)

def query_vault(question: str):
    embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {question}")
    print("=" * 50)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\nResult {i+1}")
        print(f"Source: {meta['filename']}")
        print(f"Folder: {meta['folder']}")
        print(f"Distance: {dist:.4f}")
        print(f"Content: {doc[:300]}...")
        print("-" * 30)

if __name__ == "__main__":
    while True:
        q = input("\nEnter query (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        query_vault(q)
