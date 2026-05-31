import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("vault")

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
