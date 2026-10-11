import os
import hashlib
import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

from config import settings

# Config (see config.py / .env.example)
VAULT_PATH = settings.vault_path
CHUNK_SIZE = settings.chunk_size

# Init
model = SentenceTransformer(settings.embed_model)
client = chromadb.PersistentClient(path=str(settings.chroma_path))
collection = client.get_or_create_collection(settings.collection_name)

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def ingest_vault():
    files = [f for f in Path(VAULT_PATH).rglob("*.md") if "Templates" not in f.parts]
    print(f"Found {len(files)} markdown files")

    for filepath in files:
        try:
            post = frontmatter.load(filepath)
            content = post.content.strip()
            metadata = dict(post.metadata)

            if not content:
                continue

            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                # Fix 1: collision-safe IDs
                doc_id = hashlib.md5(
                    f"{filepath}_{i}".encode()
                ).hexdigest()

                embedding = model.encode(chunk).tolist()

                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": str(filepath),
                        "filename": filepath.name,
                        "folder": str(filepath.parent),  # Fix 3: folder metadata
                        "chunk": i,
                        "title": str(metadata.get("title", filepath.stem)),
                        "tags": str(metadata.get("tags", [])),
                        "area": str(metadata.get("area", "")),
                    }]
                )

            print(f"Ingested: {filepath.name} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"Error on {filepath.name}: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_vault()
