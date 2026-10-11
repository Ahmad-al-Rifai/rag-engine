"""Inspect retrieval from the terminal.

Usage:
  venv/bin/python query.py "what are my rules"   # one question
  venv/bin/python query.py                       # interactive; 'quit' to exit
"""
import sys

from config import settings
from retrieval import retrieve


def query_vault(question: str, top_k: int = settings.default_top_k):
    try:
        hits = retrieve(question, top_k)
    except ValueError as e:
        print(f"Invalid query: {e}")
        return

    print(f"\nQuery: {question}")
    print("=" * 50)
    for i, h in enumerate(hits, 1):
        print(f"\nResult {i}")
        print(f"Source: {h.source} (chunk {h.chunk})")
        print(f"Title: {h.title}")
        print(f"Distance: {h.distance:.4f}")
        print(f"Content: {h.text[:300]}...")
        print("-" * 30)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_vault(" ".join(sys.argv[1:]))
        sys.exit(0)
    while True:
        q = input("\nEnter query (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        query_vault(q)
