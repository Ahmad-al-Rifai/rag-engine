"""Ask the vault a question and get a sourced answer from the local LLM.

Usage: venv/bin/python ask.py "what are my rules for AI agents?"
Exit codes: 0 ok, 1 LLM unavailable, 2 invalid question.
"""
import sys
import time

import rag
from llm import LLMError


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip())
        return 2
    question = " ".join(argv)
    start = time.time()
    try:
        result = rag.answer(question)
    except ValueError as e:
        print(f"Invalid question: {e}", file=sys.stderr)
        return 2
    except LLMError as e:
        print(f"LLM unavailable: {e}", file=sys.stderr)
        return 1

    print(result.answer)
    print("\nSources:")
    for s in result.cited or []:
        print(f"  [{s.n}] {s.source} — {s.title}")
    if not result.cited:
        print("  (none cited)")
    print(f"\n({len(result.retrieved)} sections retrieved, {time.time() - start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
