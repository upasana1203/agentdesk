"""
Run this once (and again whenever you add/change documents in
data/documents/) to build the ChromaDB knowledge base:

    python scripts/ingest.py
"""

import sys
from pathlib import Path

# allow running as `python scripts/ingest.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.vectorstore import ingest_documents  # noqa: E402


def main():
    print("Ingesting documents into ChromaDB...")
    count = ingest_documents()
    print(f"Done. Added {count} chunks to the vector store.")


if __name__ == "__main__":
    main()
