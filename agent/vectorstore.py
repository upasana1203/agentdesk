"""
vectorstore.py
--------------
Persistent ChromaDB vector store, using a local (free, no API key)
HuggingFace sentence-transformers embedding model.

Design decision: `sentence-transformers/all-MiniLM-L6-v2` is small (~80MB),
runs on CPU in a couple hundred ms per query, and needs no API key or
per-call billing -- important for a demo project someone else may want to
run without setting up billing on an embeddings API. The tradeoff is lower
embedding quality than e.g. OpenAI's `text-embedding-3-large`; for a small,
topically-narrow document set (a handful of company docs) that tradeoff is
fine.

Chunking: RecursiveCharacterTextSplitter, 500 characters per chunk with 50
characters of overlap. This is a standard, defensible default for
short-to-medium prose documents -- big enough to keep a paragraph's context
together, small enough to keep retrieval precise and stay well within the
embedding model's input limit.
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(_BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(_BASE_DIR, "data", "documents")
COLLECTION_NAME = "agentdesk_knowledge_base"

_embeddings = None
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def get_vectorstore() -> Chroma:
    """Load (or lazily create) the persistent Chroma collection."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore


def get_retriever(k: int = 3):
    """Return a retriever for use inside tools.py's search_documents tool."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def ingest_documents(docs_dir: str = DOCS_DIR) -> int:
    """Load every .txt file in docs_dir, chunk it, embed it, and add it to
    the persistent Chroma collection. Returns the number of chunks added.

    Safe to re-run: if you re-ingest the same files you will get duplicate
    chunks (Chroma does not dedupe by content here) -- for a demo project
    that's an acceptable simplicity tradeoff; a production ingester would
    hash file contents and use deterministic IDs to make this idempotent.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    file_paths = [
        os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.lower().endswith(".txt")
    ]
    if not file_paths:
        raise ValueError(f"No .txt documents found in {docs_dir}")

    raw_docs = []
    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def collection_is_empty() -> bool:
    store = get_vectorstore()
    return store._collection.count() == 0
