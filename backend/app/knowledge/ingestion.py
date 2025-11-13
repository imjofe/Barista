"""Menu ingestion pipeline for ChromaDB vector store."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.llm.factory import create_azure_embeddings
from app.config import Settings as AppSettings

import structlog

logger = structlog.get_logger(__name__)


def load_menu_document(menu_path: Path) -> str:
    """Load menu.md content."""
    if not menu_path.exists():
        raise FileNotFoundError(f"Menu file not found: {menu_path}")
    return menu_path.read_text(encoding="utf-8")


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of menu content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def split_menu_content(content: str) -> List[Document]:
    """Split menu content into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([content])
    return chunks


def ingest_menu_to_chroma(
    menu_path: Path,
    persist_path: str,
    collection_name: str = "barista_menu",
    settings: AppSettings | None = None,
) -> Chroma:
    """
    Ingest menu.md into ChromaDB vector store.

    Returns the initialized Chroma vector store instance.
    """
    logger.info("ingestion.start", menu_path=str(menu_path), persist_path=persist_path)

    content = load_menu_document(menu_path)
    content_hash = compute_content_hash(content)

    # Initialize embeddings first
    if settings is None:
        raise ValueError("Settings must be provided for Azure OpenAI embeddings")
    embeddings = create_azure_embeddings(settings)

    # Initialize Chroma client with consistent settings
    chroma_settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(
        path=persist_path,
        settings=chroma_settings,
    )

    # Check if collection exists and compare hash
    try:
        collection = client.get_collection(name=collection_name)
        metadata = collection.metadata or {}
        stored_hash = metadata.get("content_hash")
        if stored_hash == content_hash:
            logger.info("ingestion.skip", reason="content_unchanged")
            # Use existing collection with the same client
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            logger.info("ingestion.loaded", collection=collection_name)
            return vectorstore
        else:
            logger.info("ingestion.reindex", old_hash=stored_hash, new_hash=content_hash)
            client.delete_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist, will create it
        pass

    # Create new collection
    documents = split_menu_content(content)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,  # Use the same client instance (don't pass persist_directory when using client)
        collection_name=collection_name,
    )
    # Store content hash in collection metadata
    client.get_collection(name=collection_name).modify(
        metadata={"content_hash": content_hash}
    )
    logger.info("ingestion.complete", chunks=len(documents))

    return vectorstore

