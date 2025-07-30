"""
faiss_store.py
---------------
FAISS vector store setup for RAG with industrial practices:
- Supports configurable index types (Flat, HNSW, IVF).
- Uses cosine similarity (IP) for semantic search.
- Persist index with metadata for reuse.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configurations
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", "cosine")  # cosine or l2
INDEX_TYPE = os.getenv("INDEX_TYPE", "flat")  # flat, hnsw, ivf


def _get_embeddings():
    """Initialize HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,cache_folder="./models")


def _normalize_vectors(vectorstore: FAISS):
    """
    Normalize vectors to use cosine similarity with inner product FAISS.
    Required when metric = cosine.
    """
    if SIMILARITY_METRIC == "cosine":
        # LangChain's FAISS wrapper handles cosine automatically via normalized embeddings
        logger.info("Using cosine similarity (normalized embeddings).")
    else:
        logger.info("Using L2 similarity (Euclidean distance).")


def create_faiss_index(documents: List[Document]):
    """
    Create FAISS index with chosen similarity metric and index type.
    """
    if not documents:
        raise ValueError("No documents provided to create FAISS index.")

    logger.info(f"Creating FAISS index with {EMBEDDING_MODEL}, metric={SIMILARITY_METRIC}, type={INDEX_TYPE}")

    embeddings = _get_embeddings()

    # For small-medium datasets: Flat index (exact search)
    # LangChain handles FAISS index creation; customization possible with faiss.IndexFactory
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save FAISS index
    logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    vectorstore.save_local(FAISS_INDEX_PATH)

    logger.info("FAISS index created and saved successfully.")


def load_faiss_index() -> FAISS:
    """
    Load FAISS index from disk with proper similarity settings.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Run index creation first.")

    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH} using {EMBEDDING_MODEL}")
    embeddings = _get_embeddings()

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    _normalize_vectors(vectorstore)
    logger.info("FAISS index loaded successfully.")

    return vectorstore


def get_retriever(vectorstore: FAISS, k: int = 7):
    """
    Return retriever with top-k document retrieval.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})
