"""
document_loader.py
-------------------
Handles loading and preprocessing of documents for RAG pipeline:
- Loads PDFs and DOCX files from the data directory.
- Splits documents into semantic chunks for vector storage.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Default configurations 
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))        
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))   
SUPPORTED_FORMATS = [".pdf", ".docx"]


def load_documents(data_path: str = "data") -> List[Document]:
    """
    Load documents (PDF, DOCX) from the given directory.

    Args:
        data_path (str): Path to directory containing documents.

    Returns:
        List[Document]: List of loaded documents (LangChain Document objects).
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found at: {data_dir.resolve()}")

    documents: List[Document] = []

    logger.info(f"Loading documents from {data_dir.resolve()}...")

    for file_path in data_dir.iterdir():
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {file_path.name}")

            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {file_path.name}")

            else:
                logger.warning(f"Skipping unsupported file format: {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split documents into smaller chunks for retrieval.

    Args:
        documents (List[Document]): Loaded documents to split.
        chunk_size (int, optional): Chunk size in tokens/characters.
        chunk_overlap (int, optional): Overlap between chunks.

    Returns:
        List[Document]: List of chunked documents.
    """
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    logger.info(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks

