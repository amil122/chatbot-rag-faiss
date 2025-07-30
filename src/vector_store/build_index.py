"""
build_index.py
--------------
Loads documents and builds FAISS index.
Run once after adding/updating PDFs.
"""

import os
from dotenv import load_dotenv
from src.loaders.document_loader import load_documents, split_documents
from src.vector_store.vector_store import create_faiss_index

# Load env variables
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data")

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents(DATA_PATH)

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"Creating FAISS index with {len(chunks)} chunks...")
    create_faiss_index(chunks)

    print("FAISS index successfully created!")
