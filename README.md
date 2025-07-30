# RAG Chatbot using OpenRouter + FAISS

A Retrieval-Augmented Generation (RAG) chatbot built with:
- **LangChain** for chaining components
- **FAISS** for vector similarity search
- **Sentence-Transformers** for embeddings
- **OpenRouter API** (Mistral 7B Instruct) as the LLM
- **FastAPI** for API service
- Optional **Streamlit** frontend

---

## **Features**
- Load and index PDF documents into FAISS vector database.
- Query chatbot with natural language questions.
- Retrieves relevant document chunks and generates answers with context.
- Sources of retrieved documents included in responses.
- Modular architecture with clear separation:
  - **vector_store/** (FAISS operations)
  - **loaders/** (PDF loading & text chunking)
  - **llm/** (LLM wrapper + retrieval chain)
  - **api/** (FastAPI endpoint)
  - **schemas/** (Request/Response models)

---

## **Project Structure**

chatbot_using_rag/
│
├── src/
│ ├── api/ # FastAPI endpoints
│ │ └── chat_endpoint.py
│ ├── llm/ # LLM integration + RAG chain
│ │ └── llm_chain.py
│ ├── vector_store/ # FAISS storage and index building
│ │ ├── build_index.py
│ │ └── vector_store.py
│ ├── loaders/ # PDF loader and text splitter
│ │ └── document_loader.py
│ ├── schemas/ # Pydantic models for API
│ │ └── chat_model.py
│ └── utils/ # Helper functions (optional)
│
├── data/ # PDF files to index
├── vectorstore/ # Generated FAISS index
├── .env # Environment variables
├── requirements.txt # Dependencies
└── README.md # Documentation



---

## **Environment Variables**

Create a `.env` file in the project root:

```env
# OpenRouter API Config
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_SITE_NAME=LocalDev

# RAG Config
DATA_PATH=data
FAISS_INDEX_PATH=vectorstore/faiss_index
SIMILARITY_METRIC=cosine
INDEX_TYPE=flat
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API Config
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

---


Setup Instructions

1. Clone Repository :
git clone https://github.com/amil122/chatbot-rag-faiss.git
cd chatbot_using_rag

2. Create Virtual Environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt


Building the Vector Store:

Place your PDFs inside the data/ folder.
Build FAISS index:
    python src/vector_store/build_index.py
This will create the FAISS index in vectorstore/faiss_index.


Running the API :
    uvicorn src.api.chat_endpoint:app --reload

The API will be available at:

    Swagger Docs: http://127.0.0.1:8000/docs
    ReDoc: http://127.0.0.1:8000/redoc

Frontend (Optional):
    streamlit run frontend/app.py
