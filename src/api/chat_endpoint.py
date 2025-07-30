"""
chat_endpoint.py
----------------
FastAPI endpoint for RAG chatbot:
- POST /chat
- Input: JSON with "query"
- Output: JSON with "answer" and "sources"
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

from schemas.chat_model import ChatRequest, ChatResponse
from src.llm.llm_chain import generate_answer



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot using FAISS + OpenRouter",
    version="1.0.0"
)

# Enable CORS (customize for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint:
    - Accepts user query
    - Passes to RAG pipeline (FAISS + OpenRouter LLM)
    - Returns generated answer + source document filenames
    """
    try:
        logger.info(f"Received query: {request.query[:50]}...")  # Log first 50 chars
        result = generate_answer(request.query)

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.chat_endpoint:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )


