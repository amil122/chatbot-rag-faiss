"""
llm_chain.py
-------------
- Uses OpenRouter API (Mistral 7B Instruct).
- Retrieves context from FAISS vector store.
- Builds a retrieval-augmented generation chain via LangChain.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv


from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from vectorstore.faiss_store import load_faiss_index, get_retriever
from langchain.llms.base import LLM
from openai import OpenAI

load_dotenv()

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------
# Environment Config
# -----------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


MODEL_NAME = os.getenv("LLM_MODEL")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 7))

# Cache FAISS retriever to avoid reloading for every request
_cached_retriever = None


class OpenRouterLLM(LLM):
    """
    Custom LangChain LLM wrapper for OpenRouter API.
    Provides OpenAI-compatible interface but routes through OpenRouter endpoint.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY not set in environment variables")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.model_name = model_name

    def _call(self, prompt: str, stop=None):
        """
        Send prompt to OpenRouter API and return response text.
        """
        try:
            logger.debug(f"Sending prompt to OpenRouter (len={len(prompt)} chars)")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content
            logger.debug(f"Received response (len={len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return "Error generating response."

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "openrouter"


def _get_cached_retriever():
    """
    Singleton-like retriever to avoid reloading FAISS index every call.
    """
    global _cached_retriever
    if _cached_retriever is None:
        logger.info("Loading FAISS index and initializing retriever (cached)...")
        vectorstore = load_faiss_index()
        _cached_retriever = get_retriever(vectorstore, k=RETRIEVER_K)
    return _cached_retriever


def build_rag_chain() -> Runnable:
    """
    Build RAG chain combining FAISS retriever and OpenRouter LLM.
    Uses custom structured prompt for reliable answers.
    """
    retriever = _get_cached_retriever()

    # Improved system-like prompt
    prompt_template = """
    You are an AI assistant helping with document-based Q&A.
    Use only the provided context to answer accurately and concisely.
    If the context is insufficient, say: "I don't have enough information."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = OpenRouterLLM()

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        llm=llm
    )
    return rag_chain


def generate_answer(query: str) -> str:
    """
    Generate answer for a query using RAG pipeline.
    Includes basic input validation and structured logging.
    """
    if not query or not query.strip():
        logger.warning("Empty query received")
        return "Please provide a valid question."

    query = query.strip()
    logger.info(f"Generating answer for query (len={len(query)} chars)")

    try:
        rag_chain = build_rag_chain()
        result = rag_chain.invoke({"question": query})
        return result.get("answer", str(result))
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer."


if __name__ == "__main__":
    # Example test
    print(generate_answer("Summarize the documents in simple words."))
