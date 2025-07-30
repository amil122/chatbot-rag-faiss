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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from src.vector_store.vector_store import load_faiss_index, get_retriever
from langchain.llms.base import LLM
from openai import OpenAI
from pydantic import PrivateAttr


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



from pydantic import PrivateAttr
from langchain.llms.base import LLM
from openai import OpenAI



from pydantic import Field, PrivateAttr
from langchain.llms.base import LLM
from openai import OpenAI

class OpenRouterLLM(LLM):
    model_name: str = Field(default=MODEL_NAME)  
    _client: OpenAI = PrivateAttr()              

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__(model_name=model_name)
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY not set")

        object.__setattr__(self, "_client", OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        ))

    def _call(self, prompt: str, stop=None):
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
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
    {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = OpenRouterLLM()

    
# Create LLM chain with prompt
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    
    # Combine retriever + doc_chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=doc_chain
    )
    
    return rag_chain


def generate_answer(query: str) -> dict:
    """
    Generate answer for a query using RAG pipeline.
    Returns:
        {
            "answer": str,
            "sources": List[str]
        }
    """
    if not query or not query.strip():
        logger.warning("Empty query received")
        return {"answer": "Please provide a valid question.", "sources": []}

    query = query.strip()
    logger.info(f"Generating answer for query (len={len(query)} chars)")

    try:
        rag_chain = build_rag_chain()
        result = rag_chain.invoke({"input": query})

        # Extract answer
        answer = result.get("answer", str(result))

        # Extract sources from context documents
        sources = []
        context_docs = result.get("context", [])
        for doc in context_docs:
            source = doc.metadata.get("source")
            if source and source not in sources:
                sources.append(source)

        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {"answer": "Error generating answer.", "sources": []}



if __name__ == "__main__":
    # Example test
    print(generate_answer("Summarize the documents in simple words."))
