"""
rag_engine.py - RAG pipeline using Google Gemini
"""

import os
import logging
import uuid
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# ─── Load Environment Variables ───────────────────────────────────────────────
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ─── Prompt Template ──────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and precise AI assistant.

Use ONLY the context provided below to answer the question.

If the answer is not present in the context, say:
"I don't have enough information in the document to answer this."

Do NOT make up information.

Context:
{context}

Question:
{question}

Answer:
""",
)


class RAGEngine:
    """
    Full RAG Pipeline:
    1. Load PDF
    2. Split into chunks
    3. Generate embeddings
    4. Store in ChromaDB
    5. Retrieve relevant chunks
    6. Generate answer using Gemini
    """

    def __init__(
        self,
        model_id: str = "models/gemini-3.6-flash",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 100,
        chunk_overlap: int = 10,
        top_k: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):

        self.model_id = os.getenv("GEMINI_MODEL", model_id)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._collection_name = f"rag_{uuid.uuid4().hex[:8]}"
        self._vectorstore = None

        # ─── Configure Gemini API ─────────────────────────────────────────────
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in .env file"
            )

        genai.configure(api_key=api_key)

        # Initialize Gemini model
        self.model = genai.GenerativeModel(self.model_id)

        logger.info(
            "RAGEngine initialized | model=%s | embed=%s",
            self.model_id,
            self.embedding_model
        )

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: LOAD PDF + CHUNKING
    # ──────────────────────────────────────────────────────────────────────────
    def load_pdf(self, pdf_path: str) -> int:

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Loading PDF: %s", pdf_path)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logger.info("Loaded %d pages", len(documents))

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        logger.info("Split into %d chunks", len(chunks))

        if not chunks:
            raise ValueError(
                "No text extracted from PDF. "
                "Maybe the PDF is scanned/image-based."
            )

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: EMBEDDINGS
        # ──────────────────────────────────────────────────────────────────────
        logger.info(
            "Loading embedding model: %s",
            self.embedding_model
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ──────────────────────────────────────────────────────────────────────
        # STEP 3: VECTOR STORE
        # ──────────────────────────────────────────────────────────────────────
        logger.info(
            "Creating ChromaDB collection: %s",
            self._collection_name
        )

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=self._collection_name,
        )

        logger.info("PDF indexed successfully!")

        return len(chunks)

    # ──────────────────────────────────────────────────────────────────────────
    # QUERY FUNCTION
    # ──────────────────────────────────────────────────────────────────────────
    def query(self, question: str) -> dict[str, Any]:

        if self._vectorstore is None:
            raise RuntimeError(
                "PDF not loaded yet. Call load_pdf() first."
            )

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info("Running query: %s", question[:100])

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: RETRIEVE DOCUMENTS
        # ──────────────────────────────────────────────────────────────────────
        retrieved_docs = self._vectorstore.similarity_search(
            question,
            k=self.top_k
        )

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the PDF.",
                "sources": []
            }

        # Combine retrieved chunks into context
        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Prepare source list
        sources = []

        for doc in retrieved_docs:
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", ""),
            })

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: CREATE PROMPT
        # ──────────────────────────────────────────────────────────────────────
        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        # ──────────────────────────────────────────────────────────────────────
        # STEP 3: GEMINI GENERATION
        # ──────────────────────────────────────────────────────────────────────
        candidate_models = [
            self.model_id,
            "models/gemini-3.6-flash",
            "models/gemini-flash-latest",
            "models/gemini-3.5-flash",
            "models/gemini-3.5-flash-lite",
        ]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(candidate_models))

        last_err = None
        for m_name in candidate_models:
            try:
                model_to_use = self.model if m_name == self.model_id else genai.GenerativeModel(m_name)
                response = model_to_use.generate_content(prompt)
                answer = response.text.strip()
                if not answer:
                    answer = (
                        "I couldn't generate an answer. "
                        "Please try another question."
                    )

                self.model_id = m_name
                self.model = model_to_use
                logger.info(
                    "Generated answer (%d chars) using %s",
                    len(answer),
                    m_name
                )
                return {
                    "answer": answer,
                    "sources": sources
                }
            except Exception as e:
                err_str = str(e)
                last_err = e
                if "404" in err_str or "not found" in err_str.lower() or "no longer available" in err_str.lower():
                    logger.warning("Model %s failed with 404, trying next candidate...", m_name)
                    continue
                raise Exception(f"Gemini API error: {err_str}")

        raise Exception(f"Gemini API error: {str(last_err)}")

    # ──────────────────────────────────────────────────────────────────────────
    # CLEAR MEMORY
    # ──────────────────────────────────────────────────────────────────────────
    def clear(self):

        self._vectorstore = None

        logger.info("RAGEngine cleared.")