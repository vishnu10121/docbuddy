"""
rag_engine.py - RAG pipeline using Google Gemini
"""

import os
import logging
import uuid
from typing import Any
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Prompt Template ──────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful, precise AI assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the document to answer this."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:""",
)


class RAGEngine:
    """
    Encapsulates the full RAG pipeline:
      1. PDF loading & chunking
      2. Embedding with sentence-transformers (CPU)
      3. Vector storage in ChromaDB
      4. Retrieval + generation via Google Gemini
    """
    
    def __init__(
        self,
        model_id: str = "models/gemini-1.5-flash",  # ✅ Working (lighter model)
        
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._collection_name = f"rag_{uuid.uuid4().hex[:8]}"
        self._vectorstore = None
        
        # Force set the API key
        if not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = "AIzaSyBakZ96SfKsrisPRG-VYNF4dSCMA9QS5l0"
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your .env file"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)

        logger.info(
            "RAGEngine initialised | model=%s | embed=%s | chunk=%d | overlap=%d | top_k=%d",
            model_id, embedding_model, chunk_size, chunk_overlap, top_k,
        )

    # ── Step 1: Load & chunk PDF ──────────────────────────────────────────────
    def load_pdf(self, pdf_path: str) -> int:
        """
        Load a PDF, split it into chunks, embed them, and store in ChromaDB.
        Returns the number of chunks indexed.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Loading PDF: %s", pdf_path)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info("Loaded %d pages", len(documents))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split into %d chunks", len(chunks))

        if not chunks:
            raise ValueError("No text could be extracted from the PDF. Is it scanned/image-based?")

        # ── Step 2: Embed ──────────────────────────────────────────────────────
        logger.info("Loading embedding model (CPU): %s", self.embedding_model)
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── Step 3: Store in ChromaDB ──────────────────────────────────────────
        logger.info("Building ChromaDB vector store: collection=%s", self._collection_name)
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=self._collection_name,
        )

        logger.info("PDF loaded and indexed successfully!")
        return len(chunks)

    # ── Query ──────────────────────────────────────────────────────────────────
    def query(self, question: str) -> dict[str, Any]:
        """
        Run a RAG query using Google Gemini.
        Returns {"answer": str, "sources": list[dict]}
        """
        if self._vectorstore is None:
            raise RuntimeError("PDF has not been loaded yet. Call load_pdf() first.")

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        logger.info("Running query: %s", question[:80])

        # Step 1: Retrieve relevant documents
        retrieved_docs = self._vectorstore.similarity_search(question, k=self.top_k)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found in the PDF.",
                "sources": []
            }
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", ""),
            })

        # Step 2: Prepare prompt
        prompt = RAG_PROMPT.format(context=context, question=question)

        # Step 3: Generate answer using Gemini
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            if not answer:
                answer = "I couldn't generate an answer. Please try rephrasing your question."
            
            logger.info("Answer generated (%d chars, %d sources)", len(answer), len(sources))
            return {"answer": answer, "sources": sources}
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    # ── Clear method ──────────────────────────────────────────────────────────
    def clear(self):
        """Clear the vector store to free memory."""
        self._vectorstore = None
        logger.info("RAGEngine cleared.")