import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from rag_engine import RAGEngine
from monitor import QueryMonitor

# Load environment variables
load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextInput > div > div > input { background-color: #1e2130; color: white; }
    .stTextArea > div > div > textarea { background-color: #1e2130; color: white; }
    .answer-box {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-left: 4px solid #7c6af7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e0e0f0;
    }
    .source-box {
        background: #1a1e2e;
        border: 1px solid #2e3450;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.4rem;
        font-size: 0.82rem;
        color: #9da5c7;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c6af7, #a855f7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6b58e8, #9333ea);
        border: none;
    }
    h1 { color: #c4b5fd; }
    .sidebar .stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "monitor" not in st.session_state:
    st.session_state.monitor = QueryMonitor()

monitor: QueryMonitor = st.session_state.monitor

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("---")
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Max recommended size: 10 MB",
    )

    chunk_size = st.slider("Chunk size (tokens)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, 10)
    top_k = st.slider("Top-K retrieved chunks", 1, 8, 3)

    process_btn = st.button("⚡ Process PDF", use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("📊 View Logs", use_container_width=True):
        logs = monitor.read_logs()
        st.text_area("Query Logs", value=logs, height=200)

    st.markdown("---")
    st.markdown("**Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", monitor.get_query_count())
    with col2:
        st.metric("Avg (s)", monitor.get_avg_latency())

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("# 🧠 DocMind — RAG Assistant")
st.markdown("Upload a PDF, then ask anything about it. Powered by LangChain + ChromaDB + Google Gemini.")

# ── Process PDF ───────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        # Check if Gemini API key is set
        if not os.getenv("GEMINI_API_KEY"):
            st.error("❌ GEMINI_API_KEY is required. Please add it to secrets or .env file.")
        else:
            with st.status("Processing your PDF…", expanded=True) as status:
                try:
                    # Clear previous engine if exists
                    if st.session_state.rag_engine:
                        st.session_state.rag_engine.clear()

                    st.write("📥 Saving file…")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    st.write("✂️ Splitting into chunks…")
                    engine = RAGEngine(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                    )

                    st.write("🔢 Generating embeddings (this may take ~1–2 min on CPU)…")
                    num_chunks = engine.load_pdf(tmp_path)

                    st.write("💾 Storing in ChromaDB…")
                    os.unlink(tmp_path)

                    st.session_state.rag_engine = engine
                    st.session_state.pdf_loaded = True
                    st.session_state.chat_history = []

                    status.update(label=f"✅ Ready! Indexed {num_chunks} chunks.", state="complete")
                    monitor.log_event("pdf_loaded", {"filename": uploaded_file.name, "chunks": num_chunks})

                except Exception as e:
                    status.update(label="❌ Failed to process PDF.", state="error")
                    st.error(f"Error: {str(e)}")

# ── Chat Interface ─────────────────────────────────────────────────────────────
if st.session_state.pdf_loaded and st.session_state.rag_engine:
    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the main findings of this document?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →")

    if submitted and question.strip():
        with st.spinner("🔍 Searching and generating answer…"):
            try:
                start = time.time()
                result = st.session_state.rag_engine.query(question)
                latency = round(time.time() - start, 2)

                monitor.log_query(
                    question=question,
                    answer=result["answer"],
                    latency=latency,
                    sources=len(result["sources"]),
                )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "latency": latency,
                })
                st.rerun()

            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")

    # Render history newest-first
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"**🙋 Q:** {item['question']}")
        st.markdown(
            f"<div class='answer-box'>🤖 {item['answer']}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"📎 Sources ({len(item['sources'])})  •  ⏱ {item['latency']}s"):
            for i, src in enumerate(item["sources"], 1):
                st.markdown(
                    f"<div class='source-box'><b>Chunk {i}</b> (page {src.get('page','?')})<br>{src['content'][:300]}…</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("---")

elif not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF in the sidebar and click **Process PDF** to get started.")