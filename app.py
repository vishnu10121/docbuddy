import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from rag_engine import RAGEngine
from monitor import QueryMonitor

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocBuddy – Smart PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f1117 0%, #1a1e2e 100%); }
    .stChatInput > div > div > textarea { background-color: #1e2130; color: white; border-radius: 20px; }
    
    /* Header style */
    .header-title {
        text-align: center;
        background: linear-gradient(135deg, #7c6af7, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .header-subtitle {
        text-align: center;
        color: #9da5c7;
        margin-bottom: 2rem;
    }
    
    /* Upload box */
    .upload-box {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 2px dashed #7c6af7;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Info card */
    .info-card {
        background: #1a1e2e;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2e3450;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-left: 4px solid #7c6af7;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .source-box {
        background: #1a1e2e;
        border: 1px solid #2e3450;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7c6af7, #a855f7);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6b58e8, #9333ea);
        transform: scale(1.02);
        transition: 0.3s;
    }
    
    /* Sidebar hide button */
    .css-1rs6os { display: none; }
    
    hr {
        margin: 1.5rem 0;
        border-color: #2e3450;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = None
if "monitor" not in st.session_state:
    st.session_state.monitor = QueryMonitor()
if "process_complete" not in st.session_state:
    st.session_state.process_complete = False

monitor = st.session_state.monitor

# ─── Header Section ───────────────────────────────────────────────────────────
st.markdown('<h1 class="header-title">📄 DocBuddy</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Your Intelligent PDF Assistant — Ask Anything!</p>', unsafe_allow_html=True)

# ─── Sidebar Controls (Simple) ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🔄 Reset PDF", use_container_width=True):
        st.session_state.rag_engine = None
        st.session_state.pdf_loaded = False
        st.session_state.pdf_name = None
        st.session_state.num_chunks = None
        st.session_state.process_complete = False
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Queries", monitor.get_query_count())
    with col2:
        st.metric("⚡ Avg Response", f"{monitor.get_avg_latency()}s")
    
    st.markdown("---")
    if st.button("📜 View Logs", use_container_width=True):
        logs = monitor.read_logs()
        st.text_area("Query Logs", value=logs, height=200)

# ─── Main Content ─────────────────────────────────────────────────────────────

# If no PDF loaded, show upload box
if not st.session_state.pdf_loaded:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("### 📂 Upload Your PDF")
    st.markdown("Upload any PDF document and start asking questions instantly!")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a text-based PDF (max 10MB)"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_btn = st.button("🚀 Process PDF", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if process_btn:
        if not uploaded_file:
            st.warning("⚠️ Please upload a PDF file first.")
        elif not os.getenv("GEMINI_API_KEY"):
            st.error("❌ Gemini API key not configured. Please check secrets.")
        else:
            with st.spinner("🔄 Processing your PDF... This may take a minute..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    # Default optimal settings
                    engine = RAGEngine(
                        chunk_size=500,
                        chunk_overlap=50,
                        top_k=3,
                    )
                    
                    num_chunks = engine.load_pdf(tmp_path)
                    os.unlink(tmp_path)
                    
                    st.session_state.rag_engine = engine
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.num_chunks = num_chunks
                    st.session_state.process_complete = True
                    st.session_state.chat_history = []
                    
                    monitor.log_event("pdf_loaded", {"filename": uploaded_file.name, "chunks": num_chunks})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# If PDF loaded, show chat interface
else:
    # Success message with PDF info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(f"✅ **{st.session_state.pdf_name}**")
        st.markdown(f"<div class='info-card'>📊 Indexed into <b>{st.session_state.num_chunks}</b> chunks • Ready for questions</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat display area
    chat_container = st.container()
    
    with chat_container:
        # Show chat history
        for item in st.session_state.chat_history:
            st.markdown(f"**🙋 You:** {item['question']}")
            st.markdown(f"<div class='answer-box'>🤖 **DocBuddy:** {item['answer']}</div>", unsafe_allow_html=True)
            with st.expander(f"📎 Sources ({len(item['sources'])}) - ⏱ {item['latency']}s"):
                for i, src in enumerate(item["sources"], 1):
                    st.markdown(
                        f"<div class='source-box'><b>Chunk {i}</b> (Page {src.get('page','?')})<br>{src['content'][:300]}…</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("---")
    
    # Question input at bottom
    st.markdown("### 💬 Ask a Question")
    
    with st.form(key="question_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "Your question",
                placeholder="e.g., What is this document about? Summarize the key findings.",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Ask →", use_container_width=True)
    
    if submitted and question.strip():
        with st.spinner("🔍 Analyzing document and generating answer..."):
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
    
    # Quick suggestions
    if len(st.session_state.chat_history) == 0:
        st.markdown("---")
        st.markdown("### 💡 Try asking:")
        cols = st.columns(3)
        suggestions = [
            "What is this document about?",
            "Summarize the key points",
            "What are the main conclusions?"
        ]
        for i, col in enumerate(cols):
            with col:
                if st.button(f"📌 {suggestions[i]}", use_container_width=True):
                    # Auto-fill question (simplified - would need JS for full auto)
                    pass