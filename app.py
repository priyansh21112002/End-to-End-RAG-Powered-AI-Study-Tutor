"""
Streamlit UI for Study Tutor AI
Interactive chat interface with document upload and VRAM monitoring
"""

import streamlit as st
import torch
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import TutorLLM
from src.ingestion import (
    load_documents,
    chunk_documents,
    VectorStoreManager,
    get_document_stats
)
from src.rag import create_retriever, RAGChain
from src.config import settings

# Page config
st.set_page_config(
    page_title="Study Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-citation {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-left: 3px solid #1E88E5;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_llm():
    """Load LLM (cached)"""
    with st.spinner("Loading AI model... This may take a few minutes on first run."):
        llm = TutorLLM(use_quantization=True)
    return llm


@st.cache_resource
def load_vectorstore():
    """Load vector store (cached)"""
    try:
        manager = VectorStoreManager()
        manager.load_vectorstore()
        return manager
    except FileNotFoundError:
        return None


def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'subject' not in st.session_state:
        st.session_state.subject = "General"
    if 'vectorstore_loaded' not in st.session_state:
        st.session_state.vectorstore_loaded = False


def get_gpu_info():
    """Get GPU information"""
    if torch.cuda.is_available():
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "vram_allocated": torch.cuda.memory_allocated() / 1024**3,
            "vram_reserved": torch.cuda.memory_reserved() / 1024**3,
            "vram_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    return {"available": False}


def display_gpu_metrics():
    """Display GPU metrics in sidebar"""
    gpu_info = get_gpu_info()
    
    if gpu_info["available"]:
        st.sidebar.markdown("### 🖥️ GPU Status")
        st.sidebar.markdown(f"**GPU:** {gpu_info['name']}")
        
        # VRAM progress bar
        vram_used = gpu_info["vram_allocated"]
        vram_total = gpu_info["vram_total"]
        vram_percent = (vram_used / vram_total) * 100
        
        st.sidebar.progress(vram_percent / 100)
        st.sidebar.markdown(f"**VRAM:** {vram_used:.2f} / {vram_total:.2f} GB ({vram_percent:.1f}%)")
        
        if st.sidebar.button("Clear GPU Cache"):
            torch.cuda.empty_cache()
            st.sidebar.success("Cache cleared!")
            st.rerun()
    else:
        st.sidebar.warning("No GPU detected - using CPU")


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎓 Study Tutor AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Subject selection
        subject = st.selectbox(
            "Subject",
            ["General", "ML", "Thesis"],
            index=["General", "ML", "Thesis"].index(st.session_state.subject) if st.session_state.subject in ["General", "ML", "Thesis"] else 0
        )
        st.session_state.subject = subject
        
        # Retrieval settings
        st.markdown("### 🔍 Retrieval Settings")
        k_docs = st.slider("Documents to retrieve", 1, 10, 5)
        
        st.markdown("---")
        
        # GPU metrics
        display_gpu_metrics()
        
        st.markdown("---")
        
        # Document upload
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Ingest Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files
                    temp_paths = []
                    for file in uploaded_files:
                        temp_path = settings.raw_data_dir / file.name
                        temp_path.write_bytes(file.read())
                        temp_paths.append(str(temp_path))
                    
                    # Load and chunk
                    docs = load_documents(temp_paths, subject=subject)
                    chunks = chunk_documents(docs)
                    
                    # Add to vector store
                    manager = st.session_state.get('vectorstore_manager')
                    if manager is None:
                        manager = VectorStoreManager()
                        manager.build_vectorstore(chunks)
                        st.session_state.vectorstore_manager = manager
                    else:
                        manager.add_documents(chunks)
                    
                    st.session_state.vectorstore_loaded = True
                    st.success(f"✅ Ingested {len(docs)} documents ({len(chunks)} chunks)")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### 💬 Chat with {subject} Tutor")
    
    with col2:
        # Vector store status
        vs_manager = st.session_state.get('vectorstore_manager') or load_vectorstore()
        if vs_manager:
            stats = vs_manager.get_stats()
            st.metric("Documents in DB", stats.get('total_documents', 0))
        else:
            st.warning("No documents loaded")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(
                                f'<div class="source-citation">'
                                f'{i}. {source["source"]} (Page {source["page"]}, Subject: {source["subject"]})'
                                f'</div>',
                                unsafe_allow_html=True
                            )
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Check if vectorstore is loaded
        vs_manager = st.session_state.get('vectorstore_manager') or load_vectorstore()
        
        if vs_manager is None:
            st.error("Please upload and ingest documents first!")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Load LLM
                    llm = load_llm()
                    
                    # Create retriever and RAG chain
                    retriever = create_retriever(
                        vs_manager,
                        k=k_docs,
                        subject_filter=subject if subject != "General" else None
                    )
                    
                    rag_chain = RAGChain(
                        llm=llm,
                        retriever=retriever,
                        subject=subject
                    )
                    
                    # Get answer
                    start_time = time.time()
                    result = rag_chain.ask(prompt, k=k_docs)
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Time", f"{result['inference_time']:.2f}s")
                    with col2:
                        st.metric("📚 Sources", result['num_sources'])
                    with col3:
                        if torch.cuda.is_available():
                            st.metric("🖥️ VRAM", f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", [])
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Study Tutor AI | RAG + QLoRA | Built with LangChain & Streamlit'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
