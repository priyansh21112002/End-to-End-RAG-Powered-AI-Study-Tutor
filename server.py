"""
FastAPI Server for Study Tutor AI
Provides REST API endpoints for document ingestion and Q&A
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from ..models import TutorLLM
from ..ingestion import (
    load_documents,
    chunk_documents,
    VectorStoreManager
)
from ..rag import create_retriever, RAGChain, PracticeQuestionGenerator, SolutionExplainer
from ..config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "llm": None,
    "vectorstore_manager": None,
    "rag_chains": {}  # subject -> RAGChain
}


# Pydantic models for API
class IngestRequest(BaseModel):
    """Request model for document ingestion"""
    files: List[str] = Field(..., description="List of file paths to ingest")
    subject: str = Field(default="General", description="Subject category (ML/DL/DSA/etc)")
    source_type: str = Field(default="auto", description="Source type: auto/pdf/directory/web")


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    status: str
    message: str
    documents_processed: int
    chunks_created: int
    subject: str
    processing_time: float


class AskRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., description="User question")
    subject: str = Field(default="General", description="Subject category")
    k: Optional[int] = Field(default=None, description="Number of documents to retrieve")
    return_sources: bool = Field(default=True, description="Include source citations")


class AskResponse(BaseModel):
    """Response model for question answering"""
    answer: str
    subject: str
    sources: Optional[List[Dict[str, Any]]] = None
    inference_time: float
    num_sources: int


class PracticeRequest(BaseModel):
    """Request model for practice question generation"""
    topic: str = Field(..., description="Topic to generate questions for")
    subject: str = Field(default="General", description="Subject category")
    num_questions: int = Field(default=5, description="Number of questions")


class SolutionRequest(BaseModel):
    """Request model for step-by-step solutions"""
    problem: str = Field(..., description="Problem statement")
    subject: str = Field(default="General", description="Subject category")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    vectorstore_loaded: bool
    device: str
    vram_allocated_gb: Optional[float] = None
    gpu_name: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Study Tutor API...")
    logger.info(f"Device: {settings.get_device()}")
    
    # Load LLM
    try:
        logger.info("Loading LLM...")
        app_state["llm"] = TutorLLM(use_quantization=True)
        logger.info("LLM loaded successfully")
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        app_state["llm"] = None
    
    # Try to load existing vectorstore
    try:
        logger.info("Loading vector store...")
        vectorstore_manager = VectorStoreManager()
        vectorstore_manager.load_vectorstore()
        app_state["vectorstore_manager"] = vectorstore_manager
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"No existing vector store found: {e}")
        app_state["vectorstore_manager"] = None
    
    logger.info("API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="Study Tutor API",
    description="AI Tutor with RAG + QLoRA for educational content",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Study Tutor API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    device = settings.get_device()
    
    response = {
        "status": "healthy",
        "model_loaded": app_state["llm"] is not None,
        "vectorstore_loaded": app_state["vectorstore_manager"] is not None,
        "device": device
    }
    
    if device == "cuda" and torch.cuda.is_available():
        response["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        response["gpu_name"] = torch.cuda.get_device_name(0)
    
    return response


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents into the vector store
    
    Processes PDFs or web pages, chunks them, and adds to vector store
    """
    start_time = time.time()
    
    try:
        logger.info(f"Ingesting documents: {request.files}")
        logger.info(f"Subject: {request.subject}")
        
        # Load documents
        documents = load_documents(
            source_paths=request.files,
            subject=request.subject,
            source_type=request.source_type
        )
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents loaded")
        
        # Chunk documents
        chunks = chunk_documents(documents)
        
        # Initialize or update vector store
        if app_state["vectorstore_manager"] is None:
            logger.info("Creating new vector store...")
            vectorstore_manager = VectorStoreManager()
            vectorstore_manager.build_vectorstore(chunks)
            app_state["vectorstore_manager"] = vectorstore_manager
        else:
            logger.info("Adding to existing vector store...")
            app_state["vectorstore_manager"].add_documents(chunks)
        
        # Clear RAG chains cache to reload with new data
        app_state["rag_chains"] = {}
        
        processing_time = time.time() - start_time
        
        return IngestResponse(
            status="success",
            message=f"Successfully ingested {len(documents)} documents",
            documents_processed=len(documents),
            chunks_created=len(chunks),
            subject=request.subject,
            processing_time=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an AI tutor response with sources
    
    Uses RAG to retrieve relevant context and generate educational responses
    """
    if app_state["llm"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if app_state["vectorstore_manager"] is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Please ingest documents first.")
    
    try:
        logger.info(f"Question: {request.question[:100]}...")
        logger.info(f"Subject: {request.subject}")
        
        # Get or create RAG chain for subject
        if request.subject not in app_state["rag_chains"]:
            logger.info(f"Creating RAG chain for subject: {request.subject}")
            retriever = create_retriever(
                vectorstore_manager=app_state["vectorstore_manager"],
                subject_filter=request.subject if request.subject != "General" else None
            )
            
            app_state["rag_chains"][request.subject] = RAGChain(
                llm=app_state["llm"],
                retriever=retriever,
                subject=request.subject
            )
        
        # Get answer
        result = app_state["rag_chains"][request.subject].ask(
            question=request.question,
            k=request.k,
            return_sources=request.return_sources
        )
        
        return AskResponse(**result)
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/practice")
async def generate_practice_questions(request: PracticeRequest):
    """Generate practice questions for a topic"""
    if app_state["llm"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        retriever = None
        if app_state["vectorstore_manager"]:
            retriever = create_retriever(
                vectorstore_manager=app_state["vectorstore_manager"],
                subject_filter=request.subject if request.subject != "General" else None
            )
        
        generator = PracticeQuestionGenerator(
            llm=app_state["llm"],
            retriever=retriever,
            subject=request.subject
        )
        
        result = generator.generate(
            topic=request.topic,
            num_questions=request.num_questions
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating practice questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solution")
async def explain_solution(request: SolutionRequest):
    """Provide step-by-step solution to a problem"""
    if app_state["llm"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        retriever = None
        if app_state["vectorstore_manager"]:
            retriever = create_retriever(
                vectorstore_manager=app_state["vectorstore_manager"],
                subject_filter=request.subject if request.subject != "General" else None
            )
        
        explainer = SolutionExplainer(
            llm=app_state["llm"],
            retriever=retriever,
            subject=request.subject
        )
        
        result = explainer.explain(problem=request.problem)
        
        return result
    
    except Exception as e:
        logger.error(f"Error explaining solution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get vector store statistics"""
    if app_state["vectorstore_manager"] is None:
        return {"status": "no_vectorstore"}
    
    return app_state["vectorstore_manager"].get_stats()


@app.post("/clear-cache")
async def clear_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return {
            "status": "success",
            "message": "GPU cache cleared",
            "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)
        }
    return {"status": "no_gpu", "message": "No GPU available"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )
