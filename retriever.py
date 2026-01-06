"""
Retriever utilities for RAG pipeline
Handles document retrieval with filtering and ranking
"""

from typing import List, Optional, Dict, Any
import logging
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from ..config import settings

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """Enhanced retriever with filtering and re-ranking capabilities"""
    
    def __init__(
        self,
        base_retriever: VectorStoreRetriever,
        subject_filter: Optional[str] = None
    ):
        """
        Initialize enhanced retriever
        
        Args:
            base_retriever: Base vector store retriever
            subject_filter: Filter results by subject (ML/DL/DSA/etc)
        """
        self.base_retriever = base_retriever
        self.subject_filter = subject_filter
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        subject: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents with optional filtering
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            subject: Subject filter (overrides instance filter)
        
        Returns:
            List of relevant Document objects
        """
        # Use provided subject or instance filter
        filter_subject = subject or self.subject_filter
        
        # Retrieve documents
        if k:
            # Update retriever k
            self.base_retriever.search_kwargs["k"] = k
        
        documents = self.base_retriever.invoke(query)
        
        # Apply subject filter if specified
        if filter_subject:
            documents = [
                doc for doc in documents
                if doc.metadata.get("subject") == filter_subject
            ]
            logger.info(f"Filtered to {len(documents)} documents for subject: {filter_subject}")
        
        # Log retrieval
        logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
        
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of (Document, score) tuples
        """
        k = k or settings.retrieval_k
        
        # Get vector store from retriever
        vectorstore = self.base_retriever.vectorstore
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        logger.info(f"Retrieved {len(results)} documents with scores")
        return results
    
    def retrieve_by_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve documents with metadata filtering
        
        Args:
            query: Search query
            metadata_filter: Dictionary of metadata filters (e.g., {"subject": "ML", "page": 5})
            k: Number of documents to retrieve
        
        Returns:
            List of filtered Document objects
        """
        k = k or settings.retrieval_k
        
        vectorstore = self.base_retriever.vectorstore
        
        documents = vectorstore.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter
        )
        
        logger.info(f"Retrieved {len(documents)} documents with metadata filter: {metadata_filter}")
        return documents
    
    def get_diverse_results(
        self,
        query: str,
        k: Optional[int] = None,
        diversity_threshold: float = 0.7
    ) -> List[Document]:
        """
        Retrieve diverse documents using MMR (Maximal Marginal Relevance)
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            diversity_threshold: Lambda parameter for MMR (0=max diversity, 1=max relevance)
        
        Returns:
            List of diverse Document objects
        """
        k = k or settings.retrieval_k
        
        vectorstore = self.base_retriever.vectorstore
        
        documents = vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * 3,  # Fetch more candidates for diversity
            lambda_mult=diversity_threshold
        )
        
        logger.info(f"Retrieved {len(documents)} diverse documents")
        return documents


def create_retriever(
    vectorstore_manager,
    k: Optional[int] = None,
    subject_filter: Optional[str] = None,
    search_type: str = "similarity"
) -> EnhancedRetriever:
    """
    Create an enhanced retriever from vector store manager
    
    Args:
        vectorstore_manager: VectorStoreManager instance
        k: Number of documents to retrieve
        subject_filter: Subject to filter by
        search_type: "similarity" or "mmr"
    
    Returns:
        EnhancedRetriever instance
    """
    base_retriever = vectorstore_manager.get_retriever(
        k=k,
        search_type=search_type
    )
    
    return EnhancedRetriever(
        base_retriever=base_retriever,
        subject_filter=subject_filter
    )


def format_retrieved_docs(documents: List[Document]) -> str:
    """
    Format retrieved documents for display
    
    Args:
        documents: List of Document objects
    
    Returns:
        Formatted string
    """
    if not documents:
        return "No documents retrieved."
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        subject = doc.metadata.get("subject", "N/A")
        
        formatted.append(
            f"Document {i}:\n"
            f"  Source: {source}\n"
            f"  Page: {page}\n"
            f"  Subject: {subject}\n"
            f"  Content: {doc.page_content[:200]}...\n"
        )
    
    return "\n".join(formatted)


def get_unique_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract unique source information from documents
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of source dictionaries
    """
    sources = []
    seen = set()
    
    for doc in documents:
        source_file = doc.metadata.get("source_file", doc.metadata.get("url", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        subject = doc.metadata.get("subject", "N/A")
        
        source_key = f"{source_file}_{page}"
        
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "source": source_file,
                "page": page,
                "subject": subject
            })
    
    return sources


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Retriever utilities ready")
