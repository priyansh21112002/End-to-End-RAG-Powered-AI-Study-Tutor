"""
Vector store management using ChromaDB
Handles embedding generation and persistent storage
"""

from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from ..config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations for RAG"""
    
    def __init__(
        self,
        collection_name: str = "study_tutor",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize vector store manager
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist vector store
            embedding_model: Embedding model name
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(settings.vectorstore_dir)
        self.embedding_model_name = embedding_model or settings.embedding_model
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Embeddings run on CPU to save VRAM
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore: Optional[Chroma] = None
    
    def build_vectorstore(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Chroma:
        """
        Build vector store from documents
        
        Args:
            documents: List of chunked Document objects
            batch_size: Number of documents to process at once
        
        Returns:
            Chroma vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to build vector store")
        
        logger.info(f"Building vector store with {len(documents)} documents")
        logger.info(f"Persist directory: {self.persist_directory}")
        
        # Process documents in batches to manage memory
        if len(documents) > batch_size:
            logger.info(f"Processing in batches of {batch_size}")
            
            # Create vector store with first batch
            first_batch = documents[:batch_size]
            self.vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
            # Add remaining documents in batches
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                logger.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        else:
            # Process all at once
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        
        logger.info("Vector store built successfully")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk
        
        Returns:
            Chroma vector store instance
        """
        persist_path = Path(self.persist_directory)
        if not persist_path.exists():
            raise FileNotFoundError(f"Vector store not found at {self.persist_directory}")
        
        logger.info(f"Loading vector store from {self.persist_directory}")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Get collection info
        collection = self.vectorstore._collection
        count = collection.count()
        logger.info(f"Loaded vector store with {count} documents")
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() or load_vectorstore() first.")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.vectorstore.add_documents(documents)
        logger.info("Documents added successfully")
    
    def get_retriever(
        self,
        k: Optional[int] = None,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> VectorStoreRetriever:
        """
        Get retriever for RAG chain
        
        Args:
            k: Number of documents to retrieve
            search_type: "similarity" or "mmr" (maximal marginal relevance)
            search_kwargs: Additional search parameters
        
        Returns:
            VectorStoreRetriever instance
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore() or load_vectorstore() first.")
        
        k = k or settings.retrieval_k
        
        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs["k"] = k
        
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        logger.info(f"Created retriever with k={k}, search_type={search_type}")
        return retriever
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter (e.g., {"subject": "ML"})
        
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        k = k or settings.retrieval_k
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        # Get sample to analyze metadata
        sample = collection.peek(limit=100)
        subjects = set()
        if sample and 'metadatas' in sample:
            for metadata in sample['metadatas']:
                if metadata and 'subject' in metadata:
                    subjects.add(metadata['subject'])
        
        return {
            "status": "initialized",
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory,
            "subjects": list(subjects)
        }
    
    def delete_collection(self) -> None:
        """Delete the current collection"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
            self.vectorstore = None


def build_vectorstore(
    documents: List[Document],
    collection_name: str = "study_tutor",
    persist_directory: Optional[str] = None
) -> Chroma:
    """
    Convenience function to build vector store
    
    Args:
        documents: List of chunked documents
        collection_name: Chroma collection name
        persist_directory: Persistence directory
    
    Returns:
        Chroma vector store
    """
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    return manager.build_vectorstore(documents)


def get_retriever(
    collection_name: str = "study_tutor",
    persist_directory: Optional[str] = None,
    k: Optional[int] = None
) -> VectorStoreRetriever:
    """
    Convenience function to get retriever from existing vector store
    
    Args:
        collection_name: Chroma collection name
        persist_directory: Persistence directory
        k: Number of documents to retrieve
    
    Returns:
        VectorStoreRetriever instance
    """
    manager = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    manager.load_vectorstore()
    return manager.get_retriever(k=k)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample documents
    sample_docs = [
        Document(
            page_content="Machine learning is a subset of AI",
            metadata={"subject": "ML", "source": "test"}
        ),
        Document(
            page_content="Deep learning uses neural networks",
            metadata={"subject": "DL", "source": "test"}
        )
    ]
    
    manager = VectorStoreManager()
    # manager.build_vectorstore(sample_docs)
    # print(manager.get_stats())
    
    print("Vector store manager ready")
