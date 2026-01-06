"""
Text chunking utilities for RAG
Optimized for 8GB VRAM with appropriate chunk sizes
"""

from typing import List, Optional
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..config import settings

logger = logging.getLogger(__name__)


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter optimized for technical content
    
    Args:
        chunk_size: Size of each chunk (default from settings: 800)
        chunk_overlap: Overlap between chunks (default from settings: 100)
        separators: Custom separators (defaults to technical content separators)
    
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    # Default separators optimized for technical/educational content
    if separators is None:
        separators = [
            "\n\n\n",  # Multiple blank lines (section breaks)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            "! ",
            "? ",
            "; ",
            ", ",
            " ",       # Words
            ""         # Characters
        ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )
    
    return splitter


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    preserve_metadata: bool = True
) -> List[Document]:
    """
    Split documents into chunks while preserving metadata
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        preserve_metadata: Whether to copy metadata to chunks
    
    Returns:
        List of chunked Document objects
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []
    
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # Split documents
    chunks = splitter.split_documents(documents)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    logger.info(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")
    
    return chunks


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[dict] = None
) -> List[Document]:
    """
    Chunk a single text string into Document objects
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        metadata: Metadata to add to each chunk
    
    Returns:
        List of Document objects
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    
    # Convert to Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = metadata.copy() if metadata else {}
        doc_metadata.update({
            "chunk_id": i,
            "chunk_size": len(chunk)
        })
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    return documents


def get_optimal_chunk_size(
    documents: List[Document],
    target_chunks: int = 1000,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 400
) -> int:
    """
    Calculate optimal chunk size based on document corpus
    
    Args:
        documents: List of documents
        target_chunks: Target number of chunks
        max_chunk_size: Maximum chunk size
        min_chunk_size: Minimum chunk size
    
    Returns:
        Recommended chunk size
    """
    if not documents:
        return settings.chunk_size
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    optimal_size = total_chars // target_chunks
    
    # Clamp to reasonable range
    optimal_size = max(min_chunk_size, min(max_chunk_size, optimal_size))
    
    logger.info(f"Recommended chunk size: {optimal_size} (total chars: {total_chars})")
    return optimal_size


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test chunking
    sample_text = """
    Machine Learning is a subset of artificial intelligence that focuses on 
    developing algorithms that can learn from data.
    
    Deep Learning is a specialized branch of machine learning that uses neural 
    networks with multiple layers.
    
    Data Structures and Algorithms are fundamental concepts in computer science 
    that help solve computational problems efficiently.
    """
    
    chunks = chunk_text(sample_text, metadata={"subject": "ML", "source": "test"})
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}: {chunk.page_content[:100]}...")
