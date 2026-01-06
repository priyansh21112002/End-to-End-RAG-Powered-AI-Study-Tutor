"""
Document loading utilities
Supports PDFs and web pages with metadata preservation
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)

logger = logging.getLogger(__name__)


def load_pdf(file_path: str, subject: Optional[str] = None) -> List[Document]:
    """
    Load a single PDF file with metadata
    
    Args:
        file_path: Path to PDF file
        subject: Subject category (ML/DL/DSA/etc)
    
    Returns:
        List of Document objects with page-level metadata
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add custom metadata
        for doc in documents:
            doc.metadata["source_file"] = Path(file_path).name
            doc.metadata["source_path"] = file_path
            if subject:
                doc.metadata["subject"] = subject
        
        logger.info(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return []


def load_pdfs_from_directory(
    directory: str,
    subject: Optional[str] = None,
    glob_pattern: str = "**/*.pdf"
) -> List[Document]:
    """
    Load all PDFs from a directory
    
    Args:
        directory: Directory path containing PDFs
        subject: Subject category for all documents
        glob_pattern: Pattern to match files
    
    Returns:
        List of Document objects
    """
    try:
        loader = DirectoryLoader(
            directory,
            glob=glob_pattern,
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        
        # Add subject metadata
        if subject:
            for doc in documents:
                doc.metadata["subject"] = subject
        
        logger.info(f"Loaded {len(documents)} pages from directory {directory}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading directory {directory}: {e}")
        return []


def load_webpage(url: str, subject: Optional[str] = None) -> List[Document]:
    """
    Load content from a web page
    
    Args:
        url: Web page URL
        subject: Subject category
    
    Returns:
        List of Document objects (usually 1 per URL)
    """
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source_type"] = "web"
            doc.metadata["url"] = url
            if subject:
                doc.metadata["subject"] = subject
        
        logger.info(f"Loaded content from {url}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading webpage {url}: {e}")
        return []


def load_documents(
    source_paths: List[str],
    subject: Optional[str] = None,
    source_type: str = "auto"
) -> List[Document]:
    """
    Load documents from multiple sources (PDFs, directories, URLs)
    
    Args:
        source_paths: List of file paths, directory paths, or URLs
        subject: Subject category for all documents
        source_type: "auto", "pdf", "directory", or "web"
    
    Returns:
        Combined list of Document objects
    """
    all_documents = []
    
    for source in source_paths:
        # Auto-detect source type
        if source_type == "auto":
            if source.startswith("http://") or source.startswith("https://"):
                detected_type = "web"
            elif Path(source).is_dir():
                detected_type = "directory"
            elif Path(source).suffix.lower() == ".pdf":
                detected_type = "pdf"
            else:
                logger.warning(f"Could not detect type for {source}, skipping")
                continue
        else:
            detected_type = source_type
        
        # Load based on type
        if detected_type == "web":
            docs = load_webpage(source, subject)
        elif detected_type == "directory":
            docs = load_pdfs_from_directory(source, subject)
        elif detected_type == "pdf":
            docs = load_pdf(source, subject)
        else:
            logger.warning(f"Unsupported source type: {detected_type}")
            continue
        
        all_documents.extend(docs)
    
    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def get_document_stats(documents: List[Document]) -> Dict[str, Any]:
    """
    Get statistics about loaded documents
    
    Args:
        documents: List of Document objects
    
    Returns:
        Dictionary with statistics
    """
    if not documents:
        return {"total": 0}
    
    subjects = [doc.metadata.get("subject", "Unknown") for doc in documents]
    sources = [doc.metadata.get("source_file", doc.metadata.get("url", "Unknown")) for doc in documents]
    
    stats = {
        "total_documents": len(documents),
        "total_characters": sum(len(doc.page_content) for doc in documents),
        "subjects": dict(zip(*[list(set(subjects)), [subjects.count(s) for s in set(subjects)]])),
        "unique_sources": len(set(sources)),
        "sources": list(set(sources))[:10]  # First 10 sources
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample PDF (replace with actual path)
    # docs = load_documents(["./data/raw/sample.pdf"], subject="ML")
    # print(get_document_stats(docs))
    
    print("Document loader ready. Use load_documents() to process files.")
