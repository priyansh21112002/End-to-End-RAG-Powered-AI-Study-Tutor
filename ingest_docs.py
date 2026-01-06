"""
CLI Script for Document Ingestion
Processes PDFs and web pages, creates vector store for RAG
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import (
    load_documents,
    chunk_documents,
    VectorStoreManager,
    get_document_stats
)
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Study Tutor vector store"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Paths to PDF files, directories, or URLs"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="General",
        help="Subject category for the documents (ML, DL, DSA, Mathematics, Statistics, Python, General, or custom)"
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="auto",
        choices=["auto", "pdf", "directory", "web"],
        help="Source type (default: auto-detect)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="study_tutor",
        help="Vector store collection name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters (default from config)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap (default from config)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store (delete existing collection)"
    )
    
    args = parser.parse_args()
    
    logger.info("=== Document Ingestion ===")
    logger.info(f"Files: {args.files}")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Collection: {args.collection}")
    
    try:
        # Load documents
        logger.info("Loading documents...")
        documents = load_documents(
            source_paths=args.files,
            subject=args.subject,
            source_type=args.source_type
        )
        
        if not documents:
            logger.error("No documents loaded. Check file paths.")
            return 1
        
        # Print stats
        stats = get_document_stats(documents)
        logger.info(f"Loaded {stats['total_documents']} documents")
        logger.info(f"Total characters: {stats['total_characters']:,}")
        logger.info(f"Subjects: {stats.get('subjects', {})}")
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Initialize vector store manager
        vectorstore_manager = VectorStoreManager(
            collection_name=args.collection
        )
        
        # Reset if requested
        if args.reset:
            logger.warning("Resetting vector store...")
            try:
                vectorstore_manager.load_vectorstore()
                vectorstore_manager.delete_collection()
            except:
                pass
        
        # Build or update vector store
        try:
            logger.info("Loading existing vector store...")
            vectorstore_manager.load_vectorstore()
            logger.info("Adding new documents to existing store...")
            vectorstore_manager.add_documents(chunks)
        except FileNotFoundError:
            logger.info("Creating new vector store...")
            vectorstore_manager.build_vectorstore(chunks)
        
        # Print final stats
        final_stats = vectorstore_manager.get_stats()
        logger.info("=== Ingestion Complete ===")
        logger.info(f"Total documents in store: {final_stats['total_documents']}")
        logger.info(f"Subjects: {final_stats.get('subjects', [])}")
        logger.info(f"Persist directory: {final_stats['persist_directory']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
