"""
CLI Script for Testing RAG Pipeline
Interactive Q&A with the AI tutor from command line
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import TutorLLM
from src.ingestion import VectorStoreManager
from src.rag import create_retriever, RAGChain
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def interactive_mode(rag_chain: RAGChain, subject: str):
    """Run interactive Q&A session"""
    print(f"\n{'='*60}")
    print(f"   Study Tutor - Interactive Mode ({subject})")
    print(f"{'='*60}")
    print("Ask questions about your study materials.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy studying!")
                break
            
            # Get answer
            print("\nTutor: ", end="", flush=True)
            result = rag_chain.ask(question)
            
            print(result["answer"])
            
            # Show sources
            if result.get("sources"):
                print(f"\n[Sources: {result['num_sources']} documents, {result['inference_time']}s]")
                for i, source in enumerate(result["sources"][:3], 1):
                    print(f"  {i}. {source['source']} (Page {source['page']})")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


def single_query_mode(rag_chain: RAGChain, question: str, verbose: bool = False):
    """Answer a single question"""
    logger.info(f"Question: {question}")
    
    result = rag_chain.ask(question)
    
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"\n{'-'*60}")
    print(f"Inference time: {result['inference_time']}s")
    print(f"Sources: {result['num_sources']} documents")
    
    if verbose and result.get("sources"):
        print(f"\n{'='*60}")
        print("SOURCES:")
        print(f"{'='*60}")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n{i}. {source['source']}")
            print(f"   Page: {source['page']}, Subject: {source['subject']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG pipeline with Study Tutor AI"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="General",
        choices=["ML", "DL", "DSA", "Mathematics", "Statistics", "Python", "General"],
        help="Subject category"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (interactive mode if not provided)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="study_tutor",
        help="Vector store collection name"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use fine-tuned LoRA model"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including sources"
    )
    
    args = parser.parse_args()
    
    try:
        # Load LLM
        logger.info("Loading language model...")
        llm = TutorLLM(
            use_lora=args.use_lora,
            adapter_path=args.adapter_path
        )
        logger.info("Model loaded successfully")
        
        # Load vector store
        logger.info(f"Loading vector store: {args.collection}")
        vectorstore_manager = VectorStoreManager(collection_name=args.collection)
        vectorstore_manager.load_vectorstore()
        
        stats = vectorstore_manager.get_stats()
        logger.info(f"Vector store loaded: {stats['total_documents']} documents")
        
        # Create retriever
        retriever = create_retriever(
            vectorstore_manager=vectorstore_manager,
            k=args.k,
            subject_filter=args.subject if args.subject != "General" else None
        )
        
        # Create RAG chain
        logger.info(f"Creating RAG chain for subject: {args.subject}")
        rag_chain = RAGChain(
            llm=llm,
            retriever=retriever,
            subject=args.subject
        )
        
        # Run in appropriate mode
        if args.question:
            # Single query mode
            single_query_mode(rag_chain, args.question, args.verbose)
        else:
            # Interactive mode
            interactive_mode(rag_chain, args.subject)
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {e}")
        logger.error("Please run ingest_docs.py first to create the vector store.")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
