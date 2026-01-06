"""
Test suite for evaluating RAG system
Run this to generate quantified metrics for your project
"""

import sys
from pathlib import Path
import torch
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import TutorLLM
from src.ingestion import VectorStoreManager
from src.rag import create_retriever, RAGChain
from src.evaluation import RAGEvaluator, PerformanceMonitor
from src.config import settings


# Test dataset - questions that test different aspects
TEST_QUESTIONS = [
    "What is gradient descent and how does it work?",
    "Explain the difference between supervised and unsupervised learning",
    "What is backpropagation in neural networks?",
    "How do you prevent overfitting in machine learning models?",
    "What is the time complexity of binary search?",
    "Explain what a neural network is",
    "What are the main types of machine learning?",
    "What is regularization and why is it important?",
    "How does the Adam optimizer work?",
    "What is the difference between precision and recall?",
    "Explain what a convolutional neural network is",
    "What is transfer learning?",
    "How do transformers work in NLP?",
    "What is the purpose of batch normalization?",
    "Explain the bias-variance tradeoff"
]


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation and generate metrics report
    """
    print("="*60)
    print("RAG SYSTEM COMPREHENSIVE EVALUATION")
    print("="*60)
    print()
    
    # Load reference answers if available
    reference_qa_path = project_root / "data" / "test_qa_reference.json"
    reference_answers = None
    if reference_qa_path.exists():
        print("📚 Loading reference answers for NLG metrics...")
        with open(reference_qa_path, 'r') as f:
            qa_data = json.load(f)
            reference_answers = [qa['reference_answer'] for qa in qa_data['test_qa_pairs']]
        print(f"  ✓ Loaded {len(reference_answers)} reference answers")
        print("  → Will compute BLEU and ROUGE scores\n")
    else:
        print("ℹ️  No reference answers found. Skipping NLG metrics.")
        print(f"   To enable BLEU/ROUGE: Add {reference_qa_path}\n")
    
    # Initialize components
    print("📦 Initializing system components...")
    
    # Load model
    print("  - Loading LLM...")
    llm = TutorLLM()
    
    # Load vector store
    print("  - Loading vector store...")
    vectorstore_path = project_root / "data" / "vectorstore"
    if not vectorstore_path.exists():
        print("❌ No vector store found. Please run document ingestion first.")
        print("   Run: python src/scripts/ingest_docs.py")
        return
    
    vector_manager = VectorStoreManager(persist_directory=str(vectorstore_path))
    
    # Create retriever and RAG chain
    print("  - Creating RAG chain...")
    retriever = create_retriever(
        vector_manager.vectorstore,
        k=settings.retrieval_k
    )
    rag_chain = RAGChain(llm=llm.model, tokenizer=llm.tokenizer, retriever=retriever)
    
    print("✅ System ready!\n")
    
    # Initialize evaluators
    evaluator = RAGEvaluator(output_dir="evaluation_results")
    perf_monitor = PerformanceMonitor()
    
    # Run evaluation
    print(f"🧪 Running evaluation on {len(TEST_QUESTIONS)} test questions...\n")
    
    answers = []
    sources_list = []
    retrieval_times = []
    inference_times = []
    vram_samples = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question[:50]}...")
        
        # Track retrieval
        with perf_monitor.track_retrieval():
            docs = retriever.get_relevant_documents(question)
        
        retrieval_times.append(perf_monitor.total_retrieval_time / perf_monitor.total_queries)
        
        # Track inference
        with perf_monitor.track_inference():
            result = rag_chain.invoke(question)
        
        inference_times.append(perf_monitor.total_inference_time / perf_monitor.total_queries)
        
        # Collect metrics
        answers.append(result['answer'])
        sources_list.append(result['source_documents'])
        
        if torch.cuda.is_available():
            vram_samples.append(torch.cuda.memory_allocated() / 1024**3)
        else:
            vram_samples.append(0.0)
        
        # Show progress
        print(f"  ✓ Answer length: {len(result['answer'])} chars, "
              f"Sources: {len(result['source_documents'])}, "
              f"Time: {inference_times[-1]:.2f}s")
        print()
    
    # Evaluate answer quality
    print("\n📊 Computing metrics...\n")
    
    quality_metrics = evaluator.evaluate_answer_quality(
        answers=answers,
        sources=sources_list,
        inference_times=inference_times,
        vram_usage_samples=vram_samples,
        reference_answers=reference_answers  # Pass reference answers for NLG metrics
    )
    
    # Generate full report
    metrics = evaluator.generate_full_report(
        test_queries=TEST_QUESTIONS,
        ground_truth_docs=None  # Optional: Add ground truth for retrieval eval
    )
    
    # Save report
    json_path, md_path = evaluator.save_report(metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print()
    print(f"📈 Overall Score: {metrics.overall_score:.1f}/100")
    print()
    print("Key Metrics:")
    print(f"  • Avg Inference Time: {metrics.avg_inference_time:.3f}s")
    print(f"  • Generation Speed: {metrics.tokens_per_second:.1f} tokens/s")
    print(f"  • Answer Rate: {metrics.answer_rate*100:.1f}%")
    print(f"  • Source Citation Rate: {metrics.source_citation_rate*100:.1f}%")
    print(f"  • Max VRAM Usage: {metrics.max_vram_usage:.2f}GB")
    print(f"  • Avg Docs Retrieved: {metrics.avg_docs_per_query:.1f}")
    
    # Show NLG metrics if available
    if metrics.bleu_score > 0:
        print()
        print("NLG Quality Metrics:")
        print(f"  • BLEU-4: {metrics.bleu_score:.4f}")
        print(f"  • ROUGE-1: {metrics.rouge_1_f:.4f}")
        print(f"  • ROUGE-2: {metrics.rouge_2_f:.4f}")
        print(f"  • ROUGE-L: {metrics.rouge_l_f:.4f}")
    
    print()
    print("="*60)
    print()
    print("📝 RESUME-READY BULLETS:")
    print("="*60)
    print(metrics.to_resume_format())
    print()
    print("="*60)
    print()
    print(f"📄 Full reports saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print()
    
    # Performance summary
    perf_monitor.print_summary()
    
    return metrics


def run_quick_benchmark():
    """
    Quick benchmark on 5 questions for fast iteration
    """
    print("🚀 Running quick benchmark (5 questions)...\n")
    
    quick_questions = TEST_QUESTIONS[:5]
    
    # Initialize
    llm = TutorLLM()
    vectorstore_path = project_root / "data" / "vectorstore"
    vector_manager = VectorStoreManager(persist_directory=str(vectorstore_path))
    retriever = create_retriever(vector_manager.vectorstore, k=5)
    rag_chain = RAGChain(llm=llm.model, tokenizer=llm.tokenizer, retriever=retriever)
    
    perf_monitor = PerformanceMonitor()
    
    for i, question in enumerate(quick_questions, 1):
        print(f"[{i}/{len(quick_questions)}] {question[:60]}...")
        
        with perf_monitor.track_retrieval():
            docs = retriever.get_relevant_documents(question)
        
        with perf_monitor.track_inference():
            result = rag_chain.invoke(question)
        
        print(f"  ✓ {len(result['answer'])} chars, "
              f"{len(result['source_documents'])} sources, "
              f"{perf_monitor.total_inference_time/perf_monitor.total_queries:.2f}s\n")
    
    perf_monitor.print_summary()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        '--mode',
        choices=['full', 'quick'],
        default='full',
        help='Evaluation mode: full (15 questions) or quick (5 questions)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_benchmark()
    else:
        run_comprehensive_evaluation()
