"""
Quick test to verify evaluation module imports work correctly
Run this before running full evaluation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Testing evaluation module imports...")
print("-" * 50)

try:
    from src.evaluation import RAGEvaluator, EvaluationMetrics, PerformanceMonitor
    print("✅ Successfully imported RAGEvaluator")
    print("✅ Successfully imported EvaluationMetrics")
    print("✅ Successfully imported PerformanceMonitor")
    
    # Test instantiation
    evaluator = RAGEvaluator(output_dir="test_eval_output")
    print("✅ RAGEvaluator instantiated")
    
    monitor = PerformanceMonitor()
    print("✅ PerformanceMonitor instantiated")
    
    # Test metrics creation
    test_metrics = EvaluationMetrics(
        retrieval_recall_at_5=0.85,
        retrieval_mrr=0.82,
        avg_retrieval_time=0.058,
        avg_answer_length=350.0,
        answer_rate=0.87,
        source_citation_rate=0.93,
        avg_inference_time=2.1,
        tokens_per_second=23.4,
        max_vram_usage=3.2,
        avg_vram_usage=2.9,
        total_queries_evaluated=15,
        total_documents_retrieved=73,
        avg_docs_per_query=4.9,
        overall_score=78.5
    )
    print("✅ EvaluationMetrics created")
    
    # Test resume format
    resume_bullets = test_metrics.to_resume_format()
    print("\n📝 Sample Resume Bullets:")
    print(resume_bullets)
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYou're ready to run the full evaluation:")
    print("  python src/scripts/evaluate_system.py --mode full")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
