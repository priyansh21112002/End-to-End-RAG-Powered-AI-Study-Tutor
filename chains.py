"""
RAG chain orchestration
Combines retrieval, prompting, and LLM inference for the AI tutor
"""

from typing import Dict, Any, List, Optional
import logging
import time
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from .prompts import create_rag_prompt, format_sources, create_practice_prompt, create_solution_prompt
from .retriever import EnhancedRetriever, get_unique_sources
from ..config import settings

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain for question answering with context retrieval"""
    
    def __init__(
        self,
        llm,
        retriever: EnhancedRetriever,
        subject: str = "General"
    ):
        """
        Initialize RAG chain
        
        Args:
            llm: Language model instance
            retriever: EnhancedRetriever for document retrieval
            subject: Subject category for prompts
        """
        self.llm = llm
        self.retriever = retriever
        self.subject = subject
        
        # Create prompt template
        self.prompt = create_rag_prompt(subject)
        
        # Build chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the RAG chain using LCEL (LangChain Expression Language)"""
        
        def format_docs(docs):
            return format_sources(docs)
        
        # Create chain: retrieve -> format -> prompt -> llm
        # Note: No StrOutputParser needed since LLM._call returns string directly
        chain = (
            {
                "context": self.retriever.base_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )
        
        return chain
    
    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources
        
        Args:
            question: User question
            k: Number of documents to retrieve
            return_sources: Whether to return source documents
        
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant documents with subject filter
        k = k or 5
        
        # Use metadata filtering to ensure we get documents from correct subject
        if self.subject:
            documents = self.retriever.retrieve_by_metadata(
                question, 
                metadata_filter={"subject": self.subject},
                k=k
            )
        else:
            if k:
                self.retriever.base_retriever.search_kwargs["k"] = k
            documents = self.retriever.retrieve(question)
        
        # Generate answer - bypass chain and call LLM directly with formatted prompt
        context = format_sources(documents)
        prompt_text = self.prompt.format(context=context, question=question)
        answer = self.llm.generate(prompt_text)  # Call generate directly instead of invoke
        
        inference_time = time.time() - start_time
        
        result = {
            "answer": answer,
            "subject": self.subject,
            "inference_time": round(inference_time, 2),
            "num_sources": len(documents)
        }
        
        if return_sources:
            result["sources"] = get_unique_sources(documents)
            result["retrieved_docs"] = [
                {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        
        logger.info(f"Answered question in {inference_time:.2f}s using {len(documents)} sources")
        
        return result
    
    def batch_ask(
        self,
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch
        
        Args:
            questions: List of questions
        
        Returns:
            List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.ask(question)
            results.append(result)
        
        return results


class PracticeQuestionGenerator:
    """Generate practice questions based on topics"""
    
    def __init__(
        self,
        llm,
        retriever: Optional[EnhancedRetriever] = None,
        subject: str = "General"
    ):
        """
        Initialize practice question generator
        
        Args:
            llm: Language model instance
            retriever: Optional retriever for context
            subject: Subject category
        """
        self.llm = llm
        self.retriever = retriever
        self.subject = subject
        self.prompt = create_practice_prompt(subject)
    
    def generate(
        self,
        topic: str,
        num_questions: int = 5,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """
        Generate practice questions for a topic
        
        Args:
            topic: Topic to generate questions for
            num_questions: Number of questions to generate
            use_context: Whether to use retrieved context
        
        Returns:
            Dictionary with questions and metadata
        """
        context = ""
        
        if use_context and self.retriever:
            documents = self.retriever.retrieve(topic, k=3)
            context = format_sources(documents)
        
        # Generate questions
        prompt_text = self.prompt.format(
            subject=self.subject,
            topic=topic,
            context=context,
            num_questions=num_questions
        )
        
        questions = self.llm.invoke(prompt_text)
        
        return {
            "topic": topic,
            "subject": self.subject,
            "num_questions": num_questions,
            "questions": questions,
            "used_context": bool(context)
        }


class SolutionExplainer:
    """Provide step-by-step solutions to problems"""
    
    def __init__(
        self,
        llm,
        retriever: Optional[EnhancedRetriever] = None,
        subject: str = "General"
    ):
        """
        Initialize solution explainer
        
        Args:
            llm: Language model instance
            retriever: Optional retriever for context
            subject: Subject category
        """
        self.llm = llm
        self.retriever = retriever
        self.subject = subject
        self.prompt = create_solution_prompt(subject)
    
    def explain(
        self,
        problem: str,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """
        Generate step-by-step solution
        
        Args:
            problem: Problem statement
            use_context: Whether to use retrieved context
        
        Returns:
            Dictionary with solution and metadata
        """
        context = ""
        
        if use_context and self.retriever:
            documents = self.retriever.retrieve(problem, k=3)
            context = format_sources(documents)
        
        # Generate solution
        prompt_text = self.prompt.format(
            subject=self.subject,
            problem=problem,
            context=context
        )
        
        solution = self.llm.invoke(prompt_text)
        
        return {
            "problem": problem,
            "subject": self.subject,
            "solution": solution,
            "used_context": bool(context)
        }


def answer_query(
    question: str,
    llm,
    retriever: EnhancedRetriever,
    subject: str = "General",
    k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to answer a single query
    
    Args:
        question: User question
        llm: Language model instance
        retriever: EnhancedRetriever instance
        subject: Subject category
        k: Number of documents to retrieve
    
    Returns:
        Dictionary with answer and sources
    """
    chain = RAGChain(llm=llm, retriever=retriever, subject=subject)
    return chain.ask(question, k=k)


if __name__ == "__main__":
    # Example usage (requires initialized LLM and retriever)
    logging.basicConfig(level=logging.INFO)
    print("RAG chain components ready")
    print("\nComponents:")
    print("- RAGChain: Main Q&A chain")
    print("- PracticeQuestionGenerator: Generate practice questions")
    print("- SolutionExplainer: Step-by-step solutions")
