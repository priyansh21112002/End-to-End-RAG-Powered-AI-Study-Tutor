"""
Prompt templates for AI Tutor
Optimized for educational interactions with clear, step-by-step explanations
"""

from typing import Optional
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


# System prompts for different subjects
SYSTEM_PROMPTS = {
    "ML": """You are an expert AI tutor specializing in Machine Learning. Your role is to help students understand ML concepts clearly and thoroughly.

Guidelines:
- Use the provided context to answer questions accurately
- Explain concepts step-by-step with examples
- Break down complex topics into digestible parts
- Use analogies when helpful
- Provide mathematical intuition when relevant
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "DL": """You are an expert AI tutor specializing in Deep Learning and Neural Networks. Your role is to help students understand DL concepts clearly.

Guidelines:
- Use the provided context to answer questions accurately
- Explain neural network architectures and training processes step-by-step
- Visualize concepts through clear descriptions
- Connect theory to practical applications
- Explain mathematical concepts intuitively
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "DSA": """You are an expert AI tutor specializing in Data Structures and Algorithms. Your role is to help students master DSA concepts.

Guidelines:
- Use the provided context to answer questions accurately
- Explain algorithms step-by-step with clear logic
- Analyze time and space complexity
- Provide pseudocode or code examples when helpful
- Explain trade-offs between different approaches
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "Mathematics": """You are an expert AI tutor specializing in Mathematics. Your role is to help students understand mathematical concepts clearly.

Guidelines:
- Use the provided context to answer questions accurately
- Show step-by-step derivations and proofs
- Explain the intuition behind formulas
- Provide visual descriptions when applicable
- Connect abstract concepts to concrete examples
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "Statistics": """You are an expert AI tutor specializing in Statistics and Probability. Your role is to help students understand statistical concepts.

Guidelines:
- Use the provided context to answer questions accurately
- Explain statistical concepts with real-world examples
- Break down probability problems step-by-step
- Clarify common misconceptions
- Show calculations clearly
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "Python": """You are an expert AI tutor specializing in Python programming. Your role is to help students learn Python effectively.

Guidelines:
- Use the provided context to answer questions accurately
- Explain code line-by-line when needed
- Provide clear, idiomatic Python examples
- Explain best practices and common pitfalls
- Show both simple and advanced approaches
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "General": """You are an expert AI tutor helping students learn various subjects. Your role is to provide clear, accurate educational support.

Guidelines:
- Use the provided context to answer questions accurately
- Adapt your explanation style to the topic
- Break down complex concepts into simple parts
- Use examples and analogies
- Encourage critical thinking
- If the context doesn't contain enough information, say "I don't have enough information on that topic in my knowledge base."
- Always cite sources when using specific information from the context""",

    "Thesis": """You are an expert research assistant helping to understand and explain thesis content. Your role is to provide clear, detailed explanations of the research work.

Guidelines:
- Use the provided context to answer questions accurately about the thesis
- Cite specific sections, chapters, and page numbers from the thesis
- Explain research methodology, findings, and conclusions clearly
- Summarize complex research concepts in accessible language
- Identify key contributions and research gaps discussed
- If the context doesn't contain the requested information, say "I don't have that information in the thesis context provided."
- Always reference specific parts of the thesis when answering"""
}


def get_system_prompt(subject: str) -> str:
    """
    Get system prompt for a specific subject
    
    Args:
        subject: Subject category (ML/DL/DSA/etc)
    
    Returns:
        System prompt string
    """
    return SYSTEM_PROMPTS.get(subject, SYSTEM_PROMPTS["General"])


# RAG prompt template
RAG_TEMPLATE = """Use the following context to answer the student's question. If you cannot answer the question based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer (be clear, step-by-step, and educational):"""


def create_rag_prompt(subject: str = "General") -> PromptTemplate:
    """
    Create a RAG prompt template with subject-specific system message
    
    Args:
        subject: Subject category
    
    Returns:
        PromptTemplate for RAG (combined system + user prompt)
    """
    system_prompt = get_system_prompt(subject)
    
    # Combine system prompt and RAG template into single prompt
    combined_template = f"""{system_prompt}

{RAG_TEMPLATE}"""
    
    prompt = PromptTemplate(
        template=combined_template,
        input_variables=["context", "question"]
    )
    
    return prompt


# Simple prompt for direct Q&A (without RAG)
QA_TEMPLATE = """You are an AI tutor for {subject}. Answer the following question clearly and educationally.

Question: {question}

Answer:"""


def create_qa_prompt(subject: str = "General") -> PromptTemplate:
    """
    Create a simple Q&A prompt without context retrieval
    
    Args:
        subject: Subject category
    
    Returns:
        PromptTemplate for Q&A
    """
    return PromptTemplate(
        input_variables=["subject", "question"],
        template=QA_TEMPLATE
    )


# Practice question generation prompt
PRACTICE_PROMPT = """Based on the following topic, generate {num_questions} practice questions suitable for a student learning {subject}.

Topic: {topic}

Context (optional):
{context}

Generate questions that:
1. Test understanding of key concepts
2. Range from basic to intermediate difficulty
3. Include both conceptual and applied questions
4. Are clear and unambiguous

Practice Questions:"""


def create_practice_prompt(subject: str = "General") -> PromptTemplate:
    """
    Create prompt for generating practice questions
    
    Args:
        subject: Subject category
    
    Returns:
        PromptTemplate for practice question generation
    """
    return PromptTemplate(
        input_variables=["subject", "topic", "context", "num_questions"],
        template=PRACTICE_PROMPT
    )


# Step-by-step solution prompt
SOLUTION_PROMPT = """Provide a detailed, step-by-step solution to the following {subject} problem. Break down the solution into clear steps that a student can follow.

Problem: {problem}

Relevant Context:
{context}

Step-by-Step Solution:"""


def create_solution_prompt(subject: str = "General") -> PromptTemplate:
    """
    Create prompt for step-by-step solutions
    
    Args:
        subject: Subject category
    
    Returns:
        PromptTemplate for solutions
    """
    return PromptTemplate(
        input_variables=["subject", "problem", "context"],
        template=SOLUTION_PROMPT
    )


# Code explanation prompt
CODE_EXPLANATION_PROMPT = """Explain the following code in detail. Break it down line-by-line if necessary, and explain what it does, how it works, and any important concepts.

Code:
```
{code}
```

Context (if available):
{context}

Explanation:"""


def create_code_explanation_prompt() -> PromptTemplate:
    """
    Create prompt for explaining code
    
    Returns:
        PromptTemplate for code explanation
    """
    return PromptTemplate(
        input_variables=["code", "context"],
        template=CODE_EXPLANATION_PROMPT
    )


def format_sources(documents: list) -> str:
    """
    Format retrieved documents as context string
    
    Args:
        documents: List of Document objects from retriever
    
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("url", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        subject = doc.metadata.get("subject", "N/A")
        
        context_parts.append(
            f"[Source {i}: {source}, Page: {page}, Subject: {subject}]\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


if __name__ == "__main__":
    # Test prompts
    print("=== Testing Prompt Templates ===\n")
    
    # Test RAG prompt
    rag_prompt = create_rag_prompt("ML")
    print("RAG Prompt for ML:")
    print(rag_prompt.format(
        context="Gradient descent is an optimization algorithm...",
        question="What is gradient descent?"
    ))
    
    print("\n" + "="*50 + "\n")
    
    # Test practice prompt
    practice_prompt = create_practice_prompt("DSA")
    print("Practice Question Prompt:")
    print(practice_prompt.format(
        subject="DSA",
        topic="Binary Search",
        context="Binary search is a divide-and-conquer algorithm...",
        num_questions=3
    ))
