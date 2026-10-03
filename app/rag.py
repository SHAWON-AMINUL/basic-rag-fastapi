from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
)

from app.vectorstore import vectorstore


llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful university information assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context,
say that the information is not available.

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
)


def retrieve_documents(
    question: str,
    k: int = 5
):

    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents


def generate_answer(question: str):

    documents = retrieve_documents(
        question,
        k=5
    )

    if not documents:

        return {
            "answer": "I could not find relevant information.",
            "sources": []
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "answer": response.content,
        "sources": [
            document.metadata
            for document in documents
        ]
    }



#ollama pull qwen3:8b
#ollama list
#ollama run qwen3:8b
#/bye
