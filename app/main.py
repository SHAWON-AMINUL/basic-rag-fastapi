from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import generate_answer


app = FastAPI(
    title="Basic RAG API",
    version="1.0.0"
)


class ChatRequest(BaseModel):

    question: str


@app.get("/")
def root():

    return {
        "message": "Basic RAG API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = generate_answer(
        request.question
    )

    return result