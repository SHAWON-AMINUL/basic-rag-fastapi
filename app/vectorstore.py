from langchain_postgres import PGVector

from app.config import DATABASE_URL
from app.embeddings import embeddings


COLLECTION_NAME = "rag_documents"


vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    use_jsonb=True,
)