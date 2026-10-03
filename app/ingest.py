from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore import vectorstore


DOCUMENT_DIR = Path("documents")


def load_documents():

    documents = []

    for file_path in DOCUMENT_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        document = Document(
            page_content=text,

            metadata={
                "source": str(file_path)
            }
        )

        documents.append(document)

    return documents


def ingest_documents():

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    vectorstore.add_documents(
        chunks
    )

    print("Documents added to PostgreSQL")


if __name__ == "__main__":
    ingest_documents()







    #terminal command : #python -m app.ingesta