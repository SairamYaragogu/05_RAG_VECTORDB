
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    POLICY_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# Initialize ChromaDB
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def ingest_documents():

    pdf_files = list(POLICY_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found!")
        return

    for pdf_file in pdf_files:

        print(f"Processing: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        # Remove previous chunks of this document
        collection.delete(
            where={"source": pdf_file.name}
        )

        ids = []
        documents = []
        metadatas = []

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text() or ""

            if not page_text.strip():
                continue

            chunks = text_splitter.split_text(page_text)

            for index, chunk in enumerate(chunks):

                chunk_id = f"{pdf_file.stem}_{page_number}_{index}"

                ids.append(chunk_id)
                documents.append(chunk)

                metadatas.append({
                    "source": pdf_file.name,
                    "page": page_number
                })

        if documents:

            embeddings = embedding_model.encode(
                documents
            ).tolist()

            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            print(
                f"Successfully stored {len(documents)} chunks"
            )

        else:
            print(f"No readable text found: {pdf_file.name}")

    print("\nDocument ingestion completed!")


if __name__ == "__main__":
    ingest_documents()
