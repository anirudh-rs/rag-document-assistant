import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS_FOLDER = "./documents"

# Load all PDFs from the documents folder
all_chunks = []
pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in the documents folder.")
else:
    for filename in pdf_files:
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        print(f"Loading: {filename}")
        loader = PyPDFLoader(filepath)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    # Wipe old DB and rebuild fresh
    import shutil
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")
            print("Old database cleared.")
        except PermissionError:
            print("⚠️ Could not delete chroma_db — it may be in use.")
            print("Please stop your Streamlit app (Ctrl+C) and run ingest.py again.")
            exit()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")
    print("Done! All documents are stored and ready.")