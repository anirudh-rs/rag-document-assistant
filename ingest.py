from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("document.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Done! Your document is stored.")