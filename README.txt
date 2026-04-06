# RAG-Based Document Assistant

## What it is
A chatbot that answers questions about any PDF document you feed it.
Uses Retrieval Augmented Generation (RAG) to find the most relevant
parts of your document and passes them to an LLM to generate accurate,
grounded answers with source citations.

## Why it matters
LLMs hallucinate when asked about private or specific documents.
RAG fixes this by giving the model only the relevant context it needs,
making answers accurate and traceable to a source.

## Tech stack
- Python, LangChain (LCEL chain)
- ChromaDB (local vector database)
- OpenAI text-embedding-3-small (embeddings)
- GPT-4o (answer generation)
- Streamlit (chat UI)

## Project folder
C:\Users\aniru\OneDrive\Desktop\RAG

## Files
- .env              → stores OpenAI API key
- ingest.py         → loads PDF, chunks it, stores in ChromaDB
- app.py            → Streamlit chat interface
- chroma_db/        → local vector database (auto-generated)

## How to run
1. Add credits to OpenAI account (platform.openai.com)
2. Drop a PDF into the RAG folder, name it document.pdf
3. Run: python ingest.py
4. Run: streamlit run app.py

## Key fixes applied during build
- langchain_text_splitters instead of langchain.text_splitter
- langchain_core.prompts instead of langchain.prompts
- Used LCEL chain instead of deprecated RetrievalQA

## Status: Complete (local)

## Next steps
- Support multiple PDFs at once
- Add a file uploader in the browser UI
- Deploy to Streamlit Cloud (free)
- Swap GPT-4o for Claude

## Difficulty: Easy-Medium