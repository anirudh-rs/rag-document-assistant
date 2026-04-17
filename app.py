import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.title("Ask my document anything")

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 4})

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

retriever = load_retriever()
llm = load_llm()

prompt = PromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know, say you don't know.

Context: {context}
Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs):
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1  # PyPDF is 0-indexed
        source = doc.metadata.get("source", "document.pdf")
        label = f"📄 {source} — Page {page}"
        if label not in sources:
            sources.append(label)
    return sources

# Chain that returns both answer and source documents
rag_chain = RunnableParallel(
    answer=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ),
    docs=retriever
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(s)

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = rag_chain.invoke(question)
    answer = result["answer"]
    sources = get_sources(result["docs"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(s)