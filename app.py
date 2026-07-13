import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")

PERSIST_DIR = "chroma-db"


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use only the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """,
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}""",
        ),
    ]
)

# ---- session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


def process_pdf(uploaded_file):
    """Save uploaded PDF, load -> split -> embed -> store in Chroma."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
    )

    os.remove(tmp_path)
    return vectorstore, len(chunks)


# ---- sidebar: upload & process ----
with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process document", use_container_width=True):
            with st.spinner("Reading, chunking, and embedding your PDF..."):
                vectorstore, n_chunks = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Processed '{uploaded_file.name}' into {n_chunks} chunks.")

    if st.session_state.processed_file:
        st.info(f"Active document: **{st.session_state.processed_file}**")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- main chat area ----
st.title("💬 Chat with your PDF")
st.caption("Upload a PDF in the sidebar, process it, then ask questions below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.vectorstore is None:
    st.info("👈 Upload and process a PDF to start chatting.")
else:
    query = st.chat_input("Ask a question about your document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = prompt.invoke({"context": context, "question": query})

        llm = get_llm()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(final_prompt)
                st.markdown(response.content)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )

        with st.expander("🔎 Retrieved context"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)
                st.divider()