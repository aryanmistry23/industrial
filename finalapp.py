import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(Path(__file__).parent / ".env")

def get_config(key):
    value = st.secrets.get(key, os.getenv(key))
    if not value:
        st.error(f"Missing configuration: {key}")
        st.stop()
    return value

url = get_config("WATSONX_URL")
api_key = get_config("WATSONX_APIKEY")
project_id = get_config("WATSONX_PROJECT_ID")

llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=url,
    project_id=project_id,
    api_key=api_key,
    params={
        "max_new_tokens": 512,
        "temperature": 0.7
    }
)
PDF_DIR = Path(__file__).parent / "documents"

if not PDF_DIR.exists():
    st.error("Documents folder not found!")
    st.stop()

pdf_files = list(PDF_DIR.glob("*.pdf"))

if not pdf_files:
    st.error("No PDF files found in documents folder!")
    st.stop()

loader = PyPDFDirectoryLoader(str(PDF_DIR))
documents = loader.load()

if not documents:
    st.error("PDF files found, but no documents were extracted!")
    st.stop()
documents = [
    doc for doc in documents
    if doc.page_content.strip()
]

if not documents:
    st.error("No readable text found in your PDFs!")
    st.stop()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
if not chunks:
    st.error("No text chunks generated!")
    st.stop()

st.success(
    f"Loaded {len(pdf_files)} PDFs and generated {len(chunks)} chunks."
)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
embeddings = load_embeddings()
@st.cache_resource
def create_vectorstore(_chunks):
    return FAISS.from_documents(
        _chunks,
        embeddings
    )

vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

st.title("Industrial QA Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(
        message["content"]
    )
prompt = st.chat_input("Ask your industrial question...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.spinner("Searching documents..."):
        try:
            relevant_docs = retriever.invoke(prompt)
            context = "\n\n".join(
                doc.page_content for doc in relevant_docs
            )

            full_prompt = f"""
You are an industrial technical assistant.

Answer using only the provided context.
If the answer is not available, say:
"The information is not available in the provided documents."

Context:
{context}

Question:
{prompt}

Answer:
"""
            with st.spinner("Generating answer ..."):
                response = llm.invoke(full_prompt)
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"Error generating answer: {e}")
