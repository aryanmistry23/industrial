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
print("url:",os.getenv("WATSONX_URL"))
print("apikey:",os.getenv("WATSONX_APIKEY"))
print("project_id:",bool(os.getenv("WATSONX_PROJECT_ID")))
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=st.secrets["WATSONX_URL"],
    project_id=st.secrets["WATSONX_PROJECT_ID"],
    api_key=st.secrets["WATSONX_APIKEY"],
    params={
        "max_new_tokens": 512,
        "temperature": 0.7,
    }
)

loader = PyPDFDirectoryLoader("documents")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

st.title("Ask Watsonx")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    
    relevant_docs = retriever.invoke(prompt)

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    full_prompt = f"""
    You are an industrial technical assistant.

    Answer the question using only the context below.
    If the answer is not found in the context,
    say "The information is not available
    in the provided documents."

    Context:
    {context}

    Question:
    {prompt}

    Answer:
    """
    with st.spinner("Searching documents and generating answer..."):
        try:
            response = llm.invoke(full_prompt)

            st.chat_message("assistant").markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        except Exception as e:
            st.error(f"Error: {e}")
