import os
import io
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain (latest versions compatible)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from pypdf import PdfReader

# Load environment variables
load_dotenv()

# ---------------- CONFIG ---------------- #
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))

st.set_page_config(page_title="Course Notes Q&A Bot", page_icon="📚", layout="wide")
st.title("📚 Course Notes Q&A Bot")
st.caption("Upload notes → Build FAISS index → Ask questions → Get answers using Groq LLM")


# ---------------- HELPERS ---------------- #

def load_pdf(file: io.BytesIO) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except:
            pass
    return text


def load_text(file_bytes: bytes) -> str:
    """Load raw text file."""
    try:
        return file_bytes.decode("utf-8")
    except:
        try:
            return file_bytes.decode("latin-1")
        except:
            return ""


def files_to_documents(uploaded_files) -> List[Document]:
    docs = []
    for uf in uploaded_files:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()

        # PDF
        if ext == ".pdf":
            text = load_pdf(uf)

        # TXT / MD
        elif ext in {".txt", ".md"}:
            text = load_text(uf.getvalue())

        else:
            st.warning(f"Unsupported file type: {name}")
            continue

        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name}))
        else:
            st.warning(f"No text extracted from: {name}")

    return docs


def chunk_documents(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_or_load_vectorstore(chunks=None):
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Build new index
    if chunks:
        vs = FAISS.from_documents(chunks, get_embeddings())
        vs.save_local(INDEX_DIR)
        return vs

    # Load existing index
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


# ---------------- RAG PIPELINE (LCEL) ---------------- #

def answer_query(vs, query):
    """Retrieve docs & generate answer with Groq LLM."""
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    docs = retriever.invoke(query)

    # Build context manually
    context = "\n\n".join([d.page_content for d in docs])

    system_prompt = (
        "You are a helpful teaching assistant. "
        "Use ONLY the provided context to answer. "
        "If answer is not found, say you don't know.\n\n"
        f"Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=MODEL_NAME,
        temperature=0.2
    )

    chain = prompt | llm
    response = chain.invoke({"question": query})

    return response.content, docs


# ---------------- SIDEBAR UI ---------------- #

st.sidebar.header("Build / Update Index")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or Text files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

col1, col2 = st.sidebar.columns(2)
with col1: rebuild = st.button("Rebuild Index")
with col2: clear_idx = st.button("Clear Index")


# Clear index
if clear_idx:
    if os.path.exists(INDEX_DIR):
        for root, _, files in os.walk(INDEX_DIR):
            for f in files:
                os.remove(os.path.join(root, f))
        st.success("Index cleared!")


# Rebuild index
if rebuild:
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        docs = files_to_documents(uploaded_files)
        chunks = chunk_documents(docs)
        vs = build_or_load_vectorstore(chunks)
        st.session_state["vs_ready"] = True
        st.success(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")


# Load index if exists
vs = None
try:
    vs = build_or_load_vectorstore()
    st.session_state["vs_ready"] = True
except:
    st.info("No index found. Upload files and rebuild.")


# ---------------- MAIN CHAT UI ---------------- #

st.subheader("Ask Questions")
query = st.text_input("Enter your question:")

if "history" not in st.session_state:
    st.session_state["history"] = []


if st.button("Ask") and query:
    if not st.session_state.get("vs_ready"):
        st.warning("Build the index first.")
    else:
        answer, source_docs = answer_query(vs, query)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for d in source_docs:
            snippet = d.page_content[:200].replace("\n", " ")
            st.markdown(f"- `{d.metadata['source']}` — {snippet}...")

        st.session_state["history"].append({"q": query, "a": answer})


# Display chat history
if st.session_state["history"]:
    st.markdown("---")
    st.markdown("### Chat History")

    for turn in reversed(st.session_state["history"]):
        st.markdown(f"**Q:** {turn['q']}")
        st.markdown(f"**A:** {turn['a']}")
