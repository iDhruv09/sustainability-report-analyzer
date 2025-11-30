# -------------------------------------------------------
# app.py (COMPLETE & FINAL VERSION)
# -------------------------------------------------------

import io
import time
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv, find_dotenv
import re
import os
import pandas as pd
import plotly.express as px
# -------------------------------------------------------
# LOAD ENV + INITIALIZE GEMINI LLM
# -------------------------------------------------------
# -------------------------------------------------------
# LOAD ENV + INITIALIZE GEMINI LLM
# -------------------------------------------------------

api_key = st.secrets["Google_API_Key"]





llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    model_provider="google_genai",
    google_api_key=api_key
)


def call_llm(system_prompt, user_prompt):
    """Unified Gemini API call"""
    prompt = system_prompt + "\n\n" + user_prompt
    response = llm.invoke(prompt)
    return response.content


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Sustainability Report Analyzer",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()




# Remove top padding + adjust layout alignment
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
         /* optional: hides default Streamlit header */
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# PDF → TEXT WITH FOOTER PAGE EXTRACTION
# -------------------------------------------------------

def detect_footer_page_number(text):
    """Extract printed footer page numbers from PDF text."""
    patterns = [
        r"Page\s+(\d+)",
        r"^\s*(\d+)\s*$",
        r"(\d+)\s*$",
        r"^\s*(\d+)\s"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def pdf_to_text(file_bytes: bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for pdf_page_num, page in enumerate(reader.pages, start=1):
        full_text = page.extract_text() or ""
        footer_page = detect_footer_page_number(full_text)

        pages.append({
            "pdf_page": pdf_page_num,
            "footer_page": footer_page,
            "text": full_text
        })

    return pages


# -------------------------------------------------------
# CLEANING + CHUNKING
# -------------------------------------------------------

def clean_text(text):
    text = text.replace("\x00", " ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def chunk_text(pages, chunk_size=800, overlap=200):
    chunks = []

    for p in pages:
        words = p["text"].split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text_section = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text_section,
                "pdf_page": p["pdf_page"],
                "footer_page": p["footer_page"]
            })

            if end == len(words):
                break

            start += chunk_size - overlap

    return chunks


# -------------------------------------------------------
# FAISS VECTOR INDEX
# -------------------------------------------------------

def build_faiss_index(chunks):
    embeddings = embed_model.encode([c["text"] for c in chunks])
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve_top_k(query, index, chunks, k=6):
    if index is None or not chunks:
        return []
    q_emb = embed_model.encode([query]).astype("float32")
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]


# -------------------------------------------------------
# SUMMARY GENERATION
# -------------------------------------------------------

def summarize_report(chunks, max_chunks=12):
    if not chunks:
        return "No content available."

    selected = chunks[:max_chunks]

    context = "\n\n".join(
        f"[Page {c['footer_page'] if c['footer_page'] else '?'}] {c['text']}"
        for c in selected
    )

    system_prompt = "You are an expert ESG sustainability analyst."
    user_prompt = f"""
CONTEXT:
{context}

TASK:
Summarize the report in 5–10 bullet points.
For EACH bullet point, include the footer page number like: (Page X)
"""

    return call_llm(system_prompt, user_prompt)


# -------------------------------------------------------
# RAG CHATBOT
# -------------------------------------------------------

def build_history_text(history):
    return "\n".join(
        f"{'User' if h['role']=='user' else 'Assistant'}: {h['content']}"
        for h in history
    )


def answer_query_with_rag(question, index, chunks, history=None):
    if history is None:
        history = []

    retrieved = retrieve_top_k(question, index, chunks, k=8)

    context = "\n\n".join(c["text"] for c in retrieved) if retrieved else "NO DATA FOUND."

    history_text = build_history_text(history)

    # Detect if user wants page numbers
    page_request = any(
        kw in question.lower()
        for kw in ["page", "page number", "which page", "what page", "page no", "show page", "footer"]
    )

    system_prompt = (
        "You are an ESG sustainability report analyst. "
        "Use ONLY the provided context. "
        "Do NOT hallucinate."
    )

    user_prompt = f"""
PREVIOUS CONVERSATION:
{history_text}

CONTEXT:
{context}

QUESTION:
{question}

TASK:
Answer ONLY using information in the context.
"""

    if page_request:
        footer_pages = sorted({c["footer_page"] for c in retrieved if c["footer_page"]})
        if footer_pages:
            user_prompt += f"""

The relevant information appears on these footer page numbers:
{', '.join(str(p) for p in footer_pages)}.

Include the footer page numbers in your answer like:
(Page X).
"""
        else:
            user_prompt += """
No footer page numbers were detected for this answer.
Mention this in your response.
"""

    return call_llm(system_prompt, user_prompt)




def extract_tables_from_pdf(file_bytes):
    import pdfplumber
    tables = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            page_tables = p.extract_tables()
            for t in page_tables:
                df = pd.DataFrame(t[1:], columns=t[0])
                tables.append(df)

    return tables

def get_numeric_cols(df):
    return df.select_dtypes(include=['int64','float64']).columns.tolist()

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.title("🌿 Sustainability Report Analyzer")
st.write("Upload ESG / Sustainability reports, generate summaries, and chat using RAG.")

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------- SIDEBAR ----------------------------

with st.sidebar:
    st.header("🗂️ Upload Reports")

    uploaded_files = st.file_uploader(
        "Upload ESG / Sustainability PDF Reports",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("🔍 Process & Index Reports"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF!")
        else:
            with st.spinner("Processing reports..."):

                all_pages = []

                for f in uploaded_files:
                    file_bytes = f.read()

                    # 1️⃣ Extract text pages
                    pages = pdf_to_text(file_bytes)

                    # 2️⃣ Clean and store pages
                    cleaned_pages = [
                        {
                            "pdf_page": p["pdf_page"],
                            "footer_page": p["footer_page"],
                            "text": clean_text(p["text"])
                        }
                        for p in pages
                    ]
                    all_pages.extend(cleaned_pages)

                    # 3️⃣ Extract NUMERIC TABLES for dashboard (NEW)
                    if "tables" not in st.session_state:
                        st.session_state.tables = []

                    # IMPORTANT LINE YOU ASKED ABOUT:
                    st.session_state.tables.extend(extract_tables_from_pdf(file_bytes))


                chunks = chunk_text(all_pages)
                index, _ = build_faiss_index(chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.summary = None
                st.session_state.chat_history = []

                st.success(f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")


# -------------------------------------------------------
# TABS UI (SUMMARY | CHAT | TABLES)
# -------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📌 Report Summary",
    "💬 Chat with the Report",
    "📊 Extracted Tables"
])

# ---------------------------- TAB 1 ----------------------------
with tab1:
    st.subheader("📌 Report Summary")

    if st.session_state.index is None:
        st.info("Upload and process reports first.")
    else:
        if st.session_state.summary is None:
            if st.button("🧾 Generate Summary"):
                with st.spinner("Generating summary..."):
                    st.session_state.summary = summarize_report(st.session_state.chunks)

        if st.session_state.summary:
            st.markdown(st.session_state.summary)


# ---------------------------- TAB 2 ----------------------------
with tab2:
    st.header("💬 Chat with the Report")

    if st.session_state.index is None:
        st.info("Upload and index report(s) to start chatting.")
    else:

        # 1️⃣ Display messages FIRST
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 2️⃣ Chat input stays at the BOTTOM, like ChatGPT
        user_q = st.chat_input("Ask a question about the report...")

        if user_q:
            # Save user message
            st.session_state.chat_history.append({"role": "user", "content": user_q})

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_q)

            # Generate assistant reply
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    ans = answer_query_with_rag(
                        user_q,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.chat_history
                    )
                st.markdown(ans)

            # Save assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": ans})

            # IMPORTANT: rerun so chat_input moves down
            st.rerun()

# ---------------------------- TAB 3 ----------------------------
with tab3:
    st.header("📊 Extracted Data Dashboard")

    if "tables" not in st.session_state or len(st.session_state.tables) == 0:
        st.info("No numeric tables detected in the PDF.")
    else:
        table_index = st.selectbox(
            "Choose a table to visualize:",
            list(range(len(st.session_state.tables))),
            format_func=lambda i: f"Table {i+1}"
        )

        df = st.session_state.tables[table_index]

        st.write("### Table Preview")
        st.dataframe(df)

        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("This table has no numeric columns to visualize.")
        else:
            chart_type = st.selectbox(
                "Choose chart type:",
                ["Line", "Bar", "Pie", "Histogram"]
            )

            if chart_type == "Line":
                st.line_chart(df[numeric_cols])

            elif chart_type == "Bar":
                st.bar_chart(df[numeric_cols])

            elif chart_type == "Pie":
                if len(numeric_cols) == 1:
                    fig = px.pie(df, names=df.columns[0], values=numeric_cols[0])
                    st.plotly_chart(fig)
                else:
                    st.warning("Pie chart requires exactly 1 numeric column.")

            elif chart_type == "Histogram":
                fig = px.histogram(df, x=numeric_cols[0])
                st.plotly_chart(fig)







