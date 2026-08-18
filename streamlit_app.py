import streamlit as st
import pymupdf, os, hashlib
import chunker
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Document
from google import genai
from google.genai import types

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")

# ---------------------------------------------------------------------------
# Clients — cached so they're created once per session, not on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url="https://6f8ba350-1bba-4a5a-8c45-d619bb10120a.sa-east-1-0.aws.cloud.qdrant.io:6333",
        api_key=st.secrets["QDRANT_API_KEY"],
        cloud_inference=True,
        timeout=120
    )

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)

# ---------------------------------------------------------------------------
# Let the user optionally supply their own Gemini API key.
# Falls back to the app's own key (from secrets) if they leave it blank.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")
    user_gemini_key = st.text_input(
        "Your Gemini API key (optional)",
        type="password",
        help="Leave blank to use the app's default key."
    )

active_gemini_key = user_gemini_key.strip() if user_gemini_key.strip() else st.secrets["GEMINI_API_KEY"]

if user_gemini_key.strip():
    st.sidebar.caption("Using your API key.")
else:
    st.sidebar.caption("Using the app's default API key.")

client_qdrant = get_qdrant_client()
gemini_client = get_gemini_client(active_gemini_key)

# ---------------------------------------------------------------------------
# Let the user optionally upload their own PDF.
# Falls back to the app's default PDF if they don't upload one.
# ---------------------------------------------------------------------------
DEFAULT_PDF_PATH = "iesc1ps_merged.pdf"
DEFAULT_COLLECTION = "PDF_Chatbot"

with st.sidebar:
    uploaded_pdf = st.file_uploader("Upload your own PDF (optional)", type="pdf")
    if st.button("Clear chat"):
        st.session_state.conversation = []
        st.session_state.display_history = []
        st.rerun()

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    # Derive a stable, unique collection name from the file's contents so
    # the same PDF always maps to the same collection, and different PDFs
    # never collide with each other or with the default collection.
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    active_collection = f"PDF_Chatbot_{file_hash}"
    st.sidebar.caption(f"Using your PDF: {uploaded_pdf.name}")
else:
    pdf_bytes = None
    active_collection = DEFAULT_COLLECTION
    st.sidebar.caption("Using the app's default PDF.")

# Reset the conversation whenever the active PDF changes, since old answers
# were grounded in a different document.
if st.session_state.get("active_collection") != active_collection:
    st.session_state.conversation = []
    st.session_state.display_history = []
    st.session_state.active_collection = active_collection

# ---------------------------------------------------------------------------
# One-time ingestion per collection — cached so re-uploading the same PDF
# (same hash) skips re-ingestion, but a new PDF triggers a fresh run.
# ---------------------------------------------------------------------------
@st.cache_resource
def ingest_pdf(collection_name, pdf_source):
    if client_qdrant.collection_exists(collection_name):
        return ""

    if pdf_source is None:
        doc = pymupdf.open(DEFAULT_PDF_PATH)
    else:
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")

    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append({"page_number": page_num, "text": text})
    doc.close()

    chunks = chunker.chunker(pages, 100)

    client_qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    points = [
        PointStruct(id=i, vector=Document(text=chunk["text"], model=MODEL), payload=chunk)
        for i, chunk in enumerate(chunks)
    ]

    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client_qdrant.upsert(collection_name=collection_name, points=batch)

    return f"Ingested {len(points)} chunks."

with st.spinner("Preparing knowledge base..."):
    status = ingest_pdf(active_collection, pdf_bytes)
st.caption(status)

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []   # sent to Gemini (includes context blocks)
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # what's shown in the chat UI

SYSTEM_INSTRUCTION = """Answer the question using only the provided context. Do not write "Based on the provided context".
Also do not express insufficiency of context. Just answer based on whatever's provided.
"""

# Render past turns
for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
query = st.chat_input("What do you wanna know?")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.display_history.append({"role": "user", "text": query})

    results = client_qdrant.query_points(
        collection_name=active_collection,
        query=Document(text=query, model=MODEL),
        with_payload=True,
        limit=15
    ).points

    context = "\n".join(r.payload["text"] for r in results)

    prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    st.session_state.conversation.append({"role": "user", "parts": [{"text": prompt}]})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=st.session_state.conversation,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            )
        st.markdown(response.text)

    st.session_state.conversation.append({"role": "model", "parts": [{"text": response.text}]})
    st.session_state.display_history.append({"role": "assistant", "text": response.text})
