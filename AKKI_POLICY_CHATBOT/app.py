
import streamlit as st

from rag import generate_answer
from config import COLLECTION_NAME, CHROMA_DIR

import chromadb


# Page configuration
st.set_page_config(
    page_title="AKKI Policy Assistant",
    page_icon="👕",
    layout="centered"
)


# Custom styling
st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.main-title {
    text-align: center;
    color: #1e293b;
    font-size: 30px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="main-title">AKKI LATEST COLLECTIONS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Employee Policy Assistant</div>',
    unsafe_allow_html=True
)

st.divider()


# Check document availability
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


if collection.count() == 0:

    st.warning(
        "No policy documents available. "
        "Please add PDFs and run ingest.py first."
    )

    st.stop()


# Initialize chat history
if "messages" not in st.session_state:

    st.session_state.messages = []


# Sidebar
with st.sidebar:

    st.title("Settings")

    st.write("Company: AKKI LATEST COLLECTIONS")

    st.write("Model: Llama 3.2")

    st.write("Vector DB: ChromaDB")

    st.write(f"Stored chunks: {collection.count()}")

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            for source in message.get("sources", []):

                st.caption(
                    f"Source: {source['source']} "
                    f"| Page: {source['page']}"
                )


# Chat input
question = st.chat_input(
    "Ask anything about company policies..."
)


if question:

    # Display employee question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Searching company policies..."):

            try:

                answer, sources = generate_answer(question)

                st.markdown(answer)

                if sources:

                    st.markdown("**Reference Documents:**")

                    unique_sources = {
                        (s["source"], s["page"])
                        for s in sources
                    }

                    for source, page in sorted(unique_sources):

                        st.caption(
                            f"{source} | Page {page}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )
