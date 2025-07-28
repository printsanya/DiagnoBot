import streamlit as st
from qa_engine import initialize_qa_chain

# Page Config
st.set_page_config(page_title="DOC_bot", page_icon="🔍", layout="wide")

# ---------- CSS Styling ----------
st.markdown("""
    <style>
        h1 {
            font-size: 3em !important;
            color: #ffffff;
            margin-bottom: 0.1em;
        }
        .subtitle {
            font-size: 1.4em;
            font-weight: 400;
            color: #cccccc;
            margin-bottom: 1.5em;
        }
        .trust-box {
            background-color: #1e1e1e;
            padding: 1.2em;
            border-radius: 12px;
            border-left: 5px solid #00c7b7;
            margin-bottom: 2em;
        }
        .trust-box p {
            margin: 0.2em 0;
            color: #dddddd;
        }
        .stTextInput > div > input {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        .answer-box {
            background-color: #111111;
            padding: 1em;
            border-radius: 8px;
            border-left: 4px solid #00c7b7;
            margin-bottom: 2em;
        }
        .snippet-box {
            background-color: #1a1a1a;
            padding: 0.8em;
            border-radius: 6px;
            margin-bottom: 1em;
            font-style: italic;
            color: #dddddd;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
    <style>
        h1 {
            font-size: 3em !important;
            color: #ffffff;
            margin-bottom: 0.1em;
        }
        .subtitle {
            font-size: 1.4em;
            font-weight: 400;
            color: #cccccc;
            margin-bottom: 1.5em;
        }
    </style>

    <div style='text-align: center;'>
        <h1>🤖 DiagnoBot</h1>
        <div class="subtitle">Support you can trust like a Doctor.</div>
    </div>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
st.sidebar.title("📌 About")
st.sidebar.markdown("**DOC_bot** is a document QA tool that uses LangChain + HuggingFace to understand and respond to questions based on your CSV file.")
st.sidebar.markdown("Built using: [LangChain](https://www.langchain.com), [HuggingFace](https://huggingface.co), and [ChromaDB](https://www.trychroma.com).")

# ---------- Load the QA Chain ----------
@st.cache_resource
def get_chain():
    return initialize_qa_chain()

qa_chain = get_chain()

# ---------- Main Input ----------
query = st.text_input("💬 Ask your question here:")

if query:
    with st.spinner("🔎 Searching for the best answer..."):
        response = qa_chain.invoke({"query": query})
        answer = response['result']
        sources = response.get("source_documents", [])

    st.markdown("### 📌 Answer")
    st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    # ---------- Source Documents ----------
    if sources:
        st.markdown("### 📚 Source Snippets")
        for i, doc in enumerate(sources, 1):
            st.markdown(f"<div class='snippet-box'><b>Snippet {i}:</b><br>{doc.page_content}</div>", unsafe_allow_html=True)
