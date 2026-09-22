import os
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================
   MAIN PAGE
========================= */

.stApp {
    background-color: #F4F1E8;
}

.block-container {
    max-width: 1200px;
    padding-top: 40px;
    padding-bottom: 50px;
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background-color: #10251C;
}

section[data-testid="stSidebar"] * {
    color: #F4F1E8;
}


/* =========================
   HEADINGS
========================= */

h1 {
    color: #10251C !important;
    font-size: 48px !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #10251C !important;
}


/* =========================
   NORMAL TEXT
========================= */

p {
    color: #66736B;
}


/* =========================
   INPUT BOX
========================= */

.stTextInput input {
    background-color: #FFFFFF !important;
    color: #17221C !important;

    border: 1px solid #CFC8BA !important;

    border-radius: 10px !important;

    padding: 14px !important;

    font-size: 15px !important;
}


/* Input focus */

.stTextInput input:focus {
    border-color: #A06D23 !important;

    box-shadow:
        0 0 0 2px rgba(160, 109, 35, 0.12) !important;
}


/* =========================
   BUTTON
========================= */

.stButton > button {
    background-color: #10251C !important;

    color: #FFFFFF !important;

    border: none !important;

    border-radius: 9px !important;

    padding: 10px 25px !important;

    font-weight: 700 !important;

    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #A06D23 !important;

    color: #FFFFFF !important;
}


/* =========================
   CARDS
========================= */

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    background-color: #FFFFFF;

    border: 1px solid #DDD7C9;

    border-radius: 14px;

    padding: 18px;

    box-shadow:
        0 5px 18px rgba(20, 35, 28, 0.06);
}


/* =========================
   ANSWER BOX
========================= */

.answer-container {
    background-color: #FFFFFF;

    border: 1px solid #DDD7C9;

    border-left: 4px solid #A06D23;

    border-radius: 12px;

    padding: 22px;

    margin-top: 10px;

    box-shadow:
        0 5px 18px rgba(20, 35, 28, 0.06);
}


/* =========================
   EXPANDER
========================= */

div[data-testid="stExpander"] {
    border: 1px solid #DDD7C9;

    border-radius: 10px;

    background-color: #FFFFFF;
}


/* =========================
   SIDEBAR DIVIDER
========================= */

section[data-testid="stSidebar"] hr {
    border-color: rgba(215, 168, 75, 0.3);
}


/* =========================
   FOOTER
========================= */

.footer-text {
    text-align: center;

    color: #8A918B;

    font-size: 12px;

    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📖 PDF RAG")

    st.subheader("Assistant")

    st.caption(
        "Document intelligence using "
        "Retrieval-Augmented Generation."
    )

    st.divider()

    st.markdown("### 📄 Current Document")

    st.markdown("**GRU.pdf**")

    st.caption(
        "Vector Store: ChromaDB\n\n"
        "Retrieval: MMR\n\n"
        "Embedding: Gemini\n\n"
        "Generation: Gemini Flash"
    )

    st.divider()

    st.info(
        "Answers are generated from "
        "retrieved content in the PDF."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.caption("DOCUMENT INTELLIGENCE")

st.title("📖 PDF RAG Assistant")

st.write(
    "Ask questions about your document and receive "
    "context-aware answers using ChromaDB, "
    "LangChain and Gemini."
)

st.divider()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )


embedding_model = load_embedding_model()


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_vector_database():

    return Chroma(
        collection_name="gru_documents",
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )


vector_db = load_vector_database()


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.caption("01")

    st.subheader("Ask")

    st.write(
        "Ask natural-language questions "
        "about your document."
    )


with col2:

    st.caption("02")

    st.subheader("Retrieve")

    st.write(
        "Relevant information is retrieved "
        "from ChromaDB."
    )


with col3:

    st.caption("03")

    st.subheader("Understand")

    st.write(
        "MMR selects useful and diverse "
        "document chunks."
    )


with col4:

    st.caption("04")

    st.subheader("Answer")

    st.write(
        "Gemini generates a contextual "
        "answer from the retrieved data."
    )


st.write("")


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    return ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        google_api_key=api_key
    )


llm = load_gemini()


# ============================================================
# QUESTION SECTION
# ============================================================

st.subheader("Ask your document")

st.caption(
    "Enter a question and the system will search "
    "the relevant sections of your PDF."
)


question = st.text_input(
    "Question",
    placeholder="Example: What is the main topic of this document?",
    label_visibility="collapsed"
)


ask_button = st.button(
    "🔍 Search Document"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching document and generating answer..."
        ):

            # Retrieve relevant chunks

            retrieved_docs = retriever.invoke(
                question
            )


            # Create context

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )


            # Create prompt

            prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer is not available in the context,
say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{context}

User Question:
{question}
"""


            # Generate answer

            response = llm.invoke(
                prompt
            )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.subheader("Generated Answer")

        st.info(
            response.text
        )


        # ====================================================
        # RETRIEVED SOURCES
        # ====================================================

        with st.expander(
            "🔎 View Retrieved PDF Sections"
        ):

            for i, doc in enumerate(
                retrieved_docs,
                1
            ):

                st.markdown(
                    f"**Retrieved Section {i}**"
                )

                st.write(
                    doc.page_content
                )

                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PDF RAG Assistant • ChromaDB • LangChain • Gemini"
)