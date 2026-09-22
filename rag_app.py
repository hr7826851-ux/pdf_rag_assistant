import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# ==================================================
# 1. Load API Key
# ==================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

print("API key loaded:", True)


# ==================================================
# 2. Load PDF
# ==================================================

pdf_path = "documents/GRU.pdf"

loader = PyPDFLoader(pdf_path)

docs = loader.load()

print(f"PDF loaded: {len(docs)} pages")


# ==================================================
# 3. Split PDF into chunks
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")


# ==================================================
# 4. Create Google Embedding Model
# ==================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

print("Embedding model loaded")


# ==================================================
# 5. Store chunks + embeddings in ChromaDB
# ==================================================

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="gru_documents",
    persist_directory="./chroma_db"
)

print("Chroma database created successfully!")


# ==================================================
# 6. Check stored documents
# ==================================================

print(
    "Documents stored:",
    vector_db._collection.count()
)


# ==================================================
# 7. Create Retriever
# ==================================================

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever ready")


# ==================================================
# 8. Create Gemini Model
# ==================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

print("Gemini model ready")


# ==================================================
# 9. Ask Questions
# ==================================================

while True:

    question = input(
        "\nAsk a question about the PDF "
        "(type 'exit' to quit): "
    )

    if question.lower() == "exit":
        print("\nProgram closed.")
        break


    # ==================================================
    # 10. Retrieve relevant documents
    # ==================================================

    relevant_docs = retriever.invoke(question)

    print(
        f"\nRelevant chunks found: "
        f"{len(relevant_docs)}"
    )


    # ==================================================
    # 11. Create Context
    # ==================================================

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )


    # ==================================================
    # 12. Create Prompt
    # ==================================================

    prompt = f"""
Answer the question using only the information
provided in the PDF context below.

If the answer is not available in the PDF context,
say:

"I could not find the answer in the provided PDF."

PDF Context:
{context}

Question:
{question}
"""


    # ==================================================
    # 13. Generate Answer
    # ==================================================

    response = model.invoke(prompt)


    # ==================================================
    # 14. Display Answer
    # ==================================================

    print("\n========================================")
    print("ANSWER")
    print("========================================")

    print(response.text)

    print("========================================")