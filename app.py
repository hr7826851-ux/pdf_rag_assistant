from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Load API Key
# =========================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

print("API key loaded:", bool(API_KEY))


# =========================
# Gemini Model
# =========================
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=API_KEY,
    temperature=0
)


# =========================
# Load PDF
# =========================
pdf_path = "documents/GRU.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"PDF loaded: {len(docs)} pages")


# =========================
# Split PDF into chunks
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")


# =========================
# Ask Question
# =========================
question = input("\nAsk a question about the PDF: ")


# =========================
# Create PDF Context
# =========================
context = "\n\n".join(
    doc.page_content for doc in chunks
)


# =========================
# Generate Answer
# =========================
prompt = f"""
Answer the question using only the information provided
in the PDF context below.

PDF Context:
{context}

Question:
{question}
"""

response = model.invoke(prompt)

print("\nAnswer:")
print(response.content)