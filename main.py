import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

app = FastAPI(
    title="Medical Research RAG Chatbot",
    description="Upload medical PDFs and ask questions — AI answers from your documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global state ──────────────────────────────────────────────
vectorstore = None
qa_chain = None
uploaded_files = []

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ── Load embedding model ──────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ── Load LLM ─────────────────────────────────────────────────
print("Loading language model...")
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    device=-1  # CPU
)
llm = HuggingFacePipeline(pipeline=pipe)

# ── Schemas ───────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list
    timestamp: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a medical PDF and add it to the vector store"""
    global vectorstore, qa_chain

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and split PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Add to vector store
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=VECTOR_DIR
        )
    else:
        vectorstore.add_documents(chunks)

    # Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    uploaded_files.append({
        "name": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "uploaded_at": datetime.utcnow().isoformat()
    })

    return {
        "message": f"Successfully processed {file.filename}",
        "pages": len(documents),
        "chunks": len(chunks)
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded medical documents"""
    if qa_chain is None:
        raise HTTPException(400, "Please upload a PDF first before asking questions")

    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Get answer
    result = qa_chain({"query": request.question})

    # Extract sources
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "Unknown"),
            "preview": doc.page_content[:150] + "..."
        })

    return AnswerResponse(
        answer=result["result"],
        sources=sources,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/files")
async def get_files():
    """Get list of uploaded files"""
    return {"files": uploaded_files}

@app.delete("/reset")
async def reset():
    """Reset the vector store and uploaded files"""
    global vectorstore, qa_chain, uploaded_files
    vectorstore = None
    qa_chain = None
    uploaded_files = []
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR)
    return {"message": "Reset successful"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "files_uploaded": len(uploaded_files),
        "ready": qa_chain is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
