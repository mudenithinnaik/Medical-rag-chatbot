# 🏥 Medical Research RAG Chatbot

> **Original project by [Nithin Mude Naik](https://www.linkedin.com/in/mudenithin) © 2025**
> Project ID: MED-RAG-2025-NMN-002

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://chromadb.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 💡 What Is This?

A **Retrieval Augmented Generation (RAG)** system that lets medical professionals and researchers upload PDF documents and ask questions in plain English — getting precise answers with source citations.

> *"Instead of reading 200 pages of medical research, just ask the AI."*

---

## 🔄 How RAG Works

```
📄 Upload Medical PDF
        ↓
✂️  Split into chunks (500 tokens each)
        ↓
🧠 Embed chunks → ChromaDB vector store
        ↓
❓ User asks a question
        ↓
🔍 LangChain finds relevant chunks
        ↓
🤖 Flan-T5 generates answer from chunks
        ↓
📚 Return answer + page citations
```

---

## ✨ Features

- 📄 Upload multiple medical PDFs
- 🔍 Semantic search using sentence-transformers
- 🤖 Local LLM (Flan-T5) — no API key needed
- 📚 Source citations with page numbers
- 💬 Clean chat interface
- 🐳 Docker ready
- ⚡ FastAPI REST backend

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Python |
| LLM | Google Flan-T5 (HuggingFace) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| PDF Processing | PyPDF |
| Deployment | Docker |

---

## 🚀 Run Locally

```bash
git clone https://github.com/mudenithinnaik/medical-rag-chatbot.git
cd medical-rag-chatbot
pip install -r requirements.txt
python main.py
```

Open `http://localhost:8000`

---

## 🐳 Docker

```bash
docker build -t medical-rag .
docker run -p 8000:8000 medical-rag
```

---

## 📁 Project Structure

```
medical-rag-chatbot/
├── main.py              # FastAPI + RAG pipeline
├── templates/
│   └── index.html       # Chat UI
├── uploads/             # Uploaded PDFs
├── vectorstore/         # ChromaDB vectors
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔗 Part of My AI Health Portfolio

| Project | Description |
|---|---|
| 🐾 [AI Pet Grooming System](https://github.com/mudenithinnaik/ai-pet-grooming-system) | AI automated grooming for pets |
| 🏥 Medical RAG Chatbot (this) | AI Q&A over medical research |
| 🌿 Potato Disease Classifier | CNN plant disease detection |

---

## 👨‍💻 Author

**Nithin Mude Naik** — MS CS, St. Francis College NY
📧 nmude@sfc.edu | 🔗 [LinkedIn](https://www.linkedin.com/in/mudenithin)

---

## 📄 License

MIT License © 2025 Nithin Mude Naik
