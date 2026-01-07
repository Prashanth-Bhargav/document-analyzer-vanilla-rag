# 📄 Vanilla RAG Document Analyzer

A **classic Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask **semantic questions** about their content using a Large Language Model.

---

## 🚀 Features

- Upload a PDF document via a Streamlit UI
- Split documents into semantic chunks
- Generate embeddings using a sentence-transformer model
- Store and retrieve document chunks using a vector database
- Answer user questions using LLM-based reasoning
- Grounded responses based strictly on document context
- Graceful fallback when information is not present

---

## ▶️ How to Run

### 1. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

###2. Install dependencies

pip install -r requirements.txt

###3. Start the application

streamlit run main.py

⚠ Note: Ensure Ollama is installed and a compatible LLaMA 3.x model is available locally.
The vector database (chroma_db/) is generated at runtime and is not included in the repo.

---

## 🧠 What this project demonstrates

This project focuses on **core RAG concepts**, including:

- Document ingestion and preprocessing
- Semantic vector search
- Context-aware answer generation
- Prompt grounding to reduce hallucinations
- Understanding the strengths and limitations of vanilla RAG

It is intentionally scoped to remain **simple, explainable, and production-aligned**.

---

## ❗ Limitations (by design)

This is a **vanilla RAG system**, which means:

- Exact string-based entities (e.g., email IDs, URLs) may not always be retrieved reliably
- PDF header content may be inconsistently extracted depending on formatting
- No deterministic field extraction or schema enforcement is applied
- No agent-based routing or tool usage is implemented

These limitations are **known characteristics of embedding-based semantic retrieval** and are intentionally left unhandled to preserve a pure RAG architecture.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – Web UI
- **LangChain** – RAG orchestration
- **HuggingFace Sentence Transformers** – Embeddings
- **Chroma** – Vector database
- **Ollama (LLaMA 3.x)** – Local LLM inference

---

## 📦 Project Structure

```text
.
├── main.py              # Streamlit application
├── chroma_db/           # Persisted vector store
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation


