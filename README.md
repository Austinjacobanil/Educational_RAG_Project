# 📚 EduRAG – Educational Retrieval-Augmented Generation System

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/austinja/edu_rag)
**Repository Status:** 🚀 Fully Deployed

## 📌 Overview

**EduRAG** is a domain-specific Retrieval-Augmented Generation (RAG) system designed for the **education domain**.
It combines **state-of-the-art embeddings, vector search, and LLM reasoning** to deliver accurate, context-aware answers from educational materials.

The system is optimized for **speed** using the **Groq LPU-powered inference** and is deployed on **Hugging Face Spaces** for public access.

---

## ⚙️ Features

* 📄 **Domain-Specific RAG** – optimized for educational documents
* 🧠 **Context-Aware Answers** – uses retrieved content to ground LLM responses
* ⚡ **Ultra-Fast Inference** – powered by **Groq LPU inference**
* 🔍 **Semantic Search** – vector similarity search with **ChromaDB**
* 🔗 **Embeddings** – high-quality sentence-level embeddings from HuggingFace models
* 🎯 **Chunking Strategy** – optimized for educational notes and textbooks
* 📊 **Evaluation Metrics** – retrieval accuracy, latency, and basic RAGAS evaluation
* 🌐 **Public Demo** – deployed on **Hugging Face Spaces** using Gradio SSR



## 🛠️ Technologies Used

| Component           | Technology Used                         |
| ------------------- | --------------------------------------- |
| **Frontend UI**     | Gradio (with SSR mode)                  |
| **Backend**         | Python                                  |
| **LLM Inference**   | Groq API                                |
| **Embeddings**      | HuggingFace Sentence Transformers       |
| **Vector Database** | ChromaDB                                |
| **Deployment**      | Hugging Face Spaces                     |
| **Evaluation**      | Retrieval Accuracy, Latency, RAGAS      |
| **Chunking**        | Custom strategy for educational content |



## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/username/edu_rag.git
cd edu_rag
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Environment Variables

Create a `.env` file and add:

```env
GROQ_API_KEY=your_groq_api_key
```

### 4️⃣ Run Locally

```bash
python src/app.py
```

---



