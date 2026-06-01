---
title: DubaiNest AI
emoji: 🏙️
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: true
license: mit
short_description: Dubai Real Estate Assistant
---

# 🏙️ DubaiNest AI — Dubai Real Estate Assistant

> An end-to-end RAG-powered AI chatbot that answers questions about Dubai rental prices, area comparisons, RERA tenant rights, and move-in costs — built as part of the AI Accelerator Bootcamp by Decoding Data Science.

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-yellow)](https://huggingface.co/spaces/nipunkavindaAI/DubaiNestAI)
[![GitHub](https://img.shields.io/badge/GitHub-DubaiNestAI-black?logo=github)](https://github.com/NipunKavinda95/DubaiNestAI)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-purple)](https://www.llamaindex.ai)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green?logo=openai)](https://openai.com)

---

## 🎯 What it does

DubaiNest AI is a production-ready AI assistant that helps property seekers, expats, and investors in Dubai navigate one of the world's most complex real estate markets — through a simple chat interface.

**Ask it things like:**

- _"What is the average rent for a 1BR in JVC?"_
- _"Can my landlord increase rent by 20%?"_
- _"What is the total move-in cost for a AED 90,000 flat?"_
- _"Compare JVC vs Dubai Marina for a young professional"_
- _"Which areas have metro access under AED 80,000/year?"_

---

## 🏗️ Architecture

```
User Question
      ↓
Flask API (Waitress WSGI)
      ↓
LlamaIndex Query Engine
      ↓
OpenAI Embeddings (text-embedding-3-small)
      ↓
Pinecone Vector Store — semantic search → top 4 chunks
      ↓
Custom Prompt Template + GPT-4o-mini
      ↓
Answer → Chat UI (served from Flask)
```

---

## 🛠️ Tech Stack

### Core AI & RAG

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-6B47ED?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOSA1IDktNXYtNWwtOSA1LTktNXoiLz48L3N2Zz4=&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00BFA5?style=for-the-badge&logo=pinecone&logoColor=white)

### Backend & Deployment

![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Spaces-FFD21E?style=for-the-badge)

### Frontend

![HTML5](https://img.shields.io/badge/HTML5-UI-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

### Also Used

![LangChain](https://img.shields.io/badge/LangChain-Backup_Pipeline-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Git](https://img.shields.io/badge/Git-Version_Control-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📁 Project Structure

```
DubaiNestAI/
├── app_llama.py          # LlamaIndex RAG pipeline (active)
├── app.py                # LangChain RAG pipeline (backup)
├── Dockerfile            # Docker build config
├── requirements.txt      # Python dependencies
├── README.md
├── .gitignore
├── static/
│   └── index.html        # Chat UI
└── data/
    ├── area_guide.csv    # Dubai area rental data (20+ areas)
    ├── rera_rules.txt    # RERA tenant law rules
    └── cost_logic.txt    # Move-in cost calculations
```

---

## 📊 Knowledge Base

The AI answers are grounded in a structured knowledge base covering:

- **20+ Dubai areas** — JVC, Dubai Marina, Downtown, Business Bay, Palm Jumeirah, Dubai Hills, and more
- **Rental prices** — Studio, 1BR, 2BR, 3BR, Villa by area (2025 data)
- **RERA regulations** — Rent increase rules, tenant rights, eviction laws, dispute resolution
- **Move-in cost logic** — Security deposit, agency fee, DEWA, chiller, total calculations

---

## 🚀 Run Locally

**1. Clone the repo**

```bash
git clone https://github.com/NipunKavinda95/DubaiNestAI.git
cd DubaiNestAI
```

**2. Create `.env` file**

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=...
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
python app.py
```

**5. Open browser**

```
http://localhost:7860
```

---

## 🔄 LangChain → LlamaIndex Migration

This project was intentionally built with both frameworks to demonstrate the migration path:

|                  | LangChain (`app_lang.py`)        | LlamaIndex (`app.py`)         |
| ---------------- | -------------------------------- | ----------------------------- |
| **Loader**       | `CSVLoader, TextLoader`          | `SimpleDirectoryReader`       |
| **Splitter**     | `RecursiveCharacterTextSplitter` | `SentenceSplitter`            |
| **Vector Store** | ChromaDB (local)                 | Pinecone (cloud)              |
| **Query**        | `rag_chain.invoke()`             | `query_engine.query()`        |
| **Prompt vars**  | `{context}` `{question}`         | `{context_str}` `{query_str}` |

Switch between them in `Dockerfile`:

```dockerfile
CMD ["python", "app.py"]   # LlamaIndex (active)
CMD ["python", "app_lang.py"]         # LangChain (backup)
```

---

## 🌐 Deployment

Deployed on **HuggingFace Spaces** using Docker SDK.

|              | Detail                                                    |
| ------------ | --------------------------------------------------------- |
| **Live URL** | https://huggingface.co/spaces/nipunkavindaAI/DubaiNestAI  |
| **Port**     | 7860 (HuggingFace default)                                |
| **Server**   | Waitress (production WSGI)                                |
| **Secrets**  | `OPENAI_API_KEY`, `PINECONE_API_KEY` via HF Space Secrets |

---

## 👨‍💻 Built By

**Nipun Kavinda**
I am Nipun Kavinda, a Mechanical Automation & Maintenance Engineer now specialising in Industrial AI and intelligent automation. My engineering foundation gives me something most software developers do not have — a deep understanding of real-world physical systems, machinery, and control logic.
Dubai, UAE 🇦🇪

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/nipunkavinda)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/NipunKavinda95)