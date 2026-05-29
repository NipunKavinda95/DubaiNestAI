---
title: DubaiNest AI Backend
emoji: 🏙️
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

# DubaiNest AI — Backend API

RAG-powered Dubai Real Estate Assistant backend.  
Built with LangChain + OpenAI GPT-4o + ChromaDB + Flask.

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Ask a question |
| GET | `/` | API info |

## POST /chat

**Request:**
```json
{ "question": "What is the average rent for a 1BR in JVC?" }
```

**Response:**
```json
{
  "answer": "The average rent for a 1-bedroom in JVC is AED 65,000/year...",
  "question": "What is the average rent for a 1BR in JVC?"
}
```

## Setup

1. Add `OPENAI_API_KEY` in Space Settings → Variables and Secrets
2. Upload your `data/` folder (area_guide.csv, rera_rules.txt, cost_logic.txt)
3. Space builds automatically on push
