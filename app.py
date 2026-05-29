"""
DubaiNest AI — Backend API
Deployed on HuggingFace Spaces (Docker SDK)

This file is the entry point for the HuggingFace Space.
It loads the knowledge base, builds the LangChain RAG pipeline,
and serves it via Flask on port 7860 (HuggingFace default port).
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS


# ── LangChain imports ─────────────────────────────────────────────
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()  

# ── Flask app ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Load OpenAI API key ───────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ── Build RAG pipeline on startup ────────────────────────────────
print("🔧 Building RAG pipeline...")

# 1. Load documents
DATA_FOLDER = "./data"

loaders = [
    CSVLoader(file_path=os.path.join(DATA_FOLDER, "area_guide.csv"), encoding="utf-8"),
    TextLoader(os.path.join(DATA_FOLDER, "rera_rules.txt"), encoding="utf-8"),
    TextLoader(os.path.join(DATA_FOLDER, "cost_logic.txt"), encoding="utf-8"),
]

raw_docs = []
for loader in loaders:
    raw_docs.extend(loader.load())

print(f"✅ Loaded {len(raw_docs)} documents")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["===", "\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(raw_docs)
print(f"✅ Split into {len(chunks)} chunks")

# 3. Embed and store in Chroma
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="dubainest"
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print("✅ VectorStore ready")

# 4. Build LCEL RAG chain
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=800,
    openai_api_key=OPENAI_API_KEY
)

PROMPT_TEMPLATE = """
You are DubaiNest AI, a helpful Dubai real estate assistant.
You help users with rental prices, RERA tenant laws, move-in cost
calculations, and area comparisons in Dubai.

Rules:
- Answer ONLY using the CONTEXT provided below. Do not use outside knowledge.
- If the answer is not in the context, say: "I don't have that in my
  knowledge base. Please check dubailand.gov.ae or a licensed agent."
- Always quote prices in AED.
- Be concise. Use bullet points for comparisons and lists.
- Never give legal advice. For legal matters refer users to RDSC.

CONTEXT:
{context}

USER QUESTION:
{question}

Answer based only on the context above.
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain ready!")


# ── API Routes ────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question field"}), 400
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    try:
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer, "question": question})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "bot": "DubaiNest AI"})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "DubaiNest AI",
        "description": "Dubai Real Estate RAG Assistant",
        "endpoints": {
            "POST /chat": "Ask a question",
            "GET /health": "Health check"
        }
    })


# ── Start server ──────────────────────────────────────────────────
# HuggingFace Spaces requires port 7860
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
