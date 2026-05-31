import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI API Key ────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")  

# ── LlamaIndex imports ────────────────────────────────────────────
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

# ── Flask app ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Build RAG pipeline on startup ────────────────────────────────
print("🔧 Building RAG pipeline (LlamaIndex)...")

# ── Step 1: Configure LlamaIndex global settings ──────────────────
# LlamaIndex uses Settings instead of passing to each component
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=800,
    api_key=OPENAI_API_KEY
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
Settings.transformations = [
    SentenceSplitter(
        chunk_size=500,    # max characters per chunk
        chunk_overlap=50   # overlap to preserve context at boundaries
    )
]

# ── Step 2: Load documents from data/ folder ─────────────────────
# SimpleDirectoryReader reads ALL files in the folder automatically
# Supports CSV, TXT, PDF, and more — no need for separate loaders
DATA_FOLDER = "./data"

documents = SimpleDirectoryReader(
    input_dir=DATA_FOLDER,
    required_exts=[".csv", ".txt"],   # only load these file types
    recursive=False
).load_data()

print(f"✅ Loaded {len(documents)} documents")

# ── Step 3: Build VectorStoreIndex with Pinecone ─────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dubainest-ai"

# Create index if it doesn't exist
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)
print("✅ Pinecone VectorStore ready")

# ── Step 4: Build Query Engine with custom prompt ─────────────────
# Query engine = retriever + prompt + LLM combined
PROMPT_TEMPLATE = """\
You are DubaiNest AI, a helpful Dubai real estate assistant.
You help users with rental prices, RERA tenant laws, move-in cost
calculations, and area comparisons in Dubai.

Rules:
- Answer ONLY using the CONTEXT provided below. Do not use outside knowledge.
- If the answer is not in the context, say: "I don't have that in my \
knowledge base. Please check dubailand.gov.ae or a licensed agent."
- Always quote prices in AED. If user asks for another currency, convert AED values.
- Be concise. Use bullet points for comparisons and lists.
- Never give legal advice. For legal matters refer users to RDSC.

CONTEXT:
{context_str}

USER QUESTION:
{query_str}

Answer based only on the context above.
"""

# LlamaIndex uses {context_str} and {query_str} — different from LangChain
qa_prompt = PromptTemplate(PROMPT_TEMPLATE)

query_engine = index.as_query_engine(
    similarity_top_k=4,          # retrieve top 4 chunks
    text_qa_template=qa_prompt,  # apply custom prompt
    streaming=False
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
        # LlamaIndex uses .query() instead of .invoke()
        response = query_engine.query(question)
        return jsonify({
            "answer": str(response),   # convert response object to string
            "question": question
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "bot": "DubaiNest AI (LlamaIndex)"})


@app.route("/", methods=["GET"])
def index_route():
    html = open("static/index.html", encoding="utf-8").read()
    html = html.replace("__API_BASE_URL__", "")
    return render_template_string(html)


# ── Start server ──────────────────────────────────────────────────
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=7860)
