import os
import json
import secrets
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://nipunkavindaai-dubainestai.hf.space"
).split(",")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or secrets.token_hex(32)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

print("Building RAG pipeline...")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, max_tokens=800, api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
Settings.transformations = [SentenceSplitter(chunk_size=500, chunk_overlap=50)]

documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".csv", ".txt"], recursive=False).load_data()
print(f"Loaded {len(documents)} documents")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dubainest-ai"
existing = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing:
    pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

pinecone_index  = pc.Index(index_name)
vector_store    = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Check if Pinecone already has our vectors — avoid re-embedding/re-upserting on every restart
stats = pinecone_index.describe_index_stats()
existing_vector_count = stats.get("total_vector_count", 0)

if existing_vector_count > 0:
    print(f"Found {existing_vector_count} existing vectors in Pinecone — loading from vector store (no re-embedding)")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
else:
    print("Pinecone index is empty — embedding documents for the first time")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

print("Pinecone ready")

STANDARD_PROMPT = """\
You are DubaiNest AI, a helpful Dubai real estate assistant.
Answer ONLY using the CONTEXT below. Do not use outside knowledge.
If the answer is not in context, say: "I don't have that in my knowledge base. Please check dubailand.gov.ae or a licensed agent."
Always quote prices in AED. Be concise. Use bullet points for lists. Never give legal advice - refer to RDSC.

CONTEXT:
{context_str}

USER QUESTION:
{query_str}

Answer based only on the context above. At the end of your answer, on a new line write:
FOLLOWUPS: [suggest 2-3 short follow-up questions the user might ask next, as a JSON array of strings]
"""

SHORTLIST_PROMPT = """\
You are DubaiNest AI helping a user find the best area to live in Dubai.

User preferences:
- Budget: {budget}
- Lifestyle: {lifestyle}
- Area preference: {area_pref}

Using ONLY the CONTEXT below, recommend TOP 3 matching areas.

CONTEXT:
{context_str}

USER QUESTION:
{query_str}

Return ONLY this JSON, no markdown:
{{
  "shortlist": [
    {{
      "rank": 1,
      "area": "Area Name",
      "avg_rent_1br": "AED XX,XXX/yr",
      "avg_rent_2br": "AED XX,XXX/yr",
      "metro_access": "Yes / No",
      "vibe": "One sentence",
      "best_for": "Who this suits",
      "reason": "Why this matches"
    }}
  ],
  "summary": "One sentence overall recommendation"
}}
"""

qa_prompt = PromptTemplate(STANDARD_PROMPT)

query_engine = index.as_query_engine(similarity_top_k=4, text_qa_template=qa_prompt, streaming=False)
shortlist_engine = index.as_query_engine(similarity_top_k=6, streaming=False)

print("RAG ready!")

SHORTLIST_KEYWORDS = ["find me", "recommend", "suggest", "which area", "where should i",
    "best area", "suitable area", "help me find", "looking for",
    "where to live", "shortlist", "top areas", "good area for"]

def is_shortlist_intent(q):
    return any(kw in q.lower() for kw in SHORTLIST_KEYWORDS)

def parse_followups(answer):
    followups = []
    clean = answer
    if "FOLLOWUPS:" in answer:
        parts = answer.split("FOLLOWUPS:", 1)
        clean = parts[0].strip()
        try:
            followups = json.loads(parts[1].strip())
        except Exception:
            followups = []
    return clean, followups


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    question    = data["question"].strip()
    agent_state = data.get("agent_state", {})
    mode        = agent_state.get("mode", "normal")

    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Shortlist agent flow
    if mode == "shortlist_collecting":
        step = agent_state.get("step", 1)
        if step == 1:
            return jsonify({
                "type": "agent_question",
                "message": "Got it! Who's moving in? (e.g. single professional, young couple, family with kids)",
                "agent_state": {"mode": "shortlist_collecting", "step": 2, "budget": question, "lifestyle": "", "area_pref": ""}
            })
        elif step == 2:
            return jsonify({
                "type": "agent_question",
                "message": "Almost there! Any area preference or must-have? (e.g. near metro, close to beach — or say no preference)",
                "agent_state": {"mode": "shortlist_collecting", "step": 3, "budget": agent_state.get("budget", ""), "lifestyle": question, "area_pref": ""}
            })
        elif step == 3:
            budget    = agent_state.get("budget", "not specified")
            lifestyle = agent_state.get("lifestyle", "not specified")
            area_pref = question
            shortlist_q = f"Areas in Dubai for budget {budget}, lifestyle {lifestyle}, preference {area_pref}"
            raw_prompt  = SHORTLIST_PROMPT.format(
                budget=budget, lifestyle=lifestyle, area_pref=area_pref,
                context_str="{context_str}", query_str="{query_str}"
            )
            shortlist_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(raw_prompt)})
            try:
                response = shortlist_engine.query(shortlist_q)
                raw = str(response).strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"): raw = raw[4:]
                result = json.loads(raw.strip())
                return jsonify({
                    "type": "shortlist",
                    "shortlist": result.get("shortlist", []),
                    "summary": result.get("summary", ""),
                    "agent_state": {"mode": "normal"},
                    "followups": ["What is the move-in cost?", "Can my landlord raise rent?", "Compare the top 2 areas"]
                })
            except Exception as e:
                return jsonify({"error": f"Shortlist failed: {str(e)}"}), 500

    # Detect shortlist intent
    if mode == "normal" and is_shortlist_intent(question):
        return jsonify({
            "type": "agent_question",
            "message": "Great, let me build you a personalised shortlist!\n\nWhat is your annual budget? (e.g. AED 60,000 or AED 120,000)",
            "agent_state": {"mode": "shortlist_collecting", "step": 1, "budget": "", "lifestyle": "", "area_pref": ""}
        })

    # Standard RAG
    try:
        response = query_engine.query(question)
        answer   = str(response)
        clean, followups = parse_followups(answer)
        if not followups:
            followups = ["What is the average rent in that area?", "What are move-in costs?", "Is this area close to metro?"]
        return jsonify({"type": "answer", "answer": clean, "question": question, "followups": followups, "agent_state": {"mode": "normal"}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "bot": "DubaiNest AI v2"})


@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json()
    if not data or "area1" not in data or "area2" not in data:
        return jsonify({"error": "Missing area1 or area2"}), 400

    area1 = data["area1"].strip()
    area2 = data["area2"].strip()

    try:
        q1 = f"What is the average rent for studio, 1BR, 2BR in {area1}? What is metro access, community vibe, and who is it best suited for?"
        q2 = f"What is the average rent for studio, 1BR, 2BR in {area2}? What is metro access, community vibe, and who is it best suited for?"

        r1 = str(query_engine.query(q1))
        r2 = str(query_engine.query(q2))

        if "FOLLOWUPS:" in r1: r1 = r1.split("FOLLOWUPS:")[0].strip()
        if "FOLLOWUPS:" in r2: r2 = r2.split("FOLLOWUPS:")[0].strip()

        return jsonify({
            "type": "compare",
            "area1": {"name": area1.title(), "text": r1},
            "area2": {"name": area2.title(), "text": r2}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index_route():
    html = open("static/index.html", encoding="utf-8").read()
    return render_template_string(html.replace("__API_BASE_URL__", ""))


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=7860)