import os
import json
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# ── LlamaIndex imports ────────────────────────────────────────────
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

# ── Flask app ─────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dubainest-secret-key-change-in-prod")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ── Build RAG pipeline on startup ────────────────────────────────
print("🔧 Building RAG pipeline (LlamaIndex + Agent)...")

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
    SentenceSplitter(chunk_size=500, chunk_overlap=50)
]

DATA_FOLDER = "./data"
documents = SimpleDirectoryReader(
    input_dir=DATA_FOLDER,
    required_exts=[".csv", ".txt"],
    recursive=False
).load_data()
print(f"✅ Loaded {len(documents)} documents")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dubainest-ai"
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(index_name)
vector_store   = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)
print("✅ Pinecone VectorStore ready")

# ── Standard RAG prompt ───────────────────────────────────────────
STANDARD_PROMPT = """\
You are DubaiNest AI, a helpful Dubai real estate assistant.
Answer ONLY using the CONTEXT below. Do not use outside knowledge.
If the answer is not in context, say: "I don't have that in my knowledge base. Please check dubailand.gov.ae or a licensed agent."
Always quote prices in AED. Be concise. Use bullet points for lists. Never give legal advice — refer to RDSC.

CONTEXT:
{context_str}

USER QUESTION:
{query_str}

Answer based only on the context above. At the end of your answer, on a new line write:
FOLLOWUPS: [suggest 2-3 short follow-up questions the user might ask next, as a JSON array of strings]
"""

# ── Shortlist agent prompt ────────────────────────────────────────
SHORTLIST_PROMPT = """\
You are DubaiNest AI, a Dubai real estate assistant helping a user find the best area to live.

The user's preferences are:
- Budget: {budget}
- Lifestyle / Who's moving: {lifestyle}
- Area preference or must-haves: {area_pref}

Using ONLY the CONTEXT below, recommend the TOP 3 areas that best match these preferences.

CONTEXT:
{context_str}

Return your answer as a JSON object with this exact structure (no markdown, no extra text):
{{
  "shortlist": [
    {{
      "rank": 1,
      "area": "Area Name",
      "avg_rent_1br": "AED XX,XXX/yr",
      "avg_rent_2br": "AED XX,XXX/yr",
      "metro_access": "Yes / No / Nearby",
      "vibe": "One sentence describing the area feel",
      "best_for": "Who this suits",
      "reason": "Why this matches the user's preferences"
    }}
  ],
  "summary": "One sentence overall recommendation"
}}
"""

qa_prompt = PromptTemplate(STANDARD_PROMPT)

query_engine = index.as_query_engine(
    similarity_top_k=4,
    text_qa_template=qa_prompt,
    streaming=False
)

shortlist_engine = index.as_query_engine(
    similarity_top_k=6,
    streaming=False
)

print("✅ RAG chain + Agent ready!")


# ── Intent detection ──────────────────────────────────────────────
SHORTLIST_KEYWORDS = [
    "find me", "recommend", "suggest", "which area", "where should i",
    "best area", "suitable area", "help me find", "looking for",
    "where to live", "shortlist", "top areas", "good area for"
]

def is_shortlist_intent(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in SHORTLIST_KEYWORDS)


# ── Parse follow-ups from standard response ───────────────────────
def parse_followups(answer: str):
    """Extract FOLLOWUPS JSON from answer text, return (clean_answer, followups_list)."""
    followups = []
    clean = answer
    if "FOLLOWUPS:" in answer:
        parts = answer.split("FOLLOWUPS:", 1)
        clean = parts[0].strip()
        try:
            raw = parts[1].strip()
            followups = json.loads(raw)
        except Exception:
            followups = []
    return clean, followups


# ── API Routes ────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question field"}), 400

    question     = data["question"].strip()
    agent_state  = data.get("agent_state", {})   # frontend passes state back
    mode         = agent_state.get("mode", "normal")

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # ── SHORTLIST AGENT FLOW ──────────────────────────────────────
    if mode == "shortlist_collecting":
        step = agent_state.get("step", 1)

        if step == 1:
            # Just collected budget
            return jsonify({
                "type": "agent_question",
                "message": "Got it! 👥 Who's moving in? *(e.g. single professional, young couple, family with kids)*",
                "agent_state": {
                    "mode": "shortlist_collecting",
                    "step": 2,
                    "budget": question,
                    "lifestyle": "",
                    "area_pref": ""
                }
            })

        elif step == 2:
            # Just collected lifestyle
            return jsonify({
                "type": "agent_question",
                "message": "Almost there! 🗺️ Any area preference or must-have? *(e.g. near metro, close to beach, quiet neighbourhood — or say 'no preference')*",
                "agent_state": {
                    "mode": "shortlist_collecting",
                    "step": 3,
                    "budget": agent_state.get("budget", ""),
                    "lifestyle": question,
                    "area_pref": ""
                }
            })

        elif step == 3:
            # All collected — run shortlist query
            budget    = agent_state.get("budget", "not specified")
            lifestyle = agent_state.get("lifestyle", "not specified")
            area_pref = question

            shortlist_q = f"Areas in Dubai for budget {budget}, lifestyle {lifestyle}, preference {area_pref}"
            raw_prompt = SHORTLIST_PROMPT.format(
                budget=budget,
                lifestyle=lifestyle,
                area_pref=area_pref,
                context_str="{context_str}",
                query_str="{query_str}"
            )
            custom_prompt = PromptTemplate(raw_prompt)
            shortlist_engine.update_prompts({"response_synthesizer:text_qa_template": custom_prompt})

            try:
                response = shortlist_engine.query(shortlist_q)
                raw = str(response).strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                result = json.loads(raw.strip())
                return jsonify({
                    "type": "shortlist",
                    "shortlist": result.get("shortlist", []),
                    "summary": result.get("summary", ""),
                    "agent_state": {"mode": "normal"},
                    "followups": [
                        "What's the move-in cost for my top pick?",
                        "Can my landlord raise rent in this area?",
                        "Compare the top 2 areas side by side"
                    ]
                })
            except Exception as e:
                return jsonify({"error": f"Shortlist generation failed: {str(e)}"}), 500

    # ── DETECT SHORTLIST INTENT ───────────────────────────────────
    if mode == "normal" and is_shortlist_intent(question):
        return jsonify({
            "type": "agent_question",
            "message": "Great, let me build you a personalised shortlist! 🏙️\n\n💰 **What's your annual budget?** *(e.g. AED 60,000, AED 120,000)*",
            "agent_state": {
                "mode": "shortlist_collecting",
                "step": 1,
                "budget": "",
                "lifestyle": "",
                "area_pref": ""
            }
        })

    # ── STANDARD RAG CHAT ─────────────────────────────────────────
    try:
        response  = query_engine.query(question)
        answer    = str(response)
        clean, followups = parse_followups(answer)

        # Fallback chips if LLM didn't generate any
        if not followups:
            followups = [
                "What's the average rent in that area?",
                "What are move-in costs?",
                "Is this area close to metro?"
            ]

        return jsonify({
            "type": "answer",
            "answer": clean,
            "question": question,
            "followups": followups,
            "agent_state": {"mode": "normal"}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "bot": "DubaiNest AI v2 (Agent + RAG)"})


@app.route("/", methods=["GET"])
def index_route():
    html = open("static/index.html", encoding="utf-8").read()
    html = html.replace("__API_BASE_URL__", "")
    return render_template_string(html)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=7860)
