from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import os, time, json, re, logging
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ CONFIG ------------------
app = Flask(__name__)
app.secret_key = "replace_this_secret"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ LOAD EMBEDDINGS + VECTOR DB ------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
os.makedirs("./chroma_db", exist_ok=True)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Sample policies for vector DB
PRODUCT_POLICIES = [
    {
        "product": "Laptop",
        "policy": "Product: Laptop\nReturn Period: 30 days\nConditions: Must be in original packaging, no physical damage, all accessories included.\nSpecial Notes: Software issues are covered under warranty, not return policy."
    },
    {
        "product": "Smartphone",
        "policy": "Product: Smartphone\nReturn Period: 15 days\nConditions: Original packaging, no scratches or damage, all accessories included.\nSpecial Notes: Activated phones cannot be returned."
    },
    {
        "product": "Headphones",
        "policy": "Product: Headphones\nReturn Period: 45 days\nConditions: Unopened or defective units only.\nSpecial Notes: Defective items can be exchanged within 1 year."
    },
    {
        "product": "Clothing",
        "policy": "Product: Clothing\nReturn Period: 60 days\nConditions: Tags attached, unworn, unwashed.\nSpecial Notes: Final sale items cannot be returned."
    },
    {
        "product": "Books",
        "policy": "Product: Books\nReturn Period: 30 days\nConditions: No damage, writing, or torn pages.\nSpecial Notes: Digital books cannot be returned."
    },
    {
        "product": "Furniture",
        "policy": "Product: Furniture\nReturn Period: 90 days\nConditions: Must be unassembled and undamaged.\nSpecial Notes: Shipping fees may apply."
    },
    {
        "product": "Tablet",
        "policy": "Product: Tablet\nReturn Period: 30 days\nConditions: Original packaging, no damage.\nSpecial Notes: Screen protectors and cases must be unopened."
    },
    {
        "product": "Camera",
        "policy": "Product: Camera\nReturn Period: 30 days\nConditions: Original packaging, no damage.\nSpecial Notes: Shutter count under 1000."
    },
]

# Create vector DB if not exists
def build_vector_db():
    docs = [Document(page_content=p["policy"], metadata={"product": p["product"]}) for p in PRODUCT_POLICIES]
    return Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")

def load_vector_db():
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

if not os.path.exists("./chroma_db/index"):
    vector_db = build_vector_db()
else:
    vector_db = load_vector_db()

# ------------------ LLM SETUP ------------------
logger.info("Loading LLM...")
MODEL_NAME = "tiiuae/falcon-7b-instruct"  # free instruct-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, temperature=0.6, do_sample=True)
logger.info("LLM loaded ✅")

# ------------------ UTILITIES ------------------
def extract_date(text):
    text = text.lower()
    now = datetime.now()
    if "yesterday" in text: return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in text: return now.strftime("%Y-%m-%d")
    if "week" in text: return (now - timedelta(days=7)).strftime("%Y-%m-%d")
    match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    return match.group(1) if match else ""

def extract_condition(text):
    text = text.lower()
    if any(k in text for k in ["new", "unopened", "sealed"]): return "new"
    if any(k in text for k in ["used", "opened", "worn"]): return "used"
    if any(k in text for k in ["broken", "damaged", "faulty"]): return "damaged"
    return ""

def find_product(text):
    text = text.lower()
    for p in [p["product"].lower() for p in PRODUCT_POLICIES]:
        if p in text:
            return p
    return ""

def ask_llm(prompt):
    resp = llm(prompt)[0]["generated_text"]
    return resp.split("Response:")[-1].strip() if "Response:" in resp else resp.strip()

def fetch_policy(product):
    retriever = load_vector_db().as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(product)
    return "\n\n".join([d.page_content for d in docs]) if docs else "No policy found."

# ------------------ MAIN CHAT LOGIC ------------------
@app.route("/")
def index():
    if "history" not in session:
        session["history"] = []
    products = [p["product"] for p in PRODUCT_POLICIES]
    welcome = f"👋 Hey! Ask me anything about these products: {', '.join(products)}."
    return render_template("index.html", welcome_message=welcome, products=products)

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json()
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    # Extract info
    product = find_product(msg)
    date = extract_date(msg)
    cond = extract_condition(msg)

    # If missing info, ask for it
    missing = []
    if not product: missing.append("product")
    if "return" in msg and not date: missing.append("purchase_date")
    if "return" in msg and not cond: missing.append("condition")
    if missing:
        if "product" in missing:
            products = ", ".join([p["product"] for p in PRODUCT_POLICIES])
            reply = f"Sure! Please tell me which product you’re referring to. I can help with: {products}."
        elif "purchase_date" in missing:
            reply = "When did you purchase the item?"
        elif "condition" in missing:
            reply = "Could you describe the item’s condition? (new, used, damaged)"
        else:
            reply = "Could you share more details so I can check your return eligibility?"
        return jsonify({"response": reply})

    # Fetch relevant policy
    policy_text = fetch_policy(product)

    # Compose LLM prompt
    context = f"""
You are ReturnBot, a helpful assistant that explains return policies and eligibility.

Policy info:
{policy_text}

User message: {msg}

If the user mentions purchase date and condition, determine eligibility.
Otherwise, directly answer questions using the above policy.
Give clear, conversational response without templates.
"""
    reply = ask_llm(context)
    session["history"].append({"user": msg, "bot": reply})
    return jsonify({"response": reply})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["history"] = []
    products = [p["product"] for p in PRODUCT_POLICIES]
    welcome = f"👋 Hey! Ask me anything about these products: {', '.join(products)}."
    return jsonify({"success": True, "welcome_message": welcome})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
