import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CENTROIDS_PATH = Path(os.getenv("CENTROIDS_PATH", BASE_DIR / "centroids.pkl"))
THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.35"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

ALLOWED_CS_TERMS = [
    "cs", "it", "is", "cpe", "bscs", "bsit", "bsis", "bscpe",
    "computer science", "information technology", "information systems",
    "computer engineering", "software engineering", "programmer", "coding",
    "developer", "data science", "cybersecurity", "tech", "computing",
]

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only assist with career questions related to Computer Studies "
    "(such as BSCS, BSIT, BSIS, and BSCpE programs). As a specialized computing career "
    "counselor, I am unable to provide guidance on other fields, degree programs, or general topics."
)

app = FastAPI(title="CareerBot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000,http://localhost:4173").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []

class ChatResponse(BaseModel):
    answer: str
    intent: str
    score: float


def load_centroids() -> dict[str, np.ndarray]:
    if not CENTROIDS_PATH.exists():
        raise RuntimeError(f"Centroid artifact not found: {CENTROIDS_PATH}")
    with CENTROIDS_PATH.open("rb") as artifact:
        centroids = pickle.load(artifact)
    return {name: np.asarray(vector, dtype=np.float32) for name, vector in centroids.items()}


centroids = load_centroids()
model = SentenceTransformer(MODEL_NAME)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"]) if os.getenv("GROQ_API_KEY") else None


def embed_texts(texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def classify_by_centroid(embedding: np.ndarray) -> tuple[str, float]:
    intent_names = list(centroids)
    centroid_matrix = np.stack([centroids[name] for name in intent_names])
    scores = embedding @ centroid_matrix.T
    best_index = int(scores.argmax())
    score = float(scores[best_index])
    intent = intent_names[best_index] if score >= THRESHOLD else "out_of_scope"
    return intent, score


def is_computer_studies_query(message: str) -> bool:
    normalized = message.lower()
    return any(term in normalized for term in ALLOWED_CS_TERMS)


def generate_answer(message: str, intent: str) -> str:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not configured")
    system_prompt = (
        "You are CareerBot, an expert AI career counselor exclusively for Computer Studies "
        "students (BSCS, BSIT, BSIS, BSCpE). The user's query has been classified under "
        f"the intent '{intent}'. Provide professional, concise guidance under 3 paragraphs "
        "strictly tailored to computing and technology fields."
    )
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return completion.choices[0].message.content or REFUSAL_MESSAGE


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "centroids": len(centroids),
        "groq_configured": groq_client is not None,
        "embedding_model": MODEL_NAME,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    if not is_computer_studies_query(message):
        return ChatResponse(answer=REFUSAL_MESSAGE, intent="out_of_scope", score=0.0)

    intent, score = classify_by_centroid(embed_texts([message])[0])
    if intent == "out_of_scope":
        return ChatResponse(answer=REFUSAL_MESSAGE, intent=intent, score=score)

    try:
        answer = generate_answer(message, intent)
    except Exception:
        answer = REFUSAL_MESSAGE
    return ChatResponse(answer=answer, intent=intent, score=score)


app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="frontend")
