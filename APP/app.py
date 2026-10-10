"""
T5 Dialogue Summarizer API
Production-ready FastAPI service for abstractive dialogue summarization
Fine-tuned on the SAMSum corpus using Google's T5 (Text-to-Text Transfer Transformer).
"""

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("t5-summarizer")

# Directory resolution
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR

# Global model state
state: Dict[str, object] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "model_name": "t5-small-samsum",
    "ready": False
}


def resolve_model_path() -> Path:
    """Locate local model directory or fallback to base model identifier."""
    candidate_app = BASE_DIR / "saved_summary_model"
    candidate_root = ROOT_DIR / "saved_summary_model"

    if candidate_app.exists() and any(candidate_app.glob("*.safetensors")):
        return candidate_app
    if candidate_root.exists() and any(candidate_root.glob("*.safetensors")):
        return candidate_root
    if candidate_app.exists():
        return candidate_app
    return candidate_root


def load_artifacts():
    """Load model, tokenizer, and configure compute hardware device."""
    global state
    target_path = resolve_model_path()
    logger.info(f"Resolving model artifacts from: {target_path}")

    # Determine optimal compute device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    logger.info(f"Target execution device: {device} ({device_name})")

    # Load Tokenizer & Model
    try:
        if target_path.exists():
            tokenizer = T5Tokenizer.from_pretrained(str(target_path))
            model = T5ForConditionalGeneration.from_pretrained(str(target_path))
            logger.info("Loaded fine-tuned model and tokenizer from local checkpoint.")
        else:
            logger.warning(f"Local checkpoint not found at {target_path}. Falling back to upstream 't5-small'.")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
    except Exception as e:
        logger.error(f"Error loading local checkpoint: {e}. Falling back to 't5-small'.")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

    model.to(device)
    model.eval()

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["device"] = device
    state["device_name"] = device_name
    state["ready"] = True
    logger.info("Summarization pipeline is ready for inference.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to handle startup and teardown."""
    load_artifacts()
    yield
    logger.info("Shutting down summarization service.")


app = FastAPI(
    title="T5 Dialogue Summarizer API",
    description="Abstractive Dialogue Summarization microservice fine-tuned on SAMSum with Google T5 Transformer",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template setup
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Pydantic Schemas
class DialogueInput(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=5,
        max_length=15000,
        description="Raw conversational text or multi-turn dialogue transcript.",
        examples=[
            "Alex: Hey, did you review the Q3 system metrics?\nTaylor: Yes, the latency reduced by 35% after implementing the Redis caching layer.\nAlex: Perfect, let's include that in the engineering retro today."
        ]
    )


class SummaryOutput(BaseModel):
    summary: str
    inference_time_ms: float
    device: str
    original_char_count: int
    summary_char_count: int
    compression_ratio: float


class HealthResponse(BaseModel):
    status: str
    ready: bool
    device: str
    model_name: str
    torch_version: str


class SampleDialogue(BaseModel):
    id: str
    title: str
    category: str
    dialogue: str


# Pre-configured sample dialogues for instant recruiter demo
SAMPLE_DIALOGUES = [
    SampleDialogue(
        id="sample-tech",
        title="Engineering Standup & Deployment",
        category="Tech / DevOps",
        dialogue=(
            "Maya: Morning team! Any updates on the microservices migration?\n"
            "Carlos: I merged the auth service refactor to staging. Tests are all passing with 94% coverage.\n"
            "Priya: Great. Database migration script is also verified on staging replica without lock contention.\n"
            "Maya: Awesome. Can we schedule production deployment for 11 PM UTC to minimize traffic disruption?\n"
            "Carlos: Works for me. I will monitor Prometheus alerts during rollout.\n"
            "Priya: I'll handle rolling back read replicas if anomaly rates spike."
        )
    ),
    SampleDialogue(
        id="sample-client",
        title="Client Requirement Negotiation",
        category="Business / Sales",
        dialogue=(
            "David: Hello Rebecca, thanks for the product demo yesterday. Our executive team was impressed.\n"
            "Rebecca: Glad to hear that, David! Did you have questions regarding the enterprise tier SLA?\n"
            "David: Yes, we need guaranteed 99.99% uptime and dedicated 24/7 technical support.\n"
            "Rebecca: That is fully covered under Enterprise Gold. It includes a dedicated solutions architect.\n"
            "David: Excellent. Please send across the revised service agreement and quote by tomorrow afternoon."
        )
    ),
    SampleDialogue(
        id="sample-casual",
        title="Weekend Travel & Dinner Planning",
        category="Casual / Social",
        dialogue=(
            "Hannah: Are we still on for hiking Mount Mitchell this Saturday?\n"
            "Liam: Definitely! Weather forecast says sunny and 22 degrees.\n"
            "Sophia: Count me in. Should we meet at the trail entrance at 8 AM?\n"
            "Hannah: 8 AM is perfect. Don't forget sunscreen and plenty of water.\n"
            "Liam: I will bring homemade sandwiches and protein bars for everyone for lunch.\n"
            "Sophia: Sounds amazing, see you guys Saturday morning!"
        )
    )
]


def clean_data(text: str) -> str:
    """Preprocess dialogue string: strip html tags, normalize newlines/whitespace, and lowercase."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    """Run tokenization, beam search generation, and decoding."""
    model = state.get("model")
    tokenizer = state.get("tokenizer")
    device = state.get("device")

    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer is not initialized.")

    cleaned_dialogue = clean_data(dialogue)

    # Tokenize input sequence
    inputs = tokenizer(
        cleaned_dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate summary tokens using Beam Search
    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.0
        )

    # Decode target tokens
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


# Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve interactive web UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health inspection endpoint."""
    return HealthResponse(
        status="healthy" if state.get("ready") else "initializing",
        ready=bool(state.get("ready")),
        device=str(state.get("device_name", "unknown")),
        model_name=str(state.get("model_name")),
        torch_version=torch.__version__
    )


@app.get("/api/samples", response_model=List[SampleDialogue])
async def get_samples():
    """Retrieve sample dialogues for testing and demonstration."""
    return SAMPLE_DIALOGUES


@app.post("/summarize/", response_model=SummaryOutput)
async def summarize(dialogue_input: DialogueInput):
    """
    Summarize dialogue endpoint.
    Accepts dialogue string, preprocesses, executes beam search, and returns abstractive summary with latency metrics.
    """
    raw_text = dialogue_input.dialogue.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dialogue input cannot be empty or only whitespace."
        )

    start_time = time.perf_counter()
    try:
        summary_result = summarize_dialogue(raw_text)
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}"
        )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    orig_len = len(raw_text)
    summ_len = len(summary_result)
    reduction = round(max(0.0, (1.0 - (summ_len / max(1, orig_len)))) * 100.0, 1)

    return SummaryOutput(
        summary=summary_result,
        inference_time_ms=round(elapsed_ms, 2),
        device=str(state.get("device_name", "unknown")),
        original_char_count=orig_len,
        summary_char_count=summ_len,
        compression_ratio=reduction
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)