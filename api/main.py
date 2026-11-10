"""FastAPI application for aspect-based review intelligence.

The trained V4 model is loaded lazily on the first prediction request so the
health endpoint and API module can be imported without model artifacts.
"""
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import V4ABSAInference, detect_complaint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "absa_v4" / "model"

app = FastAPI(
    title="Customer Review Intelligence API",
    version="0.2.0",
    description=(
        "Aspect-based sentiment prediction with a transparent, "
        "rule-based complaint indicator."
    ),
)


class ReviewRequest(BaseModel):
    review: str = Field(min_length=3, max_length=10000, description="Full customer review text")
    aspect: str = Field(min_length=1, max_length=200, description="Product/service aspect to classify")


@lru_cache(maxsize=1)
def get_predictor() -> V4ABSAInference:
    """Load the model once per API process."""
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Trained V4 model directory is missing: {MODEL_DIR}. "
            "Train the model or place its saved Hugging Face files there."
        )
    return V4ABSAInference(MODEL_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_predictor.cache_info().currsize > 0}


@app.post("/predict")
def predict(request: ReviewRequest):
    try:
        predictor = get_predictor()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sentiment, confidence = predictor.predict(request.review, request.aspect)
    complaint, reason = detect_complaint(request.review, request.aspect, sentiment)
    return {
        "review_text": request.review,
        "aspect": request.aspect,
        "predicted_sentiment": sentiment,
        "sentiment_confidence": round(float(confidence), 4),
        "complaint": bool(complaint),
        "complaint_reason": reason,
        "complaint_method": "rule_based",
    }
