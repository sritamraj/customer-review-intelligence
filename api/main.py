from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.inference import ReviewIntelligence

app = FastAPI(
    title="Customer Review Intelligence API",
    version="0.1.0",
    description="Complaint classification API for e-commerce review intelligence."
)

class ReviewRequest(BaseModel):
    review: str = Field(min_length=3, max_length=10000)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: ReviewRequest):
    engine = ReviewIntelligence()
    return engine.predict(request.review)
