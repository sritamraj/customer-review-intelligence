from fastapi.testclient import TestClient

from api import main


class FakePredictor:
    def predict(self, review, aspect):
        assert review
        assert aspect
        return "negative", 0.87654


def test_health_endpoint():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_uses_aspect_and_returns_rule_based_complaint(monkeypatch):
    monkeypatch.setattr(main, "get_predictor", lambda: FakePredictor())
    client = TestClient(main.app)
    response = client.post(
        "/predict",
        json={"review": "The battery drains quickly and is disappointing.", "aspect": "battery"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_sentiment"] == "negative"
    assert body["aspect"] == "battery"
    assert body["sentiment_confidence"] == 0.8765
    assert body["complaint"] is True
    assert body["complaint_method"] == "rule_based"


def test_predict_requires_aspect():
    client = TestClient(main.app)
    response = client.post("/predict", json={"review": "The battery is poor."})
    assert response.status_code == 422
