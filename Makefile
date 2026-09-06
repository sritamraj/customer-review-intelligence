install:
	pip install -r requirements.txt

demo:
	python scripts/prepare_demo_data.py

train:
	python -m src.train --model tfidf --data data/processed/demo_reviews.csv

test:
	pytest -q

api:
	uvicorn api.main:app --reload
