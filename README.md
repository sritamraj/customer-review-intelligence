# Customer Review Intelligence
### Aspect-Based Sentiment Analysis + Complaint Classification

An end-to-end NLP research project for turning e-commerce customer reviews into structured product intelligence.

**Core tasks**
1. Aspect extraction / aspect classification
2. Aspect-level sentiment classification
3. Complaint detection
4. Complaint category classification
5. Confidence-aware inference through a FastAPI service

The project is deliberately designed as an **Applied Scientist portfolio project**: it includes baselines, transformer modeling, reproducible experiments, error analysis, calibration, unit tests, and a production-style inference API.

> This repository does not claim to use private Amazon data. The training pipeline is designed around public datasets and/or user-supplied review data.

## Why this is relevant to Amazon Applied Science

Amazon's current Applied Scientist internship description emphasizes NLP/ML, large-scale data analysis, end-to-end system ownership, experimentation, scalable ML models, and research communication. This project is designed to demonstrate those exact skills through a customer-facing NLP problem. See the role description: https://www.amazon.jobs/en/jobs/10477346/applied-scientist-i-intern-jan-2027-6-months

## System

```text
Raw Reviews
    |
    v
Data validation + cleaning
    |
    +----------------------------+
    |                            |
    v                            v
Aspect/Sentiment data       Complaint data
    |                            |
    v                            v
TF-IDF baselines            TF-IDF baselines
    |                            |
    +-------------+--------------+
                  |
                  v
        Transformer models
                  |
                  v
      Evaluation + calibration
                  |
                  v
       Error analysis / slices
                  |
                  v
       FastAPI inference API
```

## Example

Input:
> "The battery lasts all day, but the charging cable stopped working after two weeks."

Structured output:

```json
{
  "aspects": [
    {"aspect": "battery", "sentiment": "positive"},
    {"aspect": "charging cable", "sentiment": "negative"}
  ],
  "complaint": true,
  "complaint_category": "quality_or_defect"
}
```

## Repository structure

```text
customer-review-intelligence/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── README.md
├── notebooks/
│   └── 01_exploration.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── api/
│   └── main.py
├── tests/
│   ├── test_data.py
│   └── test_inference.py
├── scripts/
│   ├── prepare_demo_data.py
│   └── run_experiment.py
├── .gitignore
├── Makefile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quickstart

### 1. Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate a tiny demo dataset

```bash
python scripts/prepare_demo_data.py
```

### 3. Train a reproducible baseline

```bash
python -m src.train --model tfidf --data data/processed/demo_reviews.csv
```

### 4. Evaluate

```bash
python -m src.evaluate --predictions artifacts/predictions.csv
```

### 5. Run API

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Research roadmap

### Phase 1 — Strong baseline
- TF-IDF + Logistic Regression
- class weighting
- stratified split
- macro-F1, weighted-F1, precision, recall

### Phase 2 — Transformer
Fine-tune DistilBERT/RoBERTa-style encoder for complaint classification and aspect sentiment.

### Phase 3 — Multi-task learning
Use a shared encoder with task-specific heads:

```text
                    Review
                      |
                Transformer
                      |
          +-----------+-----------+
          |                       |
   Sentiment head          Complaint head
          |                       |
 aspect sentiment          complaint category
```

A shared representation can reduce duplicated computation while allowing each task to specialize.

### Phase 4 — Research-quality evaluation
- bootstrap confidence intervals
- calibration error
- threshold tuning
- subgroup/slice analysis
- confusion matrices
- false-positive / false-negative audit
- ablations: preprocessing, class weights, encoder size, max sequence length

## Metrics

For imbalanced classification, do not rely only on accuracy.

Primary:
- Macro F1
- Per-class F1
- Precision / Recall
- Confusion matrix

Secondary:
- ROC-AUC / PR-AUC where appropriate
- Expected Calibration Error (ECE)
- inference latency
- model size

## Dataset strategy

For a public GitHub repository, do **not** commit large review dumps.

Recommended approach:
1. Keep `data/raw/` empty except for `.gitkeep`.
2. Provide download/preparation instructions.
3. Store only a tiny demo dataset in `data/processed/`.
4. Put large artifacts in external storage or Git LFS when licensing permits.
5. Record dataset versions and preprocessing hashes.

For aspect-level sentiment, public SemEval ABSA datasets are useful for benchmarking. For complaint intelligence, use a separately labeled complaint dataset or a manually audited sample from an allowed public review source.

## What makes this stronger than a generic sentiment project

A normal portfolio project says:

> "I trained BERT for sentiment analysis."

This project says:

> "I designed an end-to-end customer-feedback intelligence system, compared classical and transformer baselines, handled class imbalance, separated aspect-level sentiment from complaint detection, evaluated calibration and error slices, and exposed the model through a production-style API."

That is much closer to an Applied Scientist narrative.

## Responsible claims

Do not put invented accuracy numbers in the README or resume.

After training, replace placeholders with measured values from your own experiment:
- Macro-F1: `XX.XX`
- Complaint F1: `XX.XX`
- Aspect sentiment F1: `XX.XX`
- p95 latency: `XX ms`

## Resume-ready project

**Customer Review Intelligence — Aspect-Based Sentiment & Complaint Classification**
- Built an end-to-end NLP pipeline for extracting product aspects, predicting aspect-level sentiment, and classifying customer complaints from e-commerce review text.
- Benchmarked TF-IDF/Logistic Regression against transformer-based models using macro-F1, per-class F1, calibration, and error-slice analysis.
- Designed a multi-task learning formulation with shared text representations and task-specific heads; packaged inference behind a FastAPI service with reproducible training/evaluation workflows.

Only use the bullets after you have actually implemented and measured the corresponding components.
