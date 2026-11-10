# Customer Review Intelligence

### Aspect-Based Sentiment Analysis & Complaint Intelligence

An end-to-end NLP research project for converting customer reviews into **aspect-level sentiment signals and complaint intelligence**.

The project investigates how classical NLP baselines compare with transformer-based models for aspect-based sentiment analysis (ABSA), with a particular focus on **class imbalance, rare-class behavior, and error analysis**.

A lightweight rule-based complaint detector is then combined with the final ABSA model to produce structured customer-feedback signals.

---

## 🎯 Project Objective

Customer reviews often contain multiple product aspects with different sentiments:

> "The battery lasts all day, but the charging cable stopped working."

A document-level sentiment classifier may label the entire review as simply positive or negative.

ABSA instead attempts to model sentiment at the **aspect level**:

```text
battery          → positive
charging cable   → negative
```

This project explores that problem using a progression of increasingly capable models.

---

# 🔬 Experimental Pipeline

```text
                 Customer Reviews
                        |
                        v
             Data preparation / cleaning
                        |
                        v
              Aspect + sentiment pairs
                        |
          +-------------+-------------+
          |                           |
          v                           v
   V1: Word TF-IDF             V2: Word + Char TF-IDF
   + Logistic Regression       + Logistic Regression
          |                           |
          +-------------+-------------+
                        |
                        v
              V3: Aspect-aware
                DistilBERT
                        |
                        v
             V4: DistilBERT +
                 Focal Loss
                        |
                        v
              Error Analysis
                        |
                        v
        Rule-based Complaint Detection
                        |
                        v
             Final Inference Pipeline
                        |
                        v
             Structured Predictions
```

The main scientific progression is:

**classical baseline → feature improvement → transformer → imbalance-aware transformer → error analysis → complaint intelligence**

---

# 📊 Experimental Results

The SemEval laptop ABSA dataset was split into:

| Split      | Examples |
| ---------- | -------: |
| Train      |    1,670 |
| Validation |      345 |
| Test       |      332 |

The training distribution is highly imbalanced, particularly for the **conflict** class.

### Model comparison

| Model                                              | Test Macro-F1 | Test Accuracy | Test Weighted-F1 |
| -------------------------------------------------- | ------------: | ------------: | ---------------: |
| V1 — Word TF-IDF + Logistic Regression             |    **0.4109** |        0.5994 |           0.5936 |
| V2 — Word + Character TF-IDF + Logistic Regression |    **0.4193** |        0.6114 |           0.6043 |
| V3 — Aspect-aware DistilBERT + Weighted CE         |    **0.5151** |        0.7048 |           0.7098 |
| V4 — Aspect-aware DistilBERT + Focal Loss          |    **0.5437** |        0.6807 |           0.6930 |

### Main result

V4 improved test Macro-F1 from:

```text
V1: 0.4109
        ↓
V4: 0.5437
```

Absolute improvement:

```text
+0.1328 Macro-F1
```

This corresponds to approximately a **32.3% relative improvement** over the V1 Macro-F1.

The transformer-based models substantially improved performance over the TF-IDF baselines, particularly for the neutral class.

---

# 🧠 V1 — Classical Baseline

### Word TF-IDF + Logistic Regression

The first experiment establishes a simple and reproducible baseline.

Features:

* word-level TF-IDF
* Logistic Regression
* stratified training data
* four sentiment classes:

  * positive
  * negative
  * neutral
  * conflict

Results:

```text
Accuracy     = 0.5994
Macro-F1     = 0.4109
Weighted-F1  = 0.5936
```

The baseline struggled with:

* neutral sentiment
* mixed/conflicting sentiment
* rare classes

This provides a reference point for subsequent experiments.

---

# 🧪 V2 — Word + Character Features

The second experiment combines:

```text
Word TF-IDF
     +
Character TF-IDF
     ↓
Logistic Regression
```

The motivation was to determine whether character-level information could improve robustness to lexical variation.

Results:

```text
Accuracy     = 0.6114
Macro-F1     = 0.4193
Weighted-F1  = 0.6043
```

Macro-F1 improvement over V1:

```text
+0.0084
```

The improvement was modest, motivating the transition to contextual transformer representations.

---

# 🤖 V3 — Aspect-Aware DistilBERT

V3 introduces contextual modeling using:

**DistilBERT (`distilbert-base-uncased`)**

Instead of classifying the review independently of the aspect, the model receives:

```text
review_text + "[SEP]" + aspect
```

For example:

```text
Review:
"The battery lasts all day but the charger is terrible."

Aspect:
"battery"
```

The model therefore receives both the review context and the target aspect.

### Training configuration

```text
Model:              DistilBERT
Input:              review + [SEP] + aspect
Max length:         256
Learning rate:      2e-5
Train batch size:   8
Evaluation batch:   16
Epochs:             4
Weight decay:       0.01
Loss:               class-weighted Cross Entropy
Selection metric:   validation Macro-F1
```

Best validation Macro-F1:

```text
0.5713
```

Test performance:

```text
Accuracy     = 0.7048
Macro-F1     = 0.5151
Weighted-F1  = 0.7098
```

### Per-class test F1

| Sentiment |     F1 |
| --------- | -----: |
| Positive  | 0.7888 |
| Negative  | 0.7491 |
| Neutral   | 0.5224 |
| Conflict  | 0.0000 |

V3 demonstrated a major improvement over the classical baselines, but the conflict class remained extremely difficult.

---

# 🔥 V4 — Focal Loss

V4 keeps the aspect-aware DistilBERT architecture but changes the training objective to **class-weighted Focal Loss**.

The motivation was to investigate whether the model could improve its treatment of difficult and underrepresented examples.

```text
V3:
DistilBERT
    +
Class-weighted Cross Entropy

V4:
DistilBERT
    +
Class-weighted Focal Loss
```

Configuration:

```text
Model:              DistilBERT
Input:              review + [SEP] + aspect
Max length:         256
Learning rate:      2e-5
Train batch size:   8
Evaluation batch:   16
Epochs:             4
Weight decay:       0.01
Focal gamma:        2.0
Selection metric:   validation Macro-F1
```

### V4 test results

```text
Accuracy     = 0.6807
Macro-F1     = 0.5437
Weighted-F1  = 0.6930
```

Per-class F1:

| Sentiment |   F1 |
| --------- | ---: |
| Positive  | 0.76 |
| Negative  | 0.74 |
| Neutral   | 0.50 |
| Conflict  | 0.18 |

V4 achieved the **highest test Macro-F1** among the evaluated models.

---

# ⚖️ Why Macro-F1?

The dataset contains substantial class imbalance.

In particular, the conflict class contains very few examples.

Therefore, accuracy alone can hide poor performance on minority classes.

The primary model-selection metric is consequently:

```text
Macro-F1
```

Macro-F1 gives each class equal importance:

```text
Macro-F1 =
(F1_positive + F1_negative + F1_neutral + F1_conflict) / 4
```

Additional metrics include:

* per-class precision
* per-class recall
* per-class F1
* weighted-F1
* accuracy
* confusion matrices

---

# 🔎 Error Analysis

Error analysis was performed after the transformer experiments rather than treating the final metric as the complete evaluation.

For V4, the 332-example test set contained:

```text
Correct predictions: 226
Errors:              106
Accuracy:            0.6807
```

The largest confusion pattern was:

```text
negative → neutral : 27
positive → neutral : 26
positive → negative : 16
neutral  → negative : 12
negative → positive : 11
```

A major observation was that the model frequently predicted **neutral** for difficult examples.

---

# 🚨 Conflict-Class Analysis

The conflict class is particularly challenging.

V4 produced:

```text
Conflict precision = 0.33
Conflict recall    = 0.12
Conflict F1       = 0.18
```

There were only **8 conflict examples** in the test set.

Therefore, this metric should not be interpreted as a stable estimate of real-world conflict performance.

More importantly, error analysis showed that most missed conflict examples were predicted as:

```text
conflict → neutral
```

This suggests that the model struggles with **compositional or mixed sentiment**, rather than simply making random errors.

Example:

> "The 4 USB ports are nice, but the two on the side only work now and then."

The sentence contains both positive and negative signals around the same aspect context.

This is an important limitation of the current model.

---

# 📌 Aspect-Level Error Analysis

Performance varies substantially by aspect.

Examples from the V4 test analysis:

| Aspect       | Correct / Total |
| ------------ | --------------: |
| battery      |           6 / 6 |
| battery life |           3 / 5 |
| screen       |         10 / 13 |
| programs     |           7 / 8 |
| keyboard     |           5 / 6 |
| warranty     |           2 / 8 |
| RAM          |           0 / 2 |
| hinge        |           1 / 2 |

These slices highlight that aggregate metrics can hide large differences between individual aspects.

---

# 🧾 Complaint Intelligence

The SemEval ABSA dataset used for the sentiment experiments does **not provide a supervised complaint label**.

Therefore, this project does **not fabricate complaint ground truth** and does not claim to have trained a supervised complaint classifier on SemEval.

Instead, complaint detection is implemented as a **transparent rule-based baseline**.

The detector combines:

* review text
* aspect context
* predicted sentiment
* complaint-related lexical patterns

Examples of complaint signals include patterns associated with:

```text
not working
doesn't work
failed
problem
issue
defective
broken
stopped working
too slow
overheating
refund
warranty
expensive
poor
disappointed
unable
```

The detector also assigns a coarse complaint reason such as:

```text
battery_power
display
input_device
shipping
service_warranty
performance_reliability
price_cost
other
```

This is intentionally transparent and interpretable rather than presented as a supervised benchmarked complaint classifier.

---

# 🔗 Final Inference Pipeline

The final inference system combines the V4 transformer with the complaint detector.

```text
Review
  +
Aspect
  |
  v
V4 DistilBERT
  |
  +----------------------+
  |                      |
  v                      v
Sentiment             Confidence
  |
  v
Aspect-aware
complaint rules
  |
  v
Complaint + reason
```

The final inference output is stored as:

```text
artifacts/absa_v4/final_inference.csv
```

The output contains fields including:

```text
review_text
aspect
predicted_sentiment
sentiment_confidence
complaint
complaint_reason
```

The final test-set inference pipeline generated predictions for:

```text
332 examples
```

---

# 📁 Repository Structure

```text
customer-review-intelligence/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── demo_reviews.csv
│   └── README.md
│
├── notebooks/
│   └── 01_exploration.md
│
├── scripts/
│   ├── prepare_demo_data.py
│   └── run_experiment.py
│
├── src/
│   ├── __init__.py
│   ├── complaint_detector.py
│   ├── complaint_data.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare_semeval.py
│   │   ├── split_semeval.py
│   │   └── train_tfidf_v2.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── models.py
│   ├── train.py
│   ├── transformer_absa.py
│   ├── transformer_absa_v4.py
│   └── utils.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_absa_splits.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_inference.py
│
├── artifacts/
│   └── .gitkeep
│
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Large model checkpoints, raw datasets, and generated experiment artifacts are intentionally excluded from the public repository.

---

# 🗃️ Dataset

The main ABSA experiments use the **SemEval laptop review ABSA data**.

The repository does not redistribute the processed benchmark files or raw benchmark XML files.

The expected local workflow is:

```text
raw SemEval data
       |
       v
prepare_semeval.py
       |
       v
split_semeval.py
       |
       v
train / validation / test
```

This approach keeps the public repository lightweight and avoids unnecessarily redistributing benchmark data.

A small demo dataset is included for basic repository testing and experimentation.

To prepare and split the SemEval data locally, place the official `Laptop_Train_v2.xml` file at `data/raw/semeval_source/semeval_data/Laptop_Train_v2.xml`, then run:

```bash
python -m src.data.prepare_semeval
python -m src.data.split_semeval
```

The split script groups aspect annotations by source sentence so that aspects from one sentence cannot be distributed across training, validation, and test sets. It targets an approximate 70/15/15 split; exact row counts can differ because sentence groups are kept intact. The previous word/character TF-IDF experiment script is now named `src/data/train_tfidf_v2.py` to distinguish model training from dataset splitting.

---

# 🚀 Reproducibility

Create a Python environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q
```

The test suite covers the demo complaint-classification pipeline, grouped ABSA split isolation, and API request/response behavior. The exact test count may change as coverage grows.

---

# 🧪 Running the Transformer Experiment

After preparing the local SemEval dataset, the transformer experiments can be run using the corresponding scripts.

V3:

```bash
python src/transformer_absa.py
```

V4:

```bash
python src/transformer_absa_v4.py
```

V4 produces:

```text
artifacts/absa_v4/
├── model/
├── test_predictions.csv
└── metrics.txt
```

---

# 🔬 Error Analysis

Run:

```bash
python src/error_analysis.py --predictions artifacts/absa_v4/test_predictions.csv
```

The analysis reports:

* overall accuracy
* error counts
* sentiment distributions
* confusion pairs
* aspect-level performance
* missed conflict examples
* representative prediction errors

Generated analysis files include:

```text
error_analysis.csv
confusion_pairs.csv
```

---

# 🧠 Final Inference

Run:

```bash
python src/inference.py
```

The final inference pipeline loads the trained V4 model and generates:

```text
artifacts/absa_v4/final_inference.csv
```

---

# 🧪 Tests

The repository contains lightweight tests covering core data/inference functionality.

Run:

```bash
pytest -q
```

Current result:

```text
2 passed
```

---

# ⚠️ Limitations

This project has several important limitations.

### 1. Small benchmark

The test set contains only 332 examples.

Results should therefore not be interpreted as production-level estimates.

### 2. Rare conflict class

Only 8 conflict examples appear in the test set.

Consequently, the conflict F1 of 0.18 has high statistical uncertainty.

### 3. Complaint labels

The SemEval benchmark does not provide complaint annotations.

The complaint component is therefore a **rule-based baseline**, not a supervised complaint-classification benchmark.

### 4. Rule-based complaint detection

Lexical rules can produce false positives when complaint keywords occur in non-complaint contexts.

### 5. Domain limitation

The main benchmark focuses on laptop reviews.

Performance may differ substantially on other product categories.

### 6. CPU training

The transformer experiments were performed locally on CPU and are therefore intended primarily as a reproducible research/portfolio workflow rather than a latency-optimized production deployment.

### 7. No claim of production readiness

The project demonstrates modeling and experimentation skills; it should not be interpreted as a production Amazon system or as using private Amazon data.

---

# 🔭 Future Research Directions

Potential extensions include:

### Better conflict modeling

Investigate:

* larger pretrained encoders
* contrastive learning
* explicit compositional sentiment modeling
* targeted augmentation
* additional conflict-labeled datasets

### Better complaint classification

Collect or use a separately labeled complaint dataset and evaluate:

```text
complaint detection
        +
complaint category
        +
severity
```

with proper supervised metrics.

### Confidence calibration

Future experiments could evaluate:

* reliability diagrams
* Expected Calibration Error
* temperature scaling
* threshold selection

These are **future research directions**, not claimed completed components of the current implementation.

### Statistical evaluation

Future work could add:

* bootstrap confidence intervals
* significance testing
* additional random seeds
* robustness experiments

---

# 🎓 Why This Is an Applied Scientist Project

The goal is not simply:

> "Train BERT for sentiment analysis."

The project instead follows an experimental modeling process:

```text
Define problem
      ↓
Build classical baseline
      ↓
Improve representation
      ↓
Introduce contextual model
      ↓
Address class imbalance
      ↓
Compare experiments
      ↓
Analyze failures
      ↓
Identify remaining limitations
      ↓
Build a lightweight downstream intelligence layer
```

The most important result is therefore not just the final F1 score.

It is the **evidence-driven progression from baseline to transformer and the analysis of why the model still fails**.

---

# 📈 Key Takeaways

### Finding 1 — Transformer representations helped

V3 increased Macro-F1 from:

```text
0.4193 → 0.5151
```

compared with the V2 classical baseline.

### Finding 2 — Focal Loss improved Macro-F1 further

V4 increased Macro-F1 to:

```text
0.5437
```

the highest score among the evaluated models.

### Finding 3 — Rare classes remain difficult

Conflict performance improved from:

```text
V3 Conflict F1 = 0.00
V4 Conflict F1 = 0.18
```

but remains weak because the class is extremely underrepresented.

### Finding 4 — Error analysis exposed systematic behavior

The dominant conflict failure mode was:

```text
conflict → neutral
```

suggesting difficulty with mixed/compositional sentiment.

### Finding 5 — Aggregate metrics are not enough

Aspect-level analysis showed substantial variation between aspects such as:

```text
battery
screen
programs
warranty
RAM
hinge
```

which motivates slice-based evaluation.

---

# 📌 Project Status

| Component                        | Status            |
| -------------------------------- | ----------------- |
| Dataset preparation              | ✅ Complete        |
| Train/validation/test split      | ✅ Complete        |
| TF-IDF baseline                  | ✅ Complete        |
| Word + character baseline        | ✅ Complete        |
| DistilBERT ABSA                  | ✅ Complete        |
| Focal Loss experiment            | ✅ Complete        |
| Error analysis                   | ✅ Complete        |
| Rule-based complaint detector    | ✅ Complete        |
| Final inference pipeline         | ✅ Complete        |
| Unit tests                       | ✅ Complete        |
| Public GitHub packaging          | ✅ Complete        |
| Supervised complaint benchmark   | ❌ Not implemented |
| Multi-task learning              | ❌ Not implemented |
| Calibration study                | 🔬 Future work    |
| Statistical confidence intervals | 🔬 Future work    |

---

# 📄 Resume Project

**Customer Review Intelligence — Aspect-Based Sentiment & Complaint Intelligence**

* Built an aspect-aware NLP pipeline for customer-review sentiment analysis, progressing from TF-IDF/Logistic Regression baselines to DistilBERT with class-weighted training and focal loss.
* Improved test Macro-F1 from **0.4109 to 0.5437** across four experiments and performed per-class, confusion-pair, and aspect-level error analysis to diagnose rare-class and mixed-sentiment failures.
* Developed a transparent rule-based complaint-intelligence layer using review/aspect context and transformer sentiment predictions, producing structured complaint reasons without fabricating unavailable supervised labels.

---

# ⚖️ Research Integrity

All reported metrics in this README correspond to experiments actually run in this repository.

The project does **not** claim:

* private Amazon data
* production Amazon deployment
* supervised complaint labels where none exist
* multi-task learning
* calibration experiments that were not run
* statistical confidence intervals that were not computed
* production-scale latency benchmarks

The objective is to demonstrate **honest experimentation, model comparison, error analysis, and scientific reasoning**.


---

# 🌐 API Usage

The API expects both the full review and the target aspect. The V4 model must first be trained or otherwise saved under `artifacts/absa_v4/model`.

Start the server from the repository root:

```bash
uvicorn api.main:app --reload
```

Health check: `GET /health`

Prediction: `POST /predict` with JSON body:

```json
{
  "review": "The battery drains quickly, but the screen is excellent.",
  "aspect": "battery"
}
```

The response includes aspect-level sentiment and confidence, plus a transparent rule-based complaint flag. The complaint flag is a heuristic output, not a supervised complaint-model prediction.
