# Expense Categorization ML Architecture & Evolution

This document outlines the machine learning training pipeline, active learning workflow, runtime inference architecture, and key improvements made over the legacy implementation for automated personal expense categorization.

---

## 1. What Changed (Legacy vs. Enhanced)

| Dimension | Legacy Implementation | Enhanced Implementation | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Input Features** | `Head` only (accidentally fit on DataFrame column headers in training) | **3 Parameters:** `Head` + `comment` + `amount` | Generic titles like `"Shopping"` or `"Groceries"` leverage bank/UPI merchant strings (`MYNTRA`, `COMPASS INDIA`, `PEDIASURE`) and transaction size. |
| **Text Preprocessing** | Raw `CountVectorizer` without cleaning | URL decoding (`unquote`) + regex stripping of UPI/SBI SMS boilerplate + Unicode preservation | Removes repetitive bank boilerplate (`has been debited from account...`, `UPI transaction reference no`) while preserving English and Hindi/Devanagari (`ऑटो`) text. |
| **Tokenization & Typo Handling** | Word tokens only | **Hybrid `FeatureUnion`:** Word n-grams `(1, 2)` + Character n-grams `char_wb (3, 5)` | Automatically handles spelling variations (`Vegitables`, `Snacs`, `Stiching`, `Physiotherepist`) without manual dictionaries. |
| **Numerical Scaling** | Ignored `amount` | `log1p(amount)` with `StandardScaler` (`0.35` weight) + `<= ₹1` neutral imputation (`₹300`) | Uses transaction magnitude as a tie-breaker without allowing zero-amount entries (`₹0.00`) to skew predictions. |
| **Classifier & Calibration** | `LinearSVC` (hard labels only, no confidence scores) | `LogisticRegression` (`C=3.0`, `class_weight='balanced'`) | Produces calibrated multi-class probabilities (`confidence`) and prevents dominant classes (`Grocery`) from drowning out rare categories. |
| **Artifact Packaging** | Two separate files (vectorizer + model) with TF-IDF skipped at inference | **Single unified `Pipeline`** (`model/expense-pipeline.joblib`, `compress=3`) | Eliminates training/serving skew by bundling feature extraction, vectorization, scaling, and classification into one compressed artifact. |
| **Inference Guardrails** | Pure ML prediction | **Hybrid Rule Guard + ML Fallback** + Confidence Threshold (`0.55`) | Evaluates domain regex rules across `head + comment`, short-circuits exact category names (when no comment overrides them), and flags uncertain predictions (`needs_review`). |
| **Active Learning Loop** | Manual CSV edits only | Auto-logging to `data/corrections_pending.csv` + interactive `approve_corrections.py` CLI (`2.0x` sample weight) | Captures low-confidence production transactions automatically and reinforces human-verified corrections during retraining. |
| **gRPC Throughput** | Single unary RPC per Google Sheet row | **Bidirectional Streaming Batch RPC** (`getExpenseCategory`) | Categorizes thousands of sheet rows in a single HTTP/2 stream instead of thousands of individual network round-trips. |

---

## 2. Training & Active Learning Pipeline (`training/train_model.py` & `model_utils.py`)

### Data Preparation & Normalization
1. **Noise Filtering:** Strips synthetic/placeholder records (`Dummy`, `test`, `T est`, `Tes`) and main sheet rows with `amount <= 1.0` (while preserving valid zero-amount items in `data/corrections.csv` such as `Crape bandage`).
2. **Active Learning Integration:** Merges verified rows from `data/corrections.csv` with `sample_weight = 2.0` (vs. `1.0` for historical rows) to reinforce edge cases.
3. **Taxonomy Alignment:**
   - Consolidates fragmented micro-categories (`Shopping` → `Accessories`).
   - Normalizes diagnostic and pathology lab entries (`blood test`, `x-ray`, `MRI`, `scan`, `Thyrocare`, `Lal PathLabs`, etc.) to `Medical`.
   - Drops categories with fewer than 5 samples to preserve stratified split integrity.

### Feature Engineering (`model_utils.py`)
- **Text Cleaning (`clean_text`):** Decodes URL-encoded characters (`%2C`, `%3A`, `%40`), strips repetitive UPI and credit card SMS templates, removes digits and punctuation while preserving Unicode word characters (`[\d_]|[^\w\s]`), and normalizes whitespace/casing.
- **Amount Imputation (`build_features`):** Replaces missing or placeholder amounts (`<= 1.0`) with a neutral `300.0` baseline before applying `np.log1p()`, preventing extreme negative z-scores during inference.

### Unified Scikit-Learn Pipeline
All preprocessing and modeling steps are serialized together into `model/expense-pipeline.joblib`:

```text
Raw DataFrame ['Head', 'comment', 'amount']
        │
        ▼
FunctionTransformer (build_features)
  ├── 'text'       ──► FeatureUnion
  │                      ├── TfidfVectorizer (word n-grams 1..2, min_df=1, sublinear_tf=True)
  │                      └── TfidfVectorizer (char_wb n-grams 3..5, min_df=2, sublinear_tf=True)
  └── 'log_amount' ──► StandardScaler (weighted at 0.35 vs text at 1.0)
        │
        ▼
LogisticRegression (C=3.0, class_weight='balanced', max_iter=1000)
        │
        ▼
Output: Predicted Category + Class Probability Distribution
```

### Validation & Export
- Evaluates precision, recall, and F1-score on an **80/20 stratified holdout split** (~96% overall accuracy).
- Retrains on **100% of the cleaned dataset** and exports `model/expense-pipeline.joblib` with `compress=3`.

---

## 3. Inference & Continuous Learning Architecture

### End-to-End Request Flow

```text
┌──────────────────────────┐        gRPC (HTTP/2 Stream)        ┌─────────────────────────────────────┐
│     home-etl-service     │ ─────────────────────────────────► │          home-category-api          │
│   (Java 25 / Spring)     │   ExpenseCategorizationRequest     │          (Python gRPC Server)       │
│                          │   { head, comment, amount }        │                                     │
│ 1. Fetch Google Sheet    │                                    │ 1. Regex Domain Rule Guard          │
│ 2. Filter Dummy/Test     │ ◄───────────────────────────────── │ 2. Exact Category Map (empty cmt)   │
│ 3. Stream Full Comments  │   ExpenseCategorizationResponse    │ 3. ML Pipeline predict_proba()      │
│ 4. Batch Save to DB      │   { category, confidence,          │ 4. Log < 0.55 to pending CSV        │
└──────────────────────────┘     needs_review }                 └─────────────────────────────────────┘
```

### Multi-Stage Prediction Logic (`interface/grpc/servicer/expenseCategorizerServer.py`)
For each incoming transaction (`head`, `comment`, `amount`), the gRPC servicer runs a four-step flow:
1. **Deterministic Domain Rules:** Evaluates high-precision regex patterns across `head + comment` (e.g., diagnostic/blood tests → `Medical`, `society helper` → `House Help`, `magic bricks` → `Bills`, `mill` → `Grocery`) returning `confidence = 1.0`.
2. **Exact Category Match (Empty Comment Only):** If `comment` is blank and `head` matches a known category name (or typo alias like `stationary` → `Education`), returns immediately with `confidence = 1.0`. When `comment` is present, this step is skipped so merchant details can override generic headers (e.g., `head="Lunch"` + `COMPASS INDIA` → `Food Outside`).
3. **ML Model Inference:** Passes a 1-row DataFrame into `self.pipeline.predict_proba()` and selects the highest-probability class.
4. **Low-Confidence Capture:** When `confidence < 0.55`, sets `needs_review = True` and appends a deduplicated row to `data/corrections_pending.csv` in a thread-safe manner.

---

## 4. Operational Workflow

### 1. Review Low-Confidence Logs & Retrain
```bash
# Interactively review data/corrections_pending.csv and promote to data/corrections.csv
python approve_corrections.py

# Retrain model with 2.0x weight on verified corrections
python training/train_model.py

# Run 52-case Unary + Bidirectional Streaming regression suite
python test/test_grpc_model.py
```

### 2. Regenerating gRPC Stubs (Only Needed on `.proto` Changes)
```bash
# Python stubs (home-category-api)
python -m grpc_tools.protoc -Iproto \
  --python_out=interface/grpc/servicer/generated \
  --pyi_out=interface/grpc/servicer/generated \
  --grpc_python_out=interface/grpc/servicer/generated \
  proto/expenseCategorizer.proto

# Java stubs (home-etl-service)
protoc --plugin=protoc-gen-grpc-java="\$(pwd)/protoc-gen-grpc-java-arm64" \
  --java_out=src/main/java \
  --grpc-java_out=src/main/java \
  -Isrc/main/resources/proto \
  src/main/resources/proto/expenseCategorizer.proto
```

### 3. Building & Running the Container (With Persistent Corrections Volume)
```bash
docker build -t alokkusingh/home-category-api .
docker run -d -p 50051:50051 \
  -v "\$(pwd)/data:/app/data" \
  --name home-category-api \
  alokkusingh/home-category-api
```