# Expense Categorization ML Architecture & Evolution

This document outlines the machine learning training pipeline, runtime inference architecture, and key improvements made over the legacy implementation for automated personal expense categorization.

---

## 1. What Changed (Legacy vs. Enhanced)

| Dimension | Legacy Implementation | Enhanced Implementation | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Input Features** | `Head` only (accidentally fit on DataFrame column headers in training) | **3 Parameters:** `Head` + `comment` + `amount` | Generic titles like `"Shopping"` or `"Groceries"` now leverage bank/UPI merchant strings (`MYNTRA`, `COMPASS INDIA`, `PEDIASURE`) and transaction size. |
| **Text Preprocessing** | Raw `CountVectorizer` without cleaning | URL decoding (`unquote`) + regex stripping of UPI/SBI SMS boilerplate | Removes repetitive bank boilerplate (`has been debited from account...`, `UPI transaction reference no`) so TF-IDF focuses on actual merchant names. |
| **Tokenization & Typo Handling** | Word tokens only | **Hybrid `FeatureUnion`:** Word n-grams `(1, 2)` + Character n-grams `char_wb (3, 5)` | Automatically handles spelling variations (`Vegitables`, `Snacs`, `Stiching`, `Physiotherepist`) without manual dictionaries. |
| **Numerical Scaling** | Ignored `amount` | `log1p(amount)` with `StandardScaler` (`0.35` weight) + `<= ₹1` neutral imputation (`₹300`) | Uses transaction magnitude as a tie-breaker without allowing zero-amount entries (`₹0.00`) to skew predictions. |
| **Classifier & Calibration** | `LinearSVC` (hard labels only, no confidence scores) | `LogisticRegression` (`C=3.0`, `class_weight='balanced'`) | Produces calibrated multi-class probabilities (`confidence`) and prevents dominant classes (`Grocery`) from drowning out rare categories. |
| **Artifact Packaging** | Two separate files (vectorizer + model) with TF-IDF skipped at inference | **Single unified `Pipeline`** (`model/expense-pipeline.joblib`) | Eliminates training/serving skew by bundling feature extraction, vectorization, scaling, and classification into one versioned artifact. |
| **Inference Guardrails** | Pure ML prediction | **Hybrid Rule Guard + ML Fallback** + Confidence Threshold (`0.55`) | Guarantees `1.0` confidence on exact category names and high-precision household rules (e.g., routing diagnostic/blood tests to `Medical`). |
| **gRPC Throughput** | Single unary RPC per Google Sheet row | **Bidirectional Streaming Batch RPC** (`getExpenseCategory`) | Categorizes thousands of sheet rows in a single HTTP/2 stream instead of thousands of individual network round-trips. |

---

## 2. Training Approach (`train_model.py` & `model_utils.py`)

### Data Preparation & Normalization
1. **Noise Filtering:** Strips synthetic/placeholder records (`Dummy`, `test`, `T est`, `Tes`) and rows with `amount <= 1.0`.
2. **Active Learning Support:** Automatically merges optional user-verified corrections from `corrections.csv` when present.
3. **Taxonomy Alignment:**
   - Consolidates fragmented micro-categories (`Shopping` → `Accessories`).
   - Normalizes diagnostic and pathology lab entries (`blood test`, `x-ray`, `MRI`, `scan`, `Thyrocare`, `Lal PathLabs`, etc.) to `Medical`.
   - Drops categories with fewer than 5 samples to preserve stratified cross-validation integrity.

### Feature Engineering (`model_utils.py`)
- **Text Cleaning (`clean_text`):** Decodes URL-encoded characters, strips repetitive UPI and credit card SMS templates, removes non-alphabetic noise, and normalizes whitespace/casing.
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
- Evaluates precision, recall, and F1-score on an **80/20 stratified holdout split**.
- Retrains on **100% of the cleaned dataset** before exporting `model/expense-pipeline.joblib`.

---

## 3. Inference Architecture (`home-category-api` & `home-etl-service`)

### End-to-End Request Flow

```text
┌──────────────────────────┐        gRPC (HTTP/2 Stream)        ┌─────────────────────────────────────┐
│     home-etl-service     │ ─────────────────────────────────► │          home-category-api          │
│   (Java 25 / Spring)     │   ExpenseCategorizationRequest     │          (Python gRPC Server)       │
│                          │   { head, comment, amount }        │                                     │
│ 1. Fetch Google Sheet    │                                    │ 1. Exact Category Map Lookup        │
│ 2. Filter Dummy/Test     │ ◄───────────────────────────────── │ 2. Regex Domain Rule Guard          │
│ 3. Stream Full Comments  │   ExpenseCategorizationResponse    │ 3. ML Pipeline predict_proba()      │
│ 4. Batch Save to DB      │   { category, confidence,          │ 4. Flag needs_review (< 0.55)       │
└──────────────────────────┘     needs_review }                 └─────────────────────────────────────┘
```

### Multi-Stage Prediction Logic (`expenseCategorizerServer.py`)
For each incoming transaction (`head`, `comment`, `amount`), the gRPC servicer runs a three-tier evaluation:
1. **Exact Category Match:** If `head` directly matches a known category name (or common typo alias like `stationary` → `Education`), returns immediately with `confidence = 1.0`.
2. **Deterministic Domain Rules:** Evaluates high-precision regex patterns across `head + comment` (such as diagnostic/blood tests → `Medical`, `society helper` → `House Help`, `magic bricks` → `Maintenance`) returning `confidence = 1.0`.
3. **ML Model Inference:** Passes a 1-row DataFrame into `self.pipeline.predict_proba()`, selects the highest-probability class, and sets `needs_review = True` if `confidence < 0.55`.

---

## 4. Operational Workflow

### Retraining the Model
```bash
# 1. Update ~/Downloads/Expense Sheet - Categorized.csv (or append to corrections.csv)
# 2. Run training and overwrite model/expense-pipeline.joblib
python train_model.py
```

### Regenerating gRPC Stubs (Only Needed on `.proto` Changes)
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

### Building & Running the Container
```bash
docker build -t alokkusingh/home-category-api .
docker run -d -p 50051:50051 --name home-category-api alokkusingh/home-category-api
```