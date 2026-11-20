import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from model_utils import build_features

# 1. Load & Clean Data
csv_path = os.path.expanduser("~/Downloads/Expense Sheet - Categorized.csv")
df = pd.read_csv(csv_path)
df.rename(columns={'head': 'Head', 'category': 'Category'}, inplace=True)

# Optional: Append manually verified corrections if present (Active Learning loop)
corrections_path = "corrections.csv"
if os.path.exists(corrections_path):
    corr_df = pd.read_csv(corrections_path)
    corr_df.rename(columns={'head': 'Head', 'category': 'Category'}, inplace=True)
    df = pd.concat([df, corr_df], ignore_index=True)
    print(f"Loaded {len(corr_df)} manual correction rows from {corrections_path}")

# Filter out dummy/test rows (including "T est" and "Tes") and zero/₹1 placeholder rows
df = df[~df['Head'].astype(str).str.strip().str.lower().str.contains(r'dummy|^t\s*est|^tes$', regex=True, na=False)]
df = df[pd.to_numeric(df['amount'], errors='coerce').fillna(0.0) > 1.0].copy()
df['Category'] = df['Category'].replace({'Shopping': 'Accessories'})

# Filter out categories with fewer than 5 samples so stratified splitting works cleanly
counts = df['Category'].value_counts()
df = df[df['Category'].isin(counts[counts >= 5].index)]

X = df[['Head', 'comment', 'amount']]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Build 3-Parameter Pipeline
# - min_df=1 keeps rare but distinct keywords (e.g., "Physiotherepist", "Banana")
# - transformer_weights keeps amount as a secondary tie-breaker so clear keywords win
text_features = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2), stop_words='english')),
    ('char_tfidf', TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, min_df=2, ngram_range=(3, 5)))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_features, 'text'),
        ('amount', StandardScaler(), ['log_amount'])
    ],
    transformer_weights={
        'text': 1.0,
        'amount': 0.35
    }
)

pipeline = Pipeline([
    ('feature_builder', FunctionTransformer(build_features, validate=False)),
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(
        C=3.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

# 3. Evaluate on 20% Holdout
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

# 4. Retrain on 100% Data & Save
pipeline.fit(X, y)
os.makedirs("model", exist_ok=True)
output_path = "model/expense-pipeline.joblib"
joblib.dump(pipeline, output_path)
print(f"Saved improved 3-parameter model to {output_path}")

# 5. Sanity Check on Previously Low-Confidence Examples
samples = pd.DataFrame([
    {"Head": "Appliances", "comment": "", "amount": 2528.10},
    {"Head": "Milk", "comment": "", "amount": 211.0},
    {"Head": "Flight ticket to n from Patna", "comment": "", "amount": 0.0},
    {"Head": "Cook", "comment": "", "amount": 0.0},
    {"Head": "Trolly bag", "comment": "", "amount": 0.0},
    {"Head": "Baby shampoo", "comment": "", "amount": 1755.0},
])
preds = pipeline.predict(samples)
probs = pipeline.predict_proba(samples).max(axis=1)
for row, pred, prob in zip(samples.to_dict(orient="records"), preds, probs):
    print(f"{row['Head']} (₹{row['amount']}) -> {pred} (confidence: {prob:.2f})")
