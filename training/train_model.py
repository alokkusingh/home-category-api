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

CSV_PATH = os.path.expanduser("~/Downloads/Expense Sheet - Categorized.csv")
CORRECTIONS_PATH = "data/corrections.csv"
MODEL_OUTPUT_PATH = "model/expense-pipeline.joblib"

DIAGNOSTIC_PATTERN = (
    r'\b(blood\s*test|lab\s*test|diagnostic|x[\s-]*ray|mri|ct\s*scan|'
    r'ultrasound|sonography|ecg|echo|pathology|biopsy|cbc|thyroid|'
    r'lipid\s*profile|vitamin\s*[bd]|checkup|health\s*check|'
    r'thyrocare|lal\s*pathlabs|aarthi\s*scans|metropolis|orange\s*health|healthians)\b'
)


def main():
    # 1. Load & Clean Main Sheet Data
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={'head': 'Head', 'category': 'Category'}, inplace=True)
    df = df[~df['Head'].str.strip().str.lower().str.contains(r'dummy|^t\s*est|^tes$', regex=True, na=False)]
    df = df[pd.to_numeric(df['amount'], errors='coerce') > 1.0].copy()
    df['sample_weight'] = 1.0

    # Merge human-verified corrections (preserving 0-amount items like 'Crape bandage')
    if os.path.exists(CORRECTIONS_PATH):
        corr_df = pd.read_csv(CORRECTIONS_PATH)
        corr_df.rename(columns={'head': 'Head', 'category': 'Category'}, inplace=True)
        corr_df = corr_df[
            ~corr_df['Head'].str.strip().str.lower().str.contains(r'dummy|^t\s*est|^tes$', regex=True, na=False)
        ].copy()
        corr_df['amount'] = pd.to_numeric(corr_df['amount'], errors='coerce').fillna(0.0)
        corr_df['sample_weight'] = 2.0
        df = pd.concat([df, corr_df], ignore_index=True)

    # Consolidate tiny overlapping categories
    df['Category'] = df['Category'].replace({'Shopping': 'Accessories'})

    # Route diagnostic & blood tests to 'Medical'
    combined_text = (df['Head'].fillna('') + ' ' + df['comment'].fillna('')).str.lower()
    df.loc[combined_text.str.contains(DIAGNOSTIC_PATTERN, regex=True, na=False), 'Category'] = 'Medical'

    # Filter out categories with fewer than 5 samples so stratified splits work cleanly
    counts = df['Category'].value_counts()
    df = df[df['Category'].isin(counts[counts >= 5].index)].copy()

    X = df[['Head', 'comment', 'amount']]
    y = df['Category']
    weights = df['sample_weight']

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Build 3-Parameter Pipeline (Word + Char N-Grams on Head+Comment + Weighted Log-Amount)
    text_features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2), stop_words='english')),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, min_df=2, ngram_range=(3, 5)))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_features, 'text'),
            ('amount', StandardScaler(), ['log_amount'])
        ],
        transformer_weights={'text': 1.0, 'amount': 0.35}
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

    # 3. Evaluate on 20% Holdout (with sample weights)
    pipeline.fit(X_train, y_train, clf__sample_weight=w_train)
    y_pred = pipeline.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    # 4. Retrain on 100% Data & Save
    pipeline.fit(X, y, clf__sample_weight=weights)
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Saved 3-parameter model to {MODEL_OUTPUT_PATH}")

    # 5. Sanity Check on Previously Low-Confidence Examples
    samples = pd.DataFrame([
        {"Head": "Appliances", "comment": "", "amount": 2528.10},
        {"Head": "Milk", "comment": "", "amount": 211.0},
        {"Head": "Flight ticket to n from Patna", "comment": "", "amount": 0.0},
        {"Head": "Cook", "comment": "", "amount": 0.0},
        {"Head": "Trolly bag", "comment": "", "amount": 0.0},
        {"Head": "Baby shampoo", "comment": "", "amount": 1755.0},
        {"Head": "ऑटो", "comment": "", "amount": 107.0},
        {"Head": "Hotter recho", "comment": "JioHotstar", "amount": 149.0},
    ])
    preds = pipeline.predict(samples)
    probs = pipeline.predict_proba(samples).max(axis=1)
    for row, pred, prob in zip(samples.to_dict(orient="records"), preds, probs):
        print(f"{row['Head']} (₹{row['amount']}) -> {pred} (confidence: {prob:.2f})")


if __name__ == "__main__":
    main()