import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics

from model_utils import build_features

# 1. Load & Clean Data
df = pd.read_csv("~/Downloads/Expense Sheet - Categorized.csv")
df.rename(columns={'head': 'Head', 'category': 'Category'}, inplace=True)

df = df[~df['Head'].str.lower().str.contains('dummy|^test', na=False)]
df = df[df['amount'] > 1.0].copy()
df['Category'] = df['Category'].replace({'Shopping': 'Accessories'})

# Filter out categories with fewer than 5 samples so stratified 5-fold CV works cleanly
counts = df['Category'].value_counts()
df = df[df['Category'].isin(counts[counts >= 5].index)]

X = df[['Head', 'comment', 'amount']]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Build 3-Parameter Pipeline (Word + Char N-Grams on Head+Comment + Scaled Log-Amount)
text_features = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), stop_words='english')),
    ('char_tfidf', TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, min_df=3, ngram_range=(3, 5)))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_features, 'text'),
        ('amount', StandardScaler(), ['log_amount'])
    ],
    transformer_weights={
        'text': 1.0,
        'amount': 0.35  # Amount acts as a tie-breaker, never overrides clear keywords
    }
)

pipeline = Pipeline([
    ('feature_builder', FunctionTransformer(build_features, validate=False)),
    ('preprocessor', preprocessor),
    ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42), cv=5))
])

# 3. Evaluate on 20% Holdout
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

# 4. Retrain on 100% Data & Save
pipeline.fit(X, y)
joblib.dump(pipeline, 'model/expense-pipeline.joblib')
print("Saved 3-parameter model to expense-pipeline.joblib")
