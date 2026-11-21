import re
import urllib.parse
import numpy as np
import pandas as pd


def clean_text(text: str) -> str:
    text = urllib.parse.unquote(str(text))
    text = re.sub(r'has been debited from account.*?to VPA', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'spent on your SBI Credit Card ending \d+ at', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'UPI transaction reference no.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'To check your available balance.*', ' ', text, flags=re.IGNORECASE)
    # Keep Unicode letters (English + Hindi/Devanagari) while removing digits & punctuation
    text = re.sub(r'[\d_]|[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    head_series = df['Head'].fillna('').astype(str)
    comment_series = df['comment'].fillna('').astype(str)
    out['text'] = (head_series + ' ' + comment_series).apply(clean_text)

    amounts = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    # Impute missing/placeholder amounts (<= 1.0) with a neutral typical expense (~300 INR)
    # so zero-amount entries do not skew log_amount to extreme negative z-scores.
    neutral_amounts = np.where(amounts <= 1.0, 300.0, amounts)
    out['log_amount'] = np.log1p(neutral_amounts).astype(float)
    return out