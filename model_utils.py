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
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


# In model_utils.py
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    head_series = df['Head'].fillna('').astype(str)
    comment_series = df['comment'].fillna('').astype(str)
    out['text'] = (head_series + ' ' + comment_series).apply(clean_text)

    amounts = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    # Replace <= 1.0 (missing/placeholder amounts) with a neutral typical expense (~₹300)
    # so missing amounts don't push log_amount to an extreme outlier
    neutral_amounts = np.where(amounts <= 1.0, 300.0, amounts)
    out['log_amount'] = np.log1p(neutral_amounts).astype(float)
    return out