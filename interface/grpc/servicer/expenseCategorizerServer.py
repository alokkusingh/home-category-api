from __future__ import annotations

import csv
import logging
import os
import re
import threading
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import grpc

from interface.grpc.servicer.generated import expenseCategorizer_pb2
from interface.grpc.servicer.generated import expenseCategorizer_pb2_grpc

logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD = 0.55
PENDING_CORRECTIONS_PATH = os.getenv("PENDING_CORRECTIONS_PATH", "data/corrections_pending.csv")

# Targeted high-precision keyword overrides for rare terms / common typos
KEYWORD_RULES = [
    (
        re.compile(
            r'\b(blood\s*test|lab\s*test|diagnostic|x[\s-]*ray|mri|ct\s*scan|'
            r'ultrasound|sonography|ecg|echo|pathology|biopsy|cbc|thyroid|'
            r'lipid\s*profile|vitamin\s*[bd]|checkup|health\s*check|'
            r'thyrocare|lal\s*pathlabs|aarthi\s*scans|metropolis|orange\s*health|healthians)\b',
            re.IGNORECASE,
        ),
        "Medical",
    ),
    (re.compile(r'\b(dummy)\b', re.IGNORECASE), "Other"),
    (re.compile(r'physiotherap', re.IGNORECASE), "Medical"),
    (re.compile(r'nursing', re.IGNORECASE), "Medical"),
    (re.compile(r'pollution\s*cert', re.IGNORECASE), "Maintenance"),
    (re.compile(r'magic\s*bricks', re.IGNORECASE), "Bills"),
    (re.compile(r'society\s*helper', re.IGNORECASE), "House Help"),
    (re.compile(r'\bcook\b', re.IGNORECASE), "House Help"),
    (re.compile(r'stiching|stitching|tailor', re.IGNORECASE), "Accessories"),
    (re.compile(r'trolly|trolley', re.IGNORECASE), "Accessories"),
    (re.compile(r'bracelet', re.IGNORECASE), "Accessories"),
    (re.compile(r'\b(snacs|snax)\b', re.IGNORECASE), "Food Outside"),
    (re.compile(r'\bwatee\b', re.IGNORECASE), "Grocery"),
    (re.compile(r'\bpuja\b', re.IGNORECASE), "Other"),
    (re.compile(r'\bcracker\b', re.IGNORECASE), "Celebration"),
    (re.compile(r'\bdoll\b', re.IGNORECASE), "Toys")
]


class ExpenseCategorizerServer(expenseCategorizer_pb2_grpc.ExpenseCategorizerServicer):
    def __init__(self, model_path: str = "model/expense-pipeline.joblib"):
        logger.info("Loading 3-parameter ML model from %s", model_path)
        self.pipeline = joblib.load(model_path)
        self.classes = self.pipeline.classes_

        self.exact_category_map = {str(c).lower(): str(c) for c in self.classes}
        self.exact_category_map.update({
            "stationary": "Education",
            "snacs": "Food Outside",
            "watee": "Grocery",
        })

        self._csv_lock = threading.Lock()
        self._logged_keys: set[tuple[str, str, float]] = set()
        self._init_pending_csv()

    def _rule_lookup(self, head: str, comment: str = "") -> str | None:
        cleaned_head = head.strip().lower()
        if cleaned_head in self.exact_category_map:
            return self.exact_category_map[cleaned_head]

        combined = f"{head} {comment}".strip()
        for pattern, category in KEYWORD_RULES:
            if pattern.search(combined):
                return category
        return None

    def _init_pending_csv(self) -> None:
        os.makedirs(os.path.dirname(PENDING_CORRECTIONS_PATH) or ".", exist_ok=True)
        if os.path.exists(PENDING_CORRECTIONS_PATH):
            with open(PENDING_CORRECTIONS_PATH, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        row.get("head", "").strip().lower(),
                        row.get("comment", "").strip().lower(),
                        round(float(row.get("amount", 0.0) or 0.0), 2),
                    )
                    self._logged_keys.add(key)
        else:
            with open(PENDING_CORRECTIONS_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "head",
                    "comment",
                    "amount",
                    "predicted_category",
                    "confidence",
                    "category",  # Fill this column with the true label before moving to corrections.csv
                ])

    def _log_low_confidence(self, head: str, comment: str, amount: float, predicted: str, confidence: float) -> None:
        dedup_key = (head.strip().lower(), comment.strip().lower(), round(amount, 2))
        with self._csv_lock:
            if dedup_key in self._logged_keys:
                return
            self._logged_keys.add(dedup_key)

            with open(PENDING_CORRECTIONS_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    head,
                    comment,
                    round(amount, 2),
                    predicted,
                    round(confidence, 4),
                    "",  # Leave blank for human confirmation
                ])

    def _predict_single(self, request) -> expenseCategorizer_pb2.ExpenseCategorizationResponse:
        head = request.head or ""
        comment = request.comment or ""
        amount = float(request.amount or 0.0)

        rule_category = self._rule_lookup(head, comment)
        if rule_category is not None:
            return expenseCategorizer_pb2.ExpenseCategorizationResponse(
                category=rule_category,
                confidence=1.0,
                needs_review=False,
            )

        input_df = pd.DataFrame([{
            "Head": head,
            "comment": comment,
            "amount": amount,
        }])

        probs = self.pipeline.predict_proba(input_df)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        category = str(self.classes[best_idx])
        needs_review = confidence < CONFIDENCE_THRESHOLD

        if needs_review:
            logger.warning(
                "Low confidence (%.2f) for head='%s', amount=%.2f -> '%s'",
                confidence, head, amount, category
            )
            self._log_low_confidence(head, comment, amount, category, confidence)

        return expenseCategorizer_pb2.ExpenseCategorizationResponse(
            category=category,
            confidence=confidence,
            needs_review=needs_review,
        )

    def getExpenseCategoryUnary(self, request, context):
        try:
            return self._predict_single(request)
        except Exception as exc:
            logger.exception("Unary categorization failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference failed: {exc}")
            return expenseCategorizer_pb2.ExpenseCategorizationResponse()

    def getExpenseCategory(self, request_iterator, context):
        try:
            for request in request_iterator:
                yield self._predict_single(request)
        except Exception as exc:
            logger.exception("Streaming categorization failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Streaming inference failed: {exc}")