import logging
import joblib
import numpy as np
import pandas as pd
import grpc

from interface.grpc.servicer.generated import expenseCategorizer_pb2
from interface.grpc.servicer.generated import expenseCategorizer_pb2_grpc

logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD = 0.55


class ExpenseCategorizerServer(expenseCategorizer_pb2_grpc.ExpenseCategorizerServicer):
    def __init__(self, model_path: str = "expense-pipeline.joblib"):
        logger.info("Loading 3-parameter ML model from %s", model_path)
        self.pipeline = joblib.load(model_path)
        self.classes = self.pipeline.classes_

    def _predict_single(self, request) -> expenseCategorizer_pb2.ExpenseCategorizationResponse:
        input_df = pd.DataFrame([{
            "Head": request.head or "",
            "comment": request.comment or "",
            "amount": float(request.amount or 0.0),
        }])

        probs = self.pipeline.predict_proba(input_df)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        category = str(self.classes[best_idx])
        needs_review = confidence < CONFIDENCE_THRESHOLD

        if needs_review:
            logger.warning(
                "Low confidence (%.2f) for head='%s', amount=%.2f -> '%s'",
                confidence, request.head, request.amount, category
            )

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