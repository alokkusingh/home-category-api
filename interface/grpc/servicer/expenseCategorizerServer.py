import logging

from module.expenseCategorizer import ExpenseCategorizer
from interface.grpc.servicer.generated import expenseCategorizer_pb2
from interface.grpc.servicer.generated import expenseCategorizer_pb2_grpc


class ExpenseCategorizerServer(expenseCategorizer_pb2_grpc.ExpenseCategorizerServicer):
    def __init__(self):
        self.expenseCategorizer = ExpenseCategorizer();

    def getExpenseCategory(self, request_iterator, context):
        for request in request_iterator:
            print("Request received")
            logging.info("Request Received: %s", request)
            response = {
                "category": self.expenseCategorizer.getCategory(request.head)[0]
            }
            print(response)
            logging.info("Response: %s", response)
            yield expenseCategorizer_pb2.ExpenseCategorizationResponse(**response)

    def getExpenseCategoryUnary(self, request, context):
        print("Request received")
        logging.info("Request Received: %s", request)
        response = {
            "category": self.expenseCategorizer.getCategory(request.head)[0]
        }
        logging.info("Response: %s", response)
        return expenseCategorizer_pb2.ExpenseCategorizationResponse(**response)