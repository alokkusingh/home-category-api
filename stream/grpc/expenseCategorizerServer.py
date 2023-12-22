
from module.expenseCategorizer import ExpenseCategorizer
from stream.grpc.stub import expenseCategorizer_pb2_grpc, expenseCategorizer_pb2


class ExpenseCategorizerServer(expenseCategorizer_pb2_grpc.ExpenseCategorizerServicer):
    def __init__(self):
        self.expenseCategorizer = ExpenseCategorizer();

    def getExpenseCategory(self, request_iterator, context):
        for request in request_iterator:
            print("Request received")
            response = {
                "category": self.expenseCategorizer.getCategory(request.head)[0]
            }
            print(response)
            yield expenseCategorizer_pb2.ExpenseCategorizationResponse(**response)

    def getExpenseCategoryUnary(self, request, context):
        print("Request received")
        response = {
            "category": self.expenseCategorizer.getCategory(request.head)[0]
        }
        return expenseCategorizer_pb2.ExpenseCategorizationResponse(**response)