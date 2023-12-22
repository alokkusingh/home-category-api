#!/usr/bin/env python
# coding: utf-8

import grpc
from concurrent import futures
from stream.grpc.stub import expenseCategorizer_pb2_grpc
from stream.grpc.expenseCategorizerServer import ExpenseCategorizerServer
#from module.expenseCategorizer import ExpenseCategorizer

#expenseCategorizer = ExpenseCategorizer();

#print(expenseCategorizer.getCategory("ranjeet"));


def serve():
    # initialize server with 2 workers
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

    # attach servicer method to the server
    expenseCategorizer_pb2_grpc.add_ExpenseCategorizerServicer_to_server(ExpenseCategorizerServer(), server)

    # start the server on the port 50051
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("Started gRPC server: 0.0.0.0:50051")

    # server loop to keep the process running
    server.wait_for_termination()


# invoke the server method
if __name__ == "__main__":
    serve()
