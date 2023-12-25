#!/usr/bin/env python
# coding: utf-8

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from concurrent import futures
from stream.grpc.stub import expenseCategorizer_pb2_grpc
from stream.grpc.stub import expenseCategorizer_pb2
from stream.grpc.expenseCategorizerServer import ExpenseCategorizerServer

import argparse
import logging

_DESCRIPTION = "Home Analytics gRPC server"
_LISTEN_HOST_PORT = "0.0.0.0:50051"
_LISTEN_PORT = 50051
_THREAD_POOL_SIZE = 2

logger = logging.getLogger()
console_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s: %(levelname)-8s %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _configure_health_server(server: grpc.Server):
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=_THREAD_POOL_SIZE),
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    health_servicer.set(service="ExpenseCategorizer", status=health_pb2.HealthCheckResponse.SERVING)


def serve():
    # initialize server with 2 workers
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_POOL_SIZE)
    )

    # attach servicer method to the server
    expenseCategorizer_pb2_grpc.add_ExpenseCategorizerServicer_to_server(ExpenseCategorizerServer(), server)
    services = tuple(service.full_name for service in health_pb2.DESCRIPTOR.services_by_name.values())
    services += tuple(service.full_name for service in expenseCategorizer_pb2.DESCRIPTOR.services_by_name.values())
    services += (reflection.SERVICE_NAME,)

    reflection.enable_server_reflection(services, server)

    # start the server on the port 50051
    server.add_insecure_port(_LISTEN_HOST_PORT)

    _configure_health_server(server)

    server.start()
    logger.info("Started gRPC server: %s", _LISTEN_HOST_PORT)
    print("Started gRPC server: ", _LISTEN_HOST_PORT)

    # server loop to keep the process running
    server.wait_for_termination()


# invoke the server method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "port",
        default=50051,
        type=int,
        nargs="?",
        help="The port on which to listen.",
    )
    parser.add_argument(
        "hostname",
        type=str,
        default=None,
        nargs="?",
        help="The name clients will see in responses.",
    )
    parser.add_argument(
        "--xds-creds",
        action="store_true",
        help="If specified, uses xDS credentials to connect to the server.",
    )
    args = parser.parse_args()
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    serve()
