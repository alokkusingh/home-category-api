#!/usr/bin/env python
# coding: utf-8

import argparse
from concurrent import futures
import logging
import signal
import sys

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from interface.grpc.servicer.expenseCategorizerServer import ExpenseCategorizerServer
from interface.grpc.servicer.generated import expenseCategorizer_pb2
from interface.grpc.servicer.generated import expenseCategorizer_pb2_grpc
import interface.grpc.servicer.generated.expenseCategorizer_pb2 as expenseCategorizer__pb2

_DESCRIPTION = "Home Analytics gRPC server"
_DEFAULT_PORT = 50051
_DEFAULT_MODEL_PATH = "model/expense-pipeline.joblib"
_THREAD_POOL_SIZE = 4

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s: %(levelname)-8s %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _configure_health_server(server: grpc.Server) -> health.HealthServicer:
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=2),
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Register both short and full protobuf service names
    health_servicer.set(service="", status=health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set(service="ExpenseCategorizer", status=health_pb2.HealthCheckResponse.SERVING)
    for service in expenseCategorizer_pb2.DESCRIPTOR.services_by_name.values():
        health_servicer.set(service=service.full_name, status=health_pb2.HealthCheckResponse.SERVING)

    return health_servicer


def serve(port: int = _DEFAULT_PORT, model_path: str = _DEFAULT_MODEL_PATH):
    listen_addr = f"0.0.0.0:{port}"

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_POOL_SIZE)
    )

    # Initialize servicer with pre-trained pipeline loaded once at startup
    servicer = ExpenseCategorizerServer(model_path=model_path)
    expenseCategorizer_pb2_grpc.add_ExpenseCategorizerServicer_to_server(servicer, server)

    # Enable server reflection
    services = tuple(service.full_name for service in health_pb2.DESCRIPTOR.services_by_name.values())
    services += tuple(service.full_name for service in expenseCategorizer_pb2.DESCRIPTOR.services_by_name.values())
    services += (reflection.SERVICE_NAME,)
    reflection.enable_server_reflection(services, server)

    server.add_insecure_port(listen_addr)
    health_servicer = _configure_health_server(server)

    def _graceful_shutdown(signum, frame):
        logger.info("Received signal %s, marking services NOT_SERVING and shutting down...", signum)
        health_servicer.enter_graceful_shutdown()
        stopped_event = server.stop(grace=5)
        stopped_event.wait(timeout=5)
        logger.info("gRPC server stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    server.start()
    logger.info("Started gRPC server on %s using model '%s'", listen_addr, model_path)
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "port",
        default=_DEFAULT_PORT,
        type=int,
        nargs="?",
        help="The port on which to listen (default: 50051).",
    )
    parser.add_argument(
        "hostname",
        type=str,
        default=None,
        nargs="?",
        help="The name clients will see in responses.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help="Path to the serialized .joblib expense categorization pipeline.",
    )
    parser.add_argument(
        "--xds-creds",
        action="store_true",
        help="If specified, uses xDS credentials to connect to the server.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    serve(port=args.port, model_path=args.model_path)
