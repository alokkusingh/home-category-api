#!/usr/bin/env python3
import os
import sys
import grpc

from interface.grpc.servicer.generated import expenseCategorizer_pb2
from interface.grpc.servicer.generated import expenseCategorizer_pb2_grpc

HOST = "127.0.0.1:50051"

# Enable ANSI colors in PyCharm Run Console and interactive terminals
USE_COLOR = sys.stdout.isatty() or os.environ.get("PYCHARM_HOSTED") == "1"

GREEN = "\033[92m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
CYAN = "\033[96m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

# (head, comment, amount, expected_category)
TEST_CASES = [
    # --- Core & Amount-Sensitivity Cases ---
    ("Chicken", "", 0.0, "Grocery"),
    ("Drawing class", "Something", 500.0, "Education"),
    ("birthday", "", 34000.0, "Celebration"),
    ("fitness", "", 19000.0, "Fitness"),
    ("Stationary", "", 100.0, "Education"),
    ("Physiotherepist", "", 100.0, "Medical"),
    ("Physiotherapist", "", 600.0, "Medical"),
    ("Physio therapist", "", 600.0, "Medical"),
    ("Physio", "", 700.0, "Medical"),
    ("Auto", "", 100.0, "Travel"),
    ("Dishwasher", "", 459.0, "Grocery"),
    (
        "Dishwasher",
        "Rs 2556.00 at FLIPKART PAYMENTS on 10-10-2024 19%3A39%3A23,2556.0",
        2556.0,
        "Grocery",
    ),
    (
        "Dishwasher",
        "INR 32472 at Flipkart In on 10-10-2024 17%3A59%3A32",
        32472.0,
        "Appliances",
    ),
    ("Protein Powder", "", 5000.0, "Fitness"),

    # --- Real CSV Edge Cases: Generic Headers Disambiguated by Comment ---
    (
        "Groceries",
        "Rs 1108.00 has been debited from account 0531 to VPA firstcry.payu@hdfcbank - Groviva, pediacure",
        1108.0,
        "Baby Care",
    ),
    (
        "Groceries",
        "Rs 1556.00 has been debited from account 0531 to VPA swiggyinstamart@icici - Pediasure+colour",
        1556.0,
        "Baby Care",
    ),
    (
        "Lunch",
        "Rs 85.00 has been debited from account 0531 to VPA compassindiafoodserrzb@mairtel",
        85.0,
        "Food Outside",
    ),
    (
        "Dress",
        "INR 1477.00 spent on your SBI Credit Card ending 1234 at MYNTRA DESIGNS",
        1477.0,
        "Accessories",
    ),
    (
        "Education",
        "Anthropic License",
        532.0,
        "Education",
    ),

    # --- Real CSV Edge Cases: Diagnostic & Medical ---
    ("Blood test", "Thyrocare", 1200.0, "Medical"),
    ("Scan", "Aarthi Scans ultrasound", 2500.0, "Medical"),
    (
        "Medicine",
        "Rs 640.00 debited to VPA apollopharmacy@icici",
        640.0,
        "Medical",
    ),

    # --- Real CSV Edge Cases: Household, Maintenance & Typos ---
    ("Magic bricks", "Tatinum pro plan", 3699.0, "Bills"),
    ("Society helper", "", 200.0, "House Help"),
    ("Blouse stiching", "", 3800.0, "Accessories"),
    ("Shoet", "Saanvi, Alok, Rachna", 3772.0, "Accessories"),
    ("Mill", "Flour mill", 500.0, "Milk"),
    ("Watee", "", 150.0, "Grocery"),
    ("Snacs", "", 210.0, "Food Outside"),
]


def run_validation() -> int:
    channel = grpc.insecure_channel(HOST)
    try:
        grpc.channel_ready_future(channel).result(timeout=3)
    except grpc.FutureTimeoutError:
        print(f"{BOLD}{RED}ERROR: Could not connect to gRPC server at {HOST}{RESET}")
        return 1

    stub = expenseCategorizer_pb2_grpc.ExpenseCategorizerStub(channel)

    print(f"\n{BOLD}{CYAN}Running {len(TEST_CASES)} gRPC model validation tests against {HOST}...{RESET}\n")
    header = f"{'STATUS':<8} {'HEAD':<20} {'AMOUNT':>10} {'EXPECTED':<15} {'ACTUAL':<15} {'CONF':>7}"
    print(f"{BOLD}{header}{RESET}")
    print("-" * len(header))

    passed = 0
    failed = 0

    # 1. Validate Unary RPC for each test case
    for head, comment, amount, expected in TEST_CASES:
        req = expenseCategorizer_pb2.ExpenseCategorizationRequest(
            head=head,
            comment=comment,
            amount=amount,
        )
        try:
            resp = stub.getExpenseCategoryUnary(req)
            actual = resp.category
            conf = resp.confidence
            ok = actual == expected
        except grpc.RpcError as err:
            actual = f"ERR:{err.code().name}"
            conf = 0.0
            ok = False

        if ok:
            passed += 1
        else:
            failed += 1

        status_color = GREEN if ok else RED
        padded_status = f"{BOLD}{status_color}{'PASS' if ok else 'FAIL':<8}{RESET}"
        conf_color = YELLOW if conf < 0.60 else GREEN
        padded_conf = f"{conf_color}{conf:>6.2%}{RESET}"

        print(f"{padded_status} {head:<20} {amount:>10.2f} {expected:<15} {actual:<15} {padded_conf}")

    # 2. Validate Bidirectional Streaming RPC in one batch
    requests = [
        expenseCategorizer_pb2.ExpenseCategorizationRequest(head=h, comment=c, amount=a)
        for h, c, a, _ in TEST_CASES
    ]
    stream_responses = list(stub.getExpenseCategory(iter(requests)))
    stream_ok = len(stream_responses) == len(TEST_CASES) and all(
        r.category == expected for r, (_, _, _, expected) in zip(stream_responses, TEST_CASES)
    )

    summary_color = GREEN if (failed == 0 and stream_ok) else RED
    print("-" * len(header))
    print(f"Unary Summary:     {BOLD}{summary_color}{passed}/{len(TEST_CASES)} passed, {failed} failed{RESET}")
    print(
        f"Streaming Batch:   {BOLD}{GREEN if stream_ok else RED}"
        f"{'PASS' if stream_ok else 'FAIL'}{RESET} ({len(stream_responses)}/{len(TEST_CASES)} responses)\n"
    )

    channel.close()
    return 0 if (failed == 0 and stream_ok) else 1


if __name__ == "__main__":
    sys.exit(run_validation())