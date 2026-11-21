#!/usr/bin/env python3
import sys
import grpc

from interface.grpc.servicer.generated import expenseCategorizer_pb2
from interface.grpc.servicer.generated import expenseCategorizer_pb2_grpc

HOST = "127.0.0.1:50051"

# (head, comment, amount, expected_category)
TEST_CASES = [
    # --- Core & Amount-Sensitivity Cases ---
    ("Chicken", "", 0.0, "Grocery"),
    ("Drawing class", "Something", 500.0, "Education"),
    ("birthday", "", 34000.0, "Celebration"),
    ("fitness", "", 1090500.0, "Fitness"),
    ("Stationary", "", 100.0, "Education"),
    ("Physiotherepist", "", 100.0, "Medical"),
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
        print(f"ERROR: Could not connect to gRPC server at {HOST}")
        return 1

    stub = expenseCategorizer_pb2_grpc.ExpenseCategorizerStub(channel)

    print(f"\nRunning {len(TEST_CASES)} gRPC model validation tests against {HOST}...\n")
    header = f"{'STATUS':<8} {'HEAD':<20} {'AMOUNT':>10} {'EXPECTED':<15} {'ACTUAL':<15} {'CONF':>7}"
    print(header)
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

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"{status:<8} {head:<20} {amount:>10.2f} {expected:<15} {actual:<15} {conf:>6.2%}")

    # 2. Validate Bidirectional Streaming RPC in one batch
    requests = [
        expenseCategorizer_pb2.ExpenseCategorizationRequest(head=h, comment=c, amount=a)
        for h, c, a, _ in TEST_CASES
    ]
    stream_responses = list(stub.getExpenseCategory(iter(requests)))
    stream_ok = len(stream_responses) == len(TEST_CASES) and all(
        r.category == expected for r, (_, _, _, expected) in zip(stream_responses, TEST_CASES)
    )

    print("-" * len(header))
    print(f"Unary Summary:     {passed}/{len(TEST_CASES)} passed, {failed} failed")
    print(f"Streaming Batch:   {'PASS' if stream_ok else 'FAIL'} ({len(stream_responses)}/{len(TEST_CASES)} responses)\n")

    channel.close()
    return 0 if (failed == 0 and stream_ok) else 1


if __name__ == "__main__":
    sys.exit(run_validation())