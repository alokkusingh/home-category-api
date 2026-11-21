#!/usr/bin/env python3
import csv
import os
import sys
import joblib

PENDING_PATH = os.getenv("PENDING_CORRECTIONS_PATH", "../data/corrections_pending.csv")
CORRECTIONS_PATH = os.getenv("CORRECTIONS_PATH", "../data/corrections.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "../model/expense-pipeline.joblib")


def load_known_categories() -> list[str]:
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            return sorted(str(c) for c in pipeline.classes_)
        except Exception:
            pass
    return []


def append_to_corrections(rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(CORRECTIONS_PATH) or ".", exist_ok=True)
    file_exists = os.path.exists(CORRECTIONS_PATH) and os.path.getsize(CORRECTIONS_PATH) > 0

    with open(CORRECTIONS_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["head", "comment", "amount", "category"])
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "head": r["head"],
                "comment": r["comment"],
                "amount": r["amount"],
                "category": r["category"],
            })


def rewrite_pending(remaining_rows: list[dict]) -> None:
    fieldnames = ["timestamp", "head", "comment", "amount", "predicted_category", "confidence", "category"]
    with open(PENDING_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remaining_rows)


def main() -> int:
    if not os.path.exists(PENDING_PATH):
        print(f"No pending corrections file found at {PENDING_PATH}.")
        return 0

    with open(PENDING_PATH, mode="r", newline="", encoding="utf-8") as f:
        pending_rows = list(csv.DictReader(f))

    if not pending_rows:
        print("No pending low-confidence rows to review!")
        return 0

    known_categories = load_known_categories()
    cat_lookup = {c.lower(): c for c in known_categories}

    print(f"\nFound {len(pending_rows)} pending row(s) in {PENDING_PATH}.")
    if known_categories:
        print("\nKnown Categories:")
        for idx, cat in enumerate(known_categories, start=1):
            print(f"  {idx:>2}. {cat}")

    print("\nControls:")
    print("  [Enter]       Accept predicted category")
    print("  [Number/Text] Assign specific category (e.g. '3' or 'Medical')")
    print("  [s]           Skip for later")
    print("  [d]           Discard row (e.g. dummy/noise entry)")
    print("  [q]           Save progress and quit\n")

    approved: list[dict] = []
    skipped: list[dict] = []
    discarded_count = 0

    for i, row in enumerate(pending_rows, start=1):
        head = row.get("head", "")
        comment = row.get("comment", "")
        amount = row.get("amount", "0")
        predicted = row.get("predicted_category", "")
        conf = float(row.get("confidence", 0.0) or 0.0)

        # If user already filled 'category' manually in CSV, auto-approve it
        pre_filled = (row.get("category") or "").strip()
        if pre_filled:
            row["category"] = pre_filled
            approved.append(row)
            continue

        print("-" * 72)
        print(f"[{i}/{len(pending_rows)}] Head: '{head}' | Amount: ₹{amount}")
        if comment:
            print(f"        Comment: {comment[:140]}{'...' if len(comment) > 140 else ''}")
        print(f"        Predicted: \033[93m{predicted}\033[0m (confidence: {conf:.2%})")

        while True:
            choice = input(f"Confirm [{predicted}] or enter category/number (s/d/q): ").strip()

            if choice == "":
                row["category"] = predicted
                approved.append(row)
                print(f"  -> Approved as \033[92m{predicted}\033[0m")
                break
            elif choice.lower() == "s":
                skipped.append(row)
                print("  -> Skipped")
                break
            elif choice.lower() == "d":
                discarded_count += 1
                print("  -> Discarded")
                break
            elif choice.lower() == "q":
                skipped.extend(pending_rows[i - 1 :])
                append_to_corrections(approved)
                rewrite_pending(skipped)
                print(f"\nSaved {len(approved)} approved row(s) to {CORRECTIONS_PATH}. Remaining: {len(skipped)}.")
                return 0
            elif choice.isdigit() and 1 <= int(choice) <= len(known_categories):
                chosen_cat = known_categories[int(choice) - 1]
                row["category"] = chosen_cat
                approved.append(row)
                print(f"  -> Labeled as \033[92m{chosen_cat}\033[0m")
                break
            elif choice.lower() in cat_lookup:
                chosen_cat = cat_lookup[choice.lower()]
                row["category"] = chosen_cat
                approved.append(row)
                print(f"  -> Labeled as \033[92m{chosen_cat}\033[0m")
                break
            else:
                confirm_new = input(f"  '{choice}' is not in existing model classes. Use anyway? (y/n): ").strip().lower()
                if confirm_new == "y":
                    row["category"] = choice
                    approved.append(row)
                    print(f"  -> Labeled as \033[92m{choice}\033[0m")
                    break

    append_to_corrections(approved)
    rewrite_pending(skipped)

    print("\n" + "=" * 72)
    print(f"Review Complete! Approved: {len(approved)} | Discarded: {discarded_count} | Remaining: {len(skipped)}")
    if approved:
        print(f"Appended {len(approved)} verified row(s) to {CORRECTIONS_PATH}.")
        print("Next step: run `python train_model.py && python test/test_grpc_model.py`")

    return 0


if __name__ == "__main__":
    sys.exit(main())