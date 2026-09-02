"""
merge_dos_ddos.py — merges DoS and DDoS into a single "DoS/DDoS" class,
building on the existing 6-class filtered data (which already dropped
Brute Force and Web-based). Result: 5 final classes.

Rationale (for the report): DoS and DDoS exhibit overlapping feature
distributions as flood-based attacks when viewed through per-flow
statistics alone — the 6-class retrain showed 64% of true DDoS flows
misclassified as DoS, with the confusion asymmetric enough to indicate a
structural feature-granularity limit (the frozen features have no
cross-flow signal like distinct-source-IP-count, which is what actually
differentiates single-source DoS from multi-source DDoS). Rather than
continuing to fight this via loss-weighting tuning, the two are merged at
the CNN layer; distinguishing single- vs multi-source floods is deferred
to the Risk Engine's per-source-IP sliding-window tracking (Phase 9),
which has access to the cross-flow context the CNN structurally lacks.

Uses an EXPLICIT manual re-mapping (not LabelEncoder.fit()), consistent
with the fix applied earlier in this project after the alphabetical-sort
bug was found.

Run locally (CPU-only, fast):
    python merge_dos_ddos.py
"""

import json
import os

import numpy as np

PROCESSED_DIR = os.path.join("data", "processed")

X_TRAIN_6CLASS = os.path.join(PROCESSED_DIR, "X_train_6class.npy")
Y_TRAIN_6CLASS = os.path.join(PROCESSED_DIR, "y_train_6class.npy")
X_TEST_6CLASS = os.path.join(PROCESSED_DIR, "X_test_6class.npy")
Y_TEST_6CLASS = os.path.join(PROCESSED_DIR, "y_test_6class.npy")
CLASS_ORDER_6_PATH = os.path.join(PROCESSED_DIR, "class_order_6class.json")

X_TRAIN_OUT = os.path.join(PROCESSED_DIR, "X_train_5class.npy")
Y_TRAIN_OUT = os.path.join(PROCESSED_DIR, "y_train_5class.npy")
X_TEST_OUT = os.path.join(PROCESSED_DIR, "X_test_5class.npy")
Y_TEST_OUT = os.path.join(PROCESSED_DIR, "y_test_5class.npy")
CLASS_ORDER_5_PATH = os.path.join(PROCESSED_DIR, "class_order_5class.json")

MERGED_CLASS_NAME = "DoS/DDoS"


def main():
    with open(CLASS_ORDER_6_PATH) as f:
        schema_6 = json.load(f)
    class_order_6 = schema_6["class_order"]
    print(f"6-class order (input): {class_order_6}")

    dos_idx = class_order_6.index("DoS")
    ddos_idx = class_order_6.index("DDoS")

    # Build the new 5-class order: keep everything except DoS/DDoS,
    # insert the merged class name once, explicit and unambiguous.
    class_order_5 = [MERGED_CLASS_NAME] + [
        c for c in class_order_6 if c not in ("DoS", "DDoS")
    ]
    print(f"5-class order (output): {class_order_5}")

    # old 6-class index -> new 5-class index, built explicitly
    old_to_new = {}
    for old_idx, name in enumerate(class_order_6):
        if name in ("DoS", "DDoS"):
            old_to_new[old_idx] = class_order_5.index(MERGED_CLASS_NAME)
        else:
            old_to_new[old_idx] = class_order_5.index(name)

    print("\nOld (6-class) index -> New (5-class) index mapping:")
    for old_idx, new_idx in sorted(old_to_new.items()):
        print(f"  {old_idx} ({class_order_6[old_idx]}) -> {new_idx} ({class_order_5[new_idx]})")

    for split_name, x_path, y_path, x_out, y_out in [
        ("train", X_TRAIN_6CLASS, Y_TRAIN_6CLASS, X_TRAIN_OUT, Y_TRAIN_OUT),
        ("test", X_TEST_6CLASS, Y_TEST_6CLASS, X_TEST_OUT, Y_TEST_OUT),
    ]:
        print(f"\nProcessing {split_name} split ...")
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"  Input shape: X={X.shape}, y={y.shape}")

        y_new = np.array([old_to_new[old] for old in y], dtype=y.dtype)

        classes, counts = np.unique(y_new, return_counts=True)
        print(f"  New class distribution:")
        for cls, count in zip(classes, counts):
            print(f"    {class_order_5[cls]}: {count}")
        print(f"  New max/min ratio: {counts.max() / counts.min():.1f}x")

        np.save(x_out, X)  # X unchanged — only labels are remapped
        np.save(y_out, y_new)
        print(f"  Saved: {x_out}")
        print(f"  Saved: {y_out}")

    with open(CLASS_ORDER_5_PATH, "w") as f:
        json.dump({
            "class_order": class_order_5,
            "merged_classes": {"DoS/DDoS": ["DoS", "DDoS"]},
            "dropped_classes": schema_6["dropped_classes"],
            "note": "DoS and DDoS merged after the 6-class model showed 64% "
                     "of true DDoS flows misclassified as DoS — diagnosed as "
                     "a per-flow feature-granularity limit (no cross-flow "
                     "source-IP-diversity signal available). Distinguishing "
                     "single- vs multi-source floods is deferred to the Risk "
                     "Engine's per-IP sliding-window tracking (Phase 9).",
        }, f, indent=2)
    print(f"\nSaved: {CLASS_ORDER_5_PATH}")
    print("\nUpload X_train_5class.npy, y_train_5class.npy, X_test_5class.npy, "
          "y_test_5class.npy, and class_order_5class.json to your Colab Drive "
          "folder, then retrain with the 5-class cells.")


if __name__ == "__main__":
    main()