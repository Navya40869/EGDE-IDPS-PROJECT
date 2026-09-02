"""
filter_classes.py — drops Brute Force and Web-based, re-encodes the
remaining 6 classes with an EXPLICIT manual mapping (not sklearn's
LabelEncoder, which silently sorts alphabetically and caused the earlier
label-mismatch bug). This guarantees CLASS_ORDER_6 below and the actual
encoded integers always agree.

Filters at the 37-COLUMN preprocessed level (before the 20-feature
selection), matching the existing pipeline pattern — frozen_features.json's
feature selection step still runs unchanged afterward, at training time.

Run locally (CPU-only, fast — no GPU needed):
    python filter_classes.py
"""

import json
import os

import numpy as np

from utils.config import CLASS_ORDER  # the original 8-class order

PROCESSED_DIR = os.path.join("data", "processed")
X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

X_TRAIN_OUT = os.path.join(PROCESSED_DIR, "X_train_6class.npy")
Y_TRAIN_OUT = os.path.join(PROCESSED_DIR, "y_train_6class.npy")
X_TEST_OUT = os.path.join(PROCESSED_DIR, "X_test_6class.npy")
Y_TEST_OUT = os.path.join(PROCESSED_DIR, "y_test_6class.npy")
CLASS_ORDER_6_PATH = os.path.join(PROCESSED_DIR, "class_order_6class.json")

DROPPED_CLASSES = ["Brute Force", "Web-based"]

# Explicit new order — deliberately NOT derived via LabelEncoder.fit(), to
# avoid a repeat of the earlier alphabetical-sort mismatch bug. This list
# IS what the new integer labels 0-5 mean, full stop — no re-derivation
# needed anywhere downstream.
CLASS_ORDER_6 = ["Benign", "DDoS", "DoS", "Mirai", "Reconnaissance", "Spoofing"]


def main():
    print(f"Original 8-class order: {CLASS_ORDER}")
    print(f"Dropping: {DROPPED_CLASSES}")
    print(f"New 6-class order: {CLASS_ORDER_6}\n")

    # Build old_index -> new_index mapping explicitly
    old_to_new = {}
    for new_idx, class_name in enumerate(CLASS_ORDER_6):
        old_idx = CLASS_ORDER.index(class_name)
        old_to_new[old_idx] = new_idx
    dropped_old_indices = {CLASS_ORDER.index(c) for c in DROPPED_CLASSES}

    print("Old index -> New index mapping:")
    for old_idx, new_idx in sorted(old_to_new.items()):
        print(f"  {old_idx} ({CLASS_ORDER[old_idx]}) -> {new_idx} ({CLASS_ORDER_6[new_idx]})")
    print(f"Dropped old indices: {dropped_old_indices} "
          f"({[CLASS_ORDER[i] for i in dropped_old_indices]})\n")

    for split_name, x_path, y_path, x_out, y_out in [
        ("train", X_TRAIN_PATH, Y_TRAIN_PATH, X_TRAIN_OUT, Y_TRAIN_OUT),
        ("test", X_TEST_PATH, Y_TEST_PATH, X_TEST_OUT, Y_TEST_OUT),
    ]:
        print(f"Processing {split_name} split ...")
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"  Original shape: X={X.shape}, y={y.shape}")

        # Keep only rows whose label is NOT in the dropped set
        keep_mask = ~np.isin(y, list(dropped_old_indices))
        X_filtered = X[keep_mask]
        y_filtered_old = y[keep_mask]

        # Re-encode old indices -> new indices explicitly (vectorized)
        y_filtered_new = np.array([old_to_new[old] for old in y_filtered_old], dtype=y.dtype)

        print(f"  Filtered shape: X={X_filtered.shape}, y={y_filtered_new.shape} "
              f"({X.shape[0] - X_filtered.shape[0]} rows dropped)")

        classes, counts = np.unique(y_filtered_new, return_counts=True)
        print(f"  New class distribution:")
        for cls, count in zip(classes, counts):
            print(f"    {CLASS_ORDER_6[cls]}: {count}")
        if len(counts) > 0:
            print(f"  New max/min ratio: {counts.max() / counts.min():.1f}x")

        np.save(x_out, X_filtered)
        np.save(y_out, y_filtered_new)
        print(f"  Saved: {x_out}")
        print(f"  Saved: {y_out}\n")

    with open(CLASS_ORDER_6_PATH, "w") as f:
        json.dump({
            "class_order": CLASS_ORDER_6,
            "dropped_classes": DROPPED_CLASSES,
            "note": "Explicit manual encoding — NOT via LabelEncoder.fit(), "
                    "to avoid the alphabetical-sort mismatch bug found in the "
                    "original 8-class model.",
        }, f, indent=2)
    print(f"Saved: {CLASS_ORDER_6_PATH}")
    print("\nUpload X_train_6class.npy, y_train_6class.npy, X_test_6class.npy, "
          "y_test_6class.npy, and class_order_6class.json to your Colab Drive "
          "folder, then use the 6-class training cells to retrain.")


if __name__ == "__main__":
    main()