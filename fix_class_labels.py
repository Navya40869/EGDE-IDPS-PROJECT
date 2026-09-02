"""
fix_class_labels.py — corrects the label-name mismatch caused by LabelEncoder's
alphabetical sorting. Run this locally to get correctly-labeled per-class results,
WITHOUT retraining.
"""

import json
import os

import joblib

PROCESSED_DIR = os.path.join("data", "processed")
ENCODER_PATH = os.path.join(PROCESSED_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(PROCESSED_DIR, "model_metadata.json")

label_encoder = joblib.load(ENCODER_PATH)
true_class_order = list(label_encoder.classes_)
print("TRUE class order used during training (from label_encoder.classes_):")
print(true_class_order)

with open(METADATA_PATH) as f:
    metadata = json.load(f)

old_per_class_f1 = metadata["per_class_f1"]
old_names_in_saved_order = list(old_per_class_f1.keys())

corrected_per_class_f1 = {
    true_class_order[i]: old_per_class_f1[old_names_in_saved_order[i]]
    for i in range(len(true_class_order))
}

print("\nCorrected per-class F1:")
for cls, f1 in sorted(corrected_per_class_f1.items(), key=lambda x: x[1]):
    print(f"  {cls}: {f1:.4f}")

metadata["per_class_f1"] = corrected_per_class_f1
metadata["class_order_note"] = (
    "class_order corrected to match label_encoder.classes_ (true alphabetical "
    "training order) — earlier version had names mismatched due to sklearn "
    "LabelEncoder.fit() sorting alphabetically regardless of input order."
)
metadata["true_class_order"] = true_class_order

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\nFixed and saved: {METADATA_PATH}")