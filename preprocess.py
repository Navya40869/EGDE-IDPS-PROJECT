"""
preprocess.py — Phase 2: Data Preprocessing

Pipeline (leakage-safe order):
    Load Stratified_data.csv
    -> Map labels (utils/config.py LABEL_MAP -> 8 classes)
    -> Drop excluded columns (Number, Variance — multi-flow aggregates)
    -> Drop rows with missing values
    -> Train/Test split (stratified by class)
    -> Fit StandardScaler on TRAIN split only
    -> Transform train and test
    -> Save scaler.pkl, label_encoder.pkl, model_metadata.json, X_test.npy

Run from the project root, with the venv activated:
    python preprocess.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.config import LABEL_MAP, CLASS_ORDER, map_labels

# ---- Paths ----
INPUT_CSV = "Stratified_data.csv"
OUTPUT_DIR = os.path.join("data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Columns to exclude per blueprint (multi-flow aggregates, not single-flow computable) ----
EXCLUDED_COLUMNS = ["Number", "Variance"]

# ---- Split settings ----
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded shape: {df.shape}\n")

    # --- Map labels to 8 high-level classes ---
    print("Mapping labels via utils/config.py LABEL_MAP ...")
    df = map_labels(df, label_col="Label")
    print("Class distribution after mapping:")
    print(df["class"].value_counts(), "\n")

    # --- Drop excluded columns ---
    present_excluded = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    print(f"Dropping excluded columns: {present_excluded}")
    df = df.drop(columns=present_excluded)

    # --- Replace infinite values with NaN, then drop all missing/inf rows ---
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} rows with missing/infinite values ({before} -> {after})\n")

    # --- Separate features / target ---
    non_feature_cols = ["Label", "class"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].copy()
    y = df["class"].copy()

    print(f"Feature columns ({len(feature_cols)}):")
    print(feature_cols, "\n")

    # --- Encode labels ---
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_ORDER)  # fixed order for consistency across retrains
    y_encoded = label_encoder.transform(y)

    # --- Train/test split (BEFORE scaling — no leakage) ---
    print(f"Splitting train/test (test_size={TEST_SIZE}, stratified by class) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}\n")

    # --- Fit StandardScaler on TRAIN ONLY ---
    print("Fitting StandardScaler on train split only ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only, never fit, on test

    # --- Save artifacts ---
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    xtest_path = os.path.join(OUTPUT_DIR, "X_test.npy")
    ytest_path = os.path.join(OUTPUT_DIR, "y_test.npy")
    xtrain_path = os.path.join(OUTPUT_DIR, "X_train.npy")
    ytrain_path = os.path.join(OUTPUT_DIR, "y_train.npy")

    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    np.save(xtest_path, X_test_scaled)
    np.save(ytest_path, y_test)
    np.save(xtrain_path, X_train_scaled)
    np.save(ytrain_path, y_train)

    print(f"Saved: {scaler_path}")
    print(f"Saved: {encoder_path}")
    print(f"Saved: {xtrain_path}, {ytrain_path}")
    print(f"Saved: {xtest_path}, {ytest_path}\n")

    # --- Save preprocessing metadata (feeds into model_metadata.json later) ---
    metadata = {
        "dataset": "CICIoT2023 (pre-stratified, raqeeb24/ciciot-2023-stratified-dataset)",
        "raw_rows": int(before),
        "rows_after_cleaning": int(after),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "excluded_columns": present_excluded,
        "class_order": CLASS_ORDER,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scaler": "StandardScaler",
        "preprocessing_date": datetime.now().strftime("%Y-%m-%d"),
    }
    metadata_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()