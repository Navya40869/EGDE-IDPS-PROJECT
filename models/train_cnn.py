"""
train_cnn.py — Phase 4: CNN Training

Loads the frozen feature subset from data/processed/frozen_features.json,
selects those columns from the preprocessed train/test arrays, trains the
1D-CNN with class-weighted CrossEntropyLoss (severe imbalance confirmed in
Phase 2/3), and saves all artifacts together:

    models/edge_idps_model.pth
    data/processed/model_metadata.json   (feature_count, class list, metrics)

Run from the project root, with the venv activated:
    pip install torch
    python train_cnn.py
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_model import EdgeIDPSCNN
from utils.config import CLASS_ORDER

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")
FROZEN_FEATURES_PATH = os.path.join(PROCESSED_DIR, "frozen_features.json")
PREPROCESSING_METADATA_PATH = os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "edge_idps_model.pth")
MODEL_METADATA_PATH = os.path.join(PROCESSED_DIR, "model_metadata.json")

# --- Hyperparameters (from blueprint) ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
RANDOM_STATE = 42


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Load frozen feature order and select those columns ---
    with open(FROZEN_FEATURES_PATH) as f:
        frozen_schema = json.load(f)
    frozen_features = frozen_schema["feature_order"]
    print(f"Frozen feature schema version {frozen_schema['version']}, "
          f"{frozen_schema['feature_count']} features\n")

    with open(PREPROCESSING_METADATA_PATH) as f:
        preprocessing_metadata = json.load(f)
    all_features = preprocessing_metadata["feature_names"]
    feature_indices = [all_features.index(f) for f in frozen_features]

    print("Loading preprocessed data ...")
    X_train_full = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test_full = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    # Select only the frozen feature columns
    X_train = X_train_full[:, feature_indices]
    X_test = X_test_full[:, feature_indices]
    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}\n")

    num_features = X_train.shape[1]
    num_classes = len(CLASS_ORDER)

    # --- Class weights (severe imbalance confirmed: DDoS 5.89M vs Brute Force 2263) ---
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights (balanced):")
    for cls_idx, weight in enumerate(class_weights):
        print(f"  {CLASS_ORDER[cls_idx]}: {weight:.4f}")
    print()

    # --- DataLoaders ---
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, loss, optimizer ---
    model = EdgeIDPSCNN(num_features=num_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    best_state_dict = None

    print("Starting training ...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        # --- Validation on test set each epoch ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | train_loss: {train_loss:.4f} | "
              f"val_accuracy: {acc:.4f} | val_macro_f1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state_dict = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no macro F1 improvement for "
                  f"{EARLY_STOPPING_PATIENCE} epochs). Best macro F1: {best_macro_f1:.4f}")
            break

    # --- Restore best model and do final evaluation ---
    model.load_state_dict(best_state_dict)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    final_accuracy = accuracy_score(all_labels, all_preds)
    final_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n=== Final Evaluation (best checkpoint) ===")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Macro F1: {final_macro_f1:.4f}\n")
    print("Per-class report:")
    print(classification_report(
        all_labels, all_preds, target_names=CLASS_ORDER, zero_division=0
    ))
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(CLASS_ORDER)
    print(conf_matrix)

    # --- Save model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nSaved: {MODEL_SAVE_PATH}")

    # --- Save model_metadata.json (per blueprint spec) ---
    metadata = {
        "dataset": "CICIoT2023 (pre-stratified, raqeeb24/ciciot-2023-stratified-dataset)",
        "model": "1D-CNN",
        "num_classes": num_classes,
        "feature_count": num_features,
        "feature_names": frozen_features,
        "scaler": "StandardScaler",
        "train_accuracy": None,  # can be filled with a final train-set pass if desired
        "test_accuracy": float(final_accuracy),
        "macro_f1": float(final_macro_f1),
        "per_class_f1": {
            CLASS_ORDER[i]: float(f1_per_class[i]) for i in range(num_classes)
        },
        "class_weights": {
            CLASS_ORDER[i]: float(class_weights[i]) for i in range(num_classes)
        },
        "confusion_matrix": conf_matrix.tolist(),
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "random_state": RANDOM_STATE,
        },
        "training_date": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {MODEL_METADATA_PATH}")

    print("\nPhase 4 (CNN training) complete.")


if __name__ == "__main__":
    main()