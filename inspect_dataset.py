import sys
import pandas as pd

EXPECTED_8_CLASSES = {
    "Benign", "DoS", "DDoS", "Reconnaissance",
    "Spoofing", "Brute Force", "Web-based", "Mirai",
}


def main(path):
    print(f"Loading {path} ...\n")
    df = pd.read_csv(path)

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Columns:")
    print(df.columns.tolist(), "\n")

    label_col = "label" if "label" in df.columns else (
        "Label" if "Label" in df.columns else None
    )
    if label_col is None:
        print("!! Could not find a 'label' or 'Label' column. Inspect columns above manually.")
        return

    print(f"Detected label column: '{label_col}'\n")
    counts = df[label_col].value_counts()
    print("Class distribution (raw label values):")
    print(counts, "\n")

    print(f"Number of unique raw label values: {df[label_col].nunique()}")
    print("(If these are fine-grained, e.g. 'DDoS-UDP_Flood', you'll still need your")
    print(" utils/config.py mapping dict to collapse them into the 8 high-level classes.)\n")

    labels_lower = set(str(l).strip() for l in df[label_col].unique())
    matched = labels_lower & EXPECTED_8_CLASSES
    if matched:
        print(f"Labels already matching the 8-class taxonomy directly: {sorted(matched)}")
    else:
        print("Labels appear fine-grained (not pre-grouped into the 8 classes) — expected, "
              "you'll map them via utils/config.py.")

    print("\nClass balance check (min/max row count per unique label):")
    print(f"  min: {counts.min()}  max: {counts.max()}  ratio: {counts.max() / counts.min():.1f}x")
    if counts.max() / counts.min() > 20:
        print("  -> Significant imbalance remains even in this 'stratified' file — ")
        print("     you may still need class weighting during training (Phase 4).")

    null_counts = df.isnull().sum()
    nulls_present = null_counts[null_counts > 0]
    print("\nColumns with missing values:")
    print(nulls_present if not nulls_present.empty else "  None")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_dataset.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])