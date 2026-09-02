"""
utils/config.py (fixed)

Label mapping dictionary for CICIoT2023 -> 8 high-level classes.

FIX (see project log): sklearn's LabelEncoder.fit() always sorts classes
alphabetically internally, regardless of the order of the list you pass it.
The original CLASS_ORDER below was written in a "logical" order (Benign, DoS,
DDoS, Reconnaissance, Spoofing, Brute Force, Web-based, Mirai), but training
actually encoded classes in alphabetical order. This caused every downstream
report (classification_report, confusion matrix, model_metadata.json) to print
the WRONG class name next to each set of stats, even though training itself
was internally consistent and correct.

Fix: CLASS_ORDER is now explicitly the alphabetically-sorted order, matching
exactly what LabelEncoder.fit() actually produces. This makes CLASS_ORDER safe
to use anywhere a class name needs to be looked up from an integer label
(risk engine, dashboard, inference.py) without re-deriving it from
label_encoder.classes_ every time.

If you ever retrain and get a NEW label_encoder.pkl, always verify:
    from utils.config import verify_class_order
    verify_class_order()
"""

LABEL_MAP = {
    # Benign
    "BENIGN": "Benign",

    # DDoS (distributed — many source IPs)
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",

    # DoS (single-source)
    "DOS-UDP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",

    # Mirai botnet
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",

    # Reconnaissance
    "RECON-PINGSWEEP": "Reconnaissance",
    "RECON-OSSCAN": "Reconnaissance",
    "RECON-PORTSCAN": "Reconnaissance",
    "RECON-HOSTDISCOVERY": "Reconnaissance",
    "VULNERABILITYSCAN": "Reconnaissance",

    # Spoofing
    "DNS_SPOOFING": "Spoofing",
    "MITM-ARPSPOOFING": "Spoofing",

    # Web-based
    "BROWSERHIJACKING": "Web-based",
    "BACKDOOR_MALWARE": "Web-based",
    "XSS": "Web-based",
    "UPLOADING_ATTACK": "Web-based",
    "SQLINJECTION": "Web-based",
    "COMMANDINJECTION": "Web-based",

    # Brute Force
    "DICTIONARYBRUTEFORCE": "Brute Force",
}

# FIXED — this is now the TRUE alphabetical order sklearn's LabelEncoder
# actually produces (verified against the trained label_encoder.pkl's
# .classes_ attribute). Integer label i corresponds to CLASS_ORDER[i].
CLASS_ORDER = [
    "Benign",          # 0
    "Brute Force",     # 1
    "DDoS",            # 2
    "DoS",             # 3
    "Mirai",           # 4
    "Reconnaissance",  # 5
    "Spoofing",        # 6
    "Web-based",       # 7
]


def map_labels(df, label_col="Label"):
    """
    Apply LABEL_MAP to a dataframe's label column, raising if any value is
    unmapped rather than silently producing NaNs.
    """
    unmapped = set(df[label_col].unique()) - set(LABEL_MAP.keys())
    if unmapped:
        raise ValueError(
            f"Unmapped labels found: {sorted(unmapped)}. "
            "Add them to LABEL_MAP in utils/config.py before proceeding."
        )
    df = df.copy()
    df["class"] = df[label_col].map(LABEL_MAP)
    return df


def verify_class_order(label_encoder_path="data/processed/label_encoder.pkl"):
    """
    Call this after any retrain to confirm CLASS_ORDER still matches the
    actual encoder's alphabetical order. Raises if they've drifted apart.
    """
    import joblib
    le = joblib.load(label_encoder_path)
    actual_order = [str(c) for c in le.classes_]
    if actual_order != CLASS_ORDER:
        raise ValueError(
            f"CLASS_ORDER is out of sync with label_encoder.pkl!\n"
            f"CLASS_ORDER:   {CLASS_ORDER}\n"
            f"encoder order: {actual_order}\n"
            "Update CLASS_ORDER in utils/config.py to match encoder order."
        )
    print("CLASS_ORDER verified — matches label_encoder.pkl exactly.")