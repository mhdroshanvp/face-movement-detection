"""
Trains a small MLP classifier on [yaw, pitch, roll, dx, dy] -> looking-at-screen
/ looking-away, using real collected data if it exists (data/collected/features.csv),
falling back to / topping up with the synthetic bootstrap dataset otherwise.

Saves:
  models/gaze_classifier.joblib   (for use from Python, see infer_demo.py)
  models/gaze_classifier.onnx     (for browser deployment via ONNX Runtime Web,
                                    or conversion to TFJS)

Prints accuracy / classification report / confusion matrix to stdout -- copy
these numbers into NOTES.md when real data has been folded in.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic_data import generate_synthetic_dataset, OUT_PATH as SYNTH_PATH  # noqa: E402
from pose_estimation import FEATURE_NAMES  # noqa: E402

COLLECTED_PATH = Path(__file__).resolve().parent.parent / "data" / "collected" / "features.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Below this many real rows, real data alone is too small/unbalanced to train
# on confidently, so it's blended with synthetic data instead of replacing it.
MIN_REAL_ROWS_TO_STANDALONE = 300


def load_dataset() -> tuple[pd.DataFrame, str]:
    if not SYNTH_PATH.exists():
        synth_df = generate_synthetic_dataset()
        SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(SYNTH_PATH, index=False)
    else:
        synth_df = pd.read_csv(SYNTH_PATH)

    if COLLECTED_PATH.exists():
        real_df = pd.read_csv(COLLECTED_PATH)
        real_df = real_df.dropna(subset=FEATURE_NAMES + ["label"])
        if len(real_df) >= MIN_REAL_ROWS_TO_STANDALONE:
            return real_df, "real-only"
        elif len(real_df) > 0:
            combined = pd.concat([synth_df, real_df], ignore_index=True)
            return combined, f"synthetic+real (blended, {len(real_df)} real rows)"

    return synth_df, "synthetic-only (no real data collected yet -- run capture_data.py)"


def main():
    df, source = load_dataset()
    print(f"Dataset source: {source}")
    print(f"Total rows: {len(df)}")
    print(df["label"].value_counts())

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)),
    ])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=pred):")
    labels = sorted(df["label"].unique())
    print(labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib_path = MODEL_DIR / "gaze_classifier.joblib"
    joblib.dump(clf, joblib_path)
    print(f"\nSaved sklearn pipeline to {joblib_path}")

    try:
        from skl2onnx import to_onnx
        onnx_model = to_onnx(clf, X_train[:1].astype(np.float32))
        onnx_path = MODEL_DIR / "gaze_classifier.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        print(f"Saved ONNX model to {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped/failed: {e}")


if __name__ == "__main__":
    main()
