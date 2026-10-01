"""
Run this LOCALLY on a machine with a webcam to collect real labeled training
data. This agent has no webcam access, so this script is meant for you to run
yourself, then re-run train_classifier.py to fold the real data in.

Controls (while the window is focused):
  a  -> label the CURRENT frame "looking-at-screen"
  l  -> label the CURRENT frame "looking-away"
  e  -> label the CURRENT frame "eyes-averted"   (head forward, eyes elsewhere)
  t  -> label the CURRENT frame "talking"        (mouth clearly open)
  q  -> quit

Tips for a decent v1 dataset (a few hundred rows per class is plenty):
  - "looking-at-screen": read/type normally, head and eyes on the screen.
  - "looking-away": turn your head left/right, look up/down, lean toward the
    edge of frame -- vary it, don't just do one pose.
  - "eyes-averted": keep your HEAD still and forward, only move your EYES to
    glance left/right/down (e.g. as if reading notes beside the keyboard).
  - "talking": open your mouth as if speaking, head/eyes roughly normal.
  - Try a couple of lighting setups and, if relevant, with/without glasses.
  - Do NOT hold one key down for one static pose for 200 frames straight;
    a lightly varied set of ~150-300 examples per class beats a huge but
    repetitive one.

Each press appends one row to data/collected/features.csv.
"""

import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_estimation import PoseEstimator, FEATURE_NAMES  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "collected" / "features.csv"

LABEL_KEYS = {
    ord("a"): "looking-at-screen",
    ord("l"): "looking-away",
    ord("e"): "eyes-averted",
    ord("t"): "talking",
}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH)
        rows = df.to_dict("records")
    else:
        rows = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Exiting.")
        return

    estimator = PoseEstimator()
    print("Press a=looking-at-screen  l=looking-away  e=eyes-averted  t=talking  q=quit & save.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame, stopping.")
                break

            geometry = estimator.analyze(frame)
            feats = geometry.as_feature_vector()

            display = frame.copy()
            status = f"faces={geometry.num_faces}"
            if feats:
                status += f" yaw={feats[0]:.1f} pitch={feats[1]:.1f} gaze=({feats[5]:.2f},{feats[6]:.2f}) ear={feats[7]:.2f} mar={feats[8]:.2f}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"saved rows: {len(rows)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "a=at-screen l=away e=eyes-averted t=talking q=quit", (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.imshow("capture_data - label frames", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in LABEL_KEYS:
                if geometry.num_faces != 1 or feats is None:
                    print("Skipped: need exactly 1 face detected to label a pose sample.")
                    continue
                label = LABEL_KEYS[key]
                row = dict(zip(FEATURE_NAMES, feats))
                row["label"] = label
                rows.append(row)
                print(f"Logged '{label}' (total {len(rows)})")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    if rows:
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"Saved {len(rows)} rows to {OUT_PATH}")
    else:
        print("No rows captured, nothing saved.")


if __name__ == "__main__":
    main()
