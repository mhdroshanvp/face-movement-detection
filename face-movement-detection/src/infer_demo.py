"""
Deliverable: webcam frame in -> status out, exam-monitoring scenario.

Runs a live loop against your webcam at ~5 fps, printing one JSON status
line per check:

    {"status": "eyes-averted", "confidence": 0.87, "timestamp": "2026-08-03T13:05:11.482Z"}

status is one of:
  looking-at-screen | looking-away | eyes-averted | talking
  | no-face-detected | multiple-faces | too-close | too-far

The first four come from the trained classifier (yaw/pitch/roll/dx/dy/gaze/ear/mar
features, see pose_estimation.py + train_classifier.py). The last four are
deterministic checks (face count, face size) that don't need ML.

"talking" is additionally gated on mouth MOTION, not just mouth openness: a
single-frame open mouth (yawn, resting jaw, brief mouth-open) looks identical
to real talking in one frame, so the classifier's "talking" call only stands
if the mouth-aspect-ratio has actually been varying over the last ~1.5s
(MAR_VARIANCE_WINDOW). A mouth held statically open gets reclassified back
to whatever the pose/gaze features say instead of a false "talking" warning.

Debounce: a single bad frame (blink, brief glance) does NOT raise a warning.
This uses a SLIDING-WINDOW MAJORITY VOTE over the last STATUS_WINDOW checks
(~VIOLATION_THRESHOLD_SEC worth) rather than requiring every single frame in
that span to be non-attentive: real talking has the mouth open and close
repeatedly, so demanding an unbroken run of "bad" frames would mean genuine
sustained talking rarely trips the debounce at all. A warning fires once the
non-attentive fraction of the window crosses VIOLATION_FRACTION, and resets
once it drops back under RESET_FRACTION (hysteresis, avoids re-firing every
frame while borderline) -- then an on-screen text warning (src/alerts.py),
plus (optionally) a violation report sent to the backend via
report_violation().

Run: python src/infer_demo.py
Quit: press 'q' in the preview window, or Ctrl+C in the terminal.
"""

import json
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_estimation import PoseEstimator, FEATURE_NAMES, SMOOTHED_FEATURES  # noqa: E402
from alerts import Alerter  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gaze_classifier.joblib"

CHECK_INTERVAL_SEC = 0.2       # ~5 checks/sec, per the "few fps is enough" requirement
VIOLATION_THRESHOLD_SEC = 2.5  # sliding-window span used for the majority-vote debounce
STATUS_WINDOW = max(1, round(VIOLATION_THRESHOLD_SEC / CHECK_INTERVAL_SEC))
VIOLATION_FRACTION = 0.6  # >=60% of the window non-attentive -> fire a warning
RESET_FRACTION = 0.3      # <30% non-attentive -> allow the next warning to fire

# Face-size (landmark bbox width, normalized 0-1) thresholds for posture/distance
# checks. These are rough starting points, not calibrated on real users --
# tune them once you've watched a few real sessions (see NOTES.md).
FACE_SIZE_TOO_FAR = 0.12
FACE_SIZE_TOO_CLOSE = 0.75

# "talking" requires the mouth to actually be moving, not just open. Variance
# of MAR over this many recent checks (at ~5 checks/sec, 8 checks ~= 1.6s)
# must clear MAR_VARIANCE_MIN, otherwise a statically-open mouth is treated
# as non-talking. Both are unvalidated starting points -- tune with real data.
MAR_VARIANCE_WINDOW = 8
MAR_VARIANCE_MIN = 0.0008

# Pose/gaze/EAR features are smoothed (rolling mean) over this many recent
# checks before being classified, to damp single-frame landmark jitter and
# momentary noise (a blink, a tiny involuntary head twitch) rather than
# reacting to every raw frame. mar is excluded (see pose_estimation.py).
SMOOTH_WINDOW = 4

# Set to your Node/Express backend base URL to actually POST violations, e.g.
#   BACKEND_URL = "http://localhost:4000"
# Left as None so this demo never makes network calls on its own.
BACKEND_URL = None


def report_violation(attempt_id: str, status: str, message: str):
    """Mirrors the existing tab-switch/fullscreen-exit violation pattern:
    POST /api/attempts/:attemptId/violations  { type: "look-away", message }
    """
    payload = {"type": "look-away", "message": message, "detail": status}
    if BACKEND_URL is None:
        print(f"[would POST /api/attempts/{attempt_id}/violations] {payload}")
        return
    import requests
    url = f"{BACKEND_URL}/api/attempts/{attempt_id}/violations"
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        print(f"Failed to report violation: {e}")


def classify(clf, geometry, mar_history: deque, feature_history: dict) -> tuple[str, float]:
    if geometry.num_faces == 0:
        return "no-face-detected", 1.0
    if geometry.num_faces >= 2:
        return "multiple-faces", 1.0

    if geometry.face_size is not None:
        if geometry.face_size < FACE_SIZE_TOO_FAR:
            return "too-far", 1.0
        if geometry.face_size > FACE_SIZE_TOO_CLOSE:
            return "too-close", 1.0

    feats = geometry.as_feature_vector()
    if feats is None or any(f is None for f in feats):
        # landmarks present but iris/EAR/MAR unavailable (e.g. old model
        # bundle without iris refinement) -- fall back to no-face-detected
        # rather than feeding Nones into the classifier.
        return "no-face-detected", 0.0

    mar_history.append(geometry.mar)

    # smooth jitter-prone features (everything but mar) with a rolling mean
    # before handing them to the classifier
    smoothed = dict(zip(FEATURE_NAMES, feats))
    for name in SMOOTHED_FEATURES:
        feature_history[name].append(smoothed[name])
        smoothed[name] = sum(feature_history[name]) / len(feature_history[name])
    feats = [smoothed[name] for name in FEATURE_NAMES]

    proba = clf.predict_proba([feats])[0]
    classes = clf.classes_
    idx = proba.argmax()
    status, confidence = classes[idx], float(proba[idx])

    if status == "talking":
        not_enough_history = len(mar_history) < mar_history.maxlen
        mouth_not_moving = (not not_enough_history
                             and statistics.pvariance(mar_history) < MAR_VARIANCE_MIN)
        if not_enough_history or mouth_not_moving:
            # mouth is open but not moving (or we can't tell yet) -- fall back
            # to the next-best class instead of assuming "talking"
            runner_up = max(
                ((c, p) for c, p in zip(classes, proba) if c != "talking"),
                key=lambda cp: cp[1],
            )
            status, confidence = runner_up

    return status, confidence


def main(attempt_id: str = "demo-attempt"):
    if not MODEL_PATH.exists():
        print(f"No trained model at {MODEL_PATH}. Run train_classifier.py first.")
        return

    clf = joblib.load(MODEL_PATH)
    estimator = PoseEstimator()
    alerter = Alerter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        return

    status_window = deque(maxlen=STATUS_WINDOW)  # recent (status, is_bad) pairs
    violation_active = False
    last_message = ""
    mar_history = deque(maxlen=MAR_VARIANCE_WINDOW)
    feature_history = {name: deque(maxlen=SMOOTH_WINDOW) for name in SMOOTHED_FEATURES}

    print("Running. Press 'q' in the preview window to quit.\n")

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame, stopping.")
                break

            geometry = estimator.analyze(frame)
            status, confidence = classify(clf, geometry, mar_history, feature_history)
            now = datetime.now(timezone.utc)

            result = {
                "status": status,
                "confidence": round(confidence, 3),
                "timestamp": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            }
            print(json.dumps(result))

            # --- sliding-window majority-vote debounce ---
            is_bad = status != "looking-at-screen"
            status_window.append((status, is_bad))
            bad_fraction = sum(1 for _, b in status_window if b) / len(status_window)

            if bad_fraction >= VIOLATION_FRACTION and not violation_active:
                violation_active = True
                # representative status = most common non-attentive status in the window
                bad_statuses = [s for s, b in status_window if b]
                rep_status = max(set(bad_statuses), key=bad_statuses.count)
                last_message = alerter.warn(rep_status)
                report_violation(attempt_id, rep_status, last_message)
            elif bad_fraction < RESET_FRACTION and violation_active:
                violation_active = False
                last_message = ""

            display = frame.copy()
            color = (0, 200, 0) if status == "looking-at-screen" else (0, 0, 255)
            cv2.putText(display, f"{status} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if violation_active:
                cv2.putText(display, last_message, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("infer_demo", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed_loop = time.time() - loop_start
            time.sleep(max(0.0, CHECK_INTERVAL_SEC - elapsed_loop))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()


if __name__ == "__main__":
    main()
