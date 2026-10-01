"""
Bootstraps a synthetic labeled dataset of
[yaw, pitch, roll, dx, dy, gaze_x, gaze_y, ear, mar] -> label
so the classifier pipeline works end-to-end before any real webcam data has
been collected.

Classes:
  looking-at-screen  head forward, eyes centered -- INCLUDES natural noise:
                     occasional blinks (low EAR) and a slightly open/relaxed
                     mouth (mild MAR, e.g. breathing), both of which are
                     normal and should NOT be confused with talking/averted.
  looking-away       head turned/tilted away, or leaning off-frame
  eyes-averted       head roughly forward, but eyes/iris looking elsewhere
                     (sneaking a glance at notes/phone without turning the head)
  talking            mouth open AND clearly more open than the "relaxed
                     mouth" noise band above

("no-face-detected" / "multiple-faces" / "too-close" / "too-far" are NOT part
of this dataset -- those are handled deterministically from num_faces /
face_size in infer_demo.py, not learned.)

v2 change: earlier version's "looking-at-screen" band was too tight (yaw/pitch
std=7 deg, no blink/breathing noise at all), so ordinary human head jitter,
blinking, or breathing with a slightly open mouth would fall outside it and
get misclassified as looking-away/talking. Widened the attentive band and
pushed the other classes' onset further out so there's a real margin between
"normal human noise" and "actual violation", instead of them nearly touching.

This is NOT a substitute for real data: it's sampled from a hand-picked
distribution of "plausible" geometry per class. Treat the resulting model as
a placeholder baseline. Once data/collected/ has a few hundred real labeled
rows (see capture_data.py), train_classifier.py will automatically prefer
the real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "synthetic_features.csv"

COLUMNS = ["yaw", "pitch", "roll", "dx", "dy", "gaze_x", "gaze_y", "ear", "mar", "label"]

# Typical EAR for open eyes ~0.25-0.35, MAR for closed mouth ~0.02-0.15 (rough,
# consistent with common EAR/MAR literature defaults; not calibrated to any
# specific person).
EAR_OPEN = (0.26, 0.05)   # mean, std
EAR_BLINK = (0.10, 0.03)  # mean, std -- eyes closed/blinking, still attentive
MAR_CLOSED = (0.06, 0.03)
MAR_RELAXED = (0.14, 0.04)  # mouth slightly open: breathing, resting jaw -- NOT talking
MAR_OPEN = (0.40, 0.10)     # clearly open -- talking


def _base_attentive(n, rng):
    feats = dict(
        yaw=rng.normal(0, 12, n),
        pitch=rng.normal(0, 12, n),
        roll=rng.normal(0, 8, n),
        dx=rng.normal(0, 0.07, n),
        dy=rng.normal(0, 0.07, n),
        gaze_x=rng.normal(0, 0.09, n),
        gaze_y=rng.normal(0, 0.09, n),
        ear=np.clip(rng.normal(*EAR_OPEN, n), 0.05, 0.5),
        mar=np.clip(rng.normal(*MAR_CLOSED, n), 0.0, 0.5),
    )
    # ~15% blinking/squinting at the moment of capture, still attentive
    blink_mask = rng.random(n) < 0.15
    feats["ear"] = np.where(blink_mask, np.clip(rng.normal(*EAR_BLINK, n), 0.02, 0.5), feats["ear"])
    # ~15% mouth slightly open (breathing, relaxed jaw), still attentive/not talking
    relaxed_mask = rng.random(n) < 0.15
    feats["mar"] = np.where(relaxed_mask, np.clip(rng.normal(*MAR_RELAXED, n), 0.0, 0.5), feats["mar"])
    return feats


def generate_synthetic_dataset(n_per_class: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # --- looking-at-screen ---
    n = n_per_class
    feats = _base_attentive(n, rng)
    for i in range(n):
        rows.append([feats[c][i] for c in COLUMNS[:-1]] + ["looking-at-screen"])

    # --- looking-away: head turned/tilted, or leaning off-frame ---
    n_sub = n_per_class // 3
    n_c = n_per_class - 2 * n_sub

    sign = rng.choice([-1, 1], n_sub)
    a = _base_attentive(n_sub, rng)
    a["yaw"] = sign * rng.uniform(32, 75, n_sub)
    a["pitch"] = rng.normal(0, 15, n_sub)
    a["dx"] = rng.normal(0, 0.08, n_sub)
    a["dy"] = rng.normal(0, 0.08, n_sub)

    sign = rng.choice([-1, 1], n_sub)
    b = _base_attentive(n_sub, rng)
    b["pitch"] = sign * rng.uniform(30, 65, n_sub)
    b["yaw"] = rng.normal(0, 15, n_sub)
    b["dx"] = rng.normal(0, 0.08, n_sub)
    b["dy"] = rng.normal(0, 0.08, n_sub)

    sign_x = rng.choice([-1, 1], n_c)
    sign_y = rng.choice([-1, 1], n_c)
    c = _base_attentive(n_c, rng)
    c["dx"] = sign_x * rng.uniform(0.28, 0.48, n_c)
    c["dy"] = sign_y * rng.uniform(0.28, 0.48, n_c)
    c["yaw"] = rng.normal(0, 20, n_c)
    c["pitch"] = rng.normal(0, 20, n_c)

    for block, n_block in [(a, n_sub), (b, n_sub), (c, n_c)]:
        for i in range(n_block):
            rows.append([block[col][i] for col in COLUMNS[:-1]] + ["looking-away"])

    # --- eyes-averted: head stays roughly forward, but iris offset is large ---
    n = n_per_class
    feats = _base_attentive(n, rng)
    sign_x = rng.choice([-1, 1], n)
    sign_y = rng.choice([-1, 1], n)
    # mix of horizontal glance, vertical glance, and combined
    mode = rng.integers(0, 3, n)
    gaze_x = np.where(mode != 1, sign_x * rng.uniform(0.28, 0.50, n), rng.normal(0, 0.07, n))
    gaze_y = np.where(mode != 0, sign_y * rng.uniform(0.28, 0.50, n), rng.normal(0, 0.07, n))
    feats["gaze_x"] = gaze_x
    feats["gaze_y"] = gaze_y
    # head can drift a little while eyes do the work, but stays well under the
    # looking-away threshold
    feats["yaw"] = rng.normal(0, 14, n)
    feats["pitch"] = rng.normal(0, 14, n)
    for i in range(n):
        rows.append([feats[c][i] for c in COLUMNS[:-1]] + ["eyes-averted"])

    # --- talking: mouth clearly open (beyond the "relaxed/breathing" band), pose/gaze roughly normal ---
    n = n_per_class
    feats = _base_attentive(n, rng)
    feats["mar"] = np.clip(rng.normal(*MAR_OPEN, n), 0.22, 0.8)
    feats["yaw"] = rng.normal(0, 14, n)
    feats["pitch"] = rng.normal(0, 14, n)
    for i in range(n):
        rows.append([feats[c][i] for c in COLUMNS[:-1]] + ["talking"])

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} synthetic rows to {OUT_PATH}")
    print(df["label"].value_counts())
