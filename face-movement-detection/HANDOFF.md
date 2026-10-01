# Handoff: Face Movement Detection -> MERN Integration

For the developer wiring this into the exam portal. This folder is a
self-contained Python data-science deliverable: face/eye/mouth monitoring
that decides whether a student is attentive during an exam. It is **not**
wired into the MERN app yet — that's this handoff.

Full technical detail (model design, dataset, accuracy, limitations) is in
[NOTES.md](NOTES.md). This file is the short version: what you actually
need to integrate it.

## What it does

Given a webcam frame, it outputs one status:

```
looking-at-screen | looking-away | eyes-averted | talking
| no-face-detected | multiple-faces | too-close | too-far
```

`looking-at-screen` = fine, no warning. Everything else, if it persists
for ~2.5 continuous seconds (not a single bad frame), is a violation and
should show a warning message. Message text per status is defined in
`src/alerts.py`'s `MESSAGES` dict — reuse those strings or swap them for
whatever copy the portal already uses for tab-switch/fullscreen warnings.

**Known limitation, be aware of it:** the model is currently trained on
synthetic (hand-generated) data, not real faces. It runs and demos fine but
its real-world accuracy hasn't been measured yet. Real data collection is
planned but not done — see NOTES.md's "Dataset" section. Don't block
integration work on this; the model file can be swapped for a retrained one
later without changing the interface below.

## Two ways to integrate — pick one

### Option A: Browser-side (recommended, no Python service)

- `models/face_landmarker.task` — MediaPipe face landmark model. Has a
  JS/WASM build: `@mediapipe/tasks-vision` (npm package) reads this exact
  file format and gives you the same landmarks/pose this Python code uses.
- `models/gaze_classifier.onnx` — the trained classifier, run client-side
  via `onnxruntime-web` (npm package).
- Feature order the ONNX model expects (must match exactly):
  `[yaw, pitch, roll, dx, dy, gaze_x, gaze_y, ear, mar]` — see
  `src/pose_estimation.py` for how each is computed from landmarks (the
  same math needs porting to JS; it's plain arithmetic on landmark
  coordinates, no ML involved in that step).
- This keeps webcam frames entirely on the student's machine — nothing
  streamed to a server, which is probably preferable for an exam portal
  anyway.

### Option B: Python microservice

- Run `src/infer_demo.py`'s logic as a small service (Flask/FastAPI) the
  React frontend streams frames to, or that pulls frames from a shared
  location.
- `src/infer_demo.py` already has a `report_violation(attempt_id, status,
  message)` function stubbed out to call your endpoint — just point
  `BACKEND_URL` at it and it's live.
- Needs the Python venv running (`requirements.txt`) as a long-lived process.

**Whichever you pick**, the debounce logic (don't fire on one bad frame,
require sustained non-attentive status) lives in `src/infer_demo.py` and
should be ported alongside — firing a warning on every single frame would
be extremely noisy.

## Backend contract (either option)

Matches the existing tab-switch/fullscreen-exit pattern:

```
POST /api/attempts/:attemptId/violations
{ "type": "look-away", "message": "Please look at the screen.", "detail": "looking-away" }
```

`type` is always `"look-away"` for anything this system detects (all 7
non-attentive statuses currently map to it); `detail` carries which
specific status triggered it, in case you want finer-grained handling or
reporting later. This isn't final — if you'd rather split these into
separate violation types (e.g. `"no-face"` vs `"look-away"` vs
`"multiple-people"`), that's a reasonable change, just wasn't decided yet.

## Files that matter to you

```
models/
  face_landmarker.task     MediaPipe model -- for Option A (JS)
  gaze_classifier.onnx     trained classifier -- for Option A (JS)
  gaze_classifier.joblib   same classifier -- for Option B (Python)
src/
  pose_estimation.py       feature extraction logic -- reference for porting to JS (Option A)
                            or use directly (Option B)
  infer_demo.py            reference implementation: debounce logic, status
                            mapping, report_violation() stub -- read this
                            first regardless of which option you pick
  alerts.py                warning message text per status
```

Everything else (`train_classifier.py`, `capture_data.py`,
`generate_synthetic_data.py`, `venv/`, `data/`) is model-training tooling,
not needed for integration — ignore unless you're retraining the model.
