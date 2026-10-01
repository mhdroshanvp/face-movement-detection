# Face Movement / Gaze Detection — Notes

## Approach

Hybrid geometric-features + trained-classifier design, not a raw end-to-end
CNN on pixels:

1. **MediaPipe FaceLandmarker** (`models/face_landmarker.task`, Google's
   official model bundle, 478 landmarks incl. iris) extracts, per frame
   (`src/pose_estimation.py`):
   - `yaw`, `pitch`, `roll` — head orientation, from the facial
     transformation matrix.
   - `dx`, `dy` — face-center offset from frame center (leaning off-frame).
   - `gaze_x`, `gaze_y` — iris position within the eye socket, i.e. **where
     the eyes are pointed independent of head orientation**. This is what
     catches "head facing the screen, eyes sneaking a look at notes/phone".
   - `ear` (eye-aspect-ratio) — classic blink/closed-eye feature (Soukupova
     & Cech 2016 formula).
   - `mar` (mouth-aspect-ratio) — open-mouth / talking feature.
   - `face_size` — landmark bounding-box width, a proxy for distance from
     the camera (too close / too far / leaned out of frame → "sit properly").
2. A **9-feature → 4-class MLP classifier** (scikit-learn,
   `hidden_layer_sizes=(16, 8)`) is trained on
   `[yaw, pitch, roll, dx, dy, gaze_x, gaze_y, ear, mar] ->
   looking-at-screen | looking-away | eyes-averted | talking`
   (`src/train_classifier.py`). This is the trained-model deliverable, as
   opposed to hardcoded angle thresholds.
3. `num_faces == 0`, `num_faces >= 2`, and `face_size` outside a plausible
   range are handled **deterministically** (`no-face-detected` /
   `multiple-faces` / `too-close` / `too-far`) rather than by the
   classifier — there's no ambiguity to learn there, it's just geometry.
4. `src/infer_demo.py` adds the debounce logic: a status must persist for
   `VIOLATION_THRESHOLD_SEC` (default 2.5s) before a warning fires, so a
   blink or a quick glance doesn't trigger anything.
5. `src/alerts.py` fires the actual warning: an on-screen text message, one
   per violation *type* (e.g. "Please keep your eyes on the screen." vs.
   "Please remain quiet and focus on your exam."), not a single generic
   warning. (A spoken/TTS version was built and tried, then dropped per
   product decision — text-only is what's wanted for now.)

## Why classifier classes vs. deterministic classes

Pose/gaze/mouth-open genuinely need a learned decision boundary — "is 15°
of yaw plus a bit of gaze offset suspicious or not" isn't a clean threshold,
and different people/cameras will shift what "normal" looks like, which is
exactly what a classifier trained on real data (once collected) can adapt
to. Face count and face size don't have that ambiguity: 0 faces is 0 faces,
regardless of person or camera — spending model capacity "learning" that
would be pointless, so those stay as simple threshold checks.

## Dataset

**Currently: synthetic bootstrap only** (`src/generate_synthetic_data.py`,
6000 rows, 1500/class across 4 classes). Feature values are sampled from
hand-picked distributions meant to resemble plausible geometry per class —
e.g. `eyes-averted` samples large `gaze_x`/`gaze_y` offsets while keeping
`yaw`/`pitch` in the "roughly forward" range, `talking` samples an elevated
`mar` with everything else normal. See the module docstring for the exact
distributions used per class.

This was a deliberate choice to get the full pipeline (extraction → training
→ ONNX export → live demo) working and testable immediately, since this
agent has no webcam access to collect real frames itself.

**This is a placeholder, not production data.** `src/capture_data.py` is
provided for you to run locally with your own webcam to collect real labeled
examples (press `a`/`l`/`e`/`t` to label live frames as looking-at-screen /
looking-away / eyes-averted / talking). `train_classifier.py` already knows
how to use it: if `data/collected/features.csv` has ≥300 rows it trains on
real data alone; with fewer rows it blends real + synthetic.
**Recommend collecting real data — ideally from a few different people,
lighting setups, and with/without glasses — before relying on this for
actual exam proctoring.** The `eyes-averted` class in particular is worth
extra real examples: it's the hardest one to fake convincingly with
synthetic geometry (real eye movement + head micro-motion interact in ways
a hand-picked distribution won't fully capture).

## Accuracy

On the synthetic dataset's held-out 20% test split (1200 rows, 4 classes):

```
Test accuracy: 0.984
                   precision  recall  f1-score
     eyes-averted       0.98    0.99      0.99
looking-at-screen       0.98    0.97      0.97
     looking-away       0.98    0.98      0.98
          talking       1.00    0.99      1.00
```

(Slightly lower than the first version's 0.992 — deliberately: v1's
"looking-at-screen" band was too tight and had zero overlap with the other
classes, which let it hit ~99% but also caused real false positives on
ordinary human noise — see "Real-usage fixes" below. The v2 numbers above
reflect a wider, more realistic attentive band with intentional overlap
near the boundaries, which is a harder — but more honest — classification
problem.)

**Caveat: this number is inflated and not a real-world accuracy estimate.**
The test set is drawn from the same synthetic distribution as the training
set, so this mainly confirms the classifier can learn the decision boundary
implied by the generation rules — it says nothing about performance on real
webcam footage, real lighting, or real human variability, and nothing about
whether `eyes-averted` vs `looking-at-screen` is actually reliably
separable from real iris-tracking noise (webcam iris tracking is
noticeably noisier than head-pose tracking). Re-run `train_classifier.py`
after collecting real data and update these numbers before treating this as
a real accuracy figure.

## Real-usage fixes (from live testing)

Live testing against a real face surfaced issues the synthetic-only
pipeline couldn't: these are documented here because they're the kind of
thing real data collection would have caught earlier, and are a preview of
what to expect more of before this is trustworthy for real exams.

- **False "talking" from a statically open mouth** (yawn, resting jaw) —
  fixed by requiring MAR to actually *vary* over ~1.6s before "talking"
  stands (`MAR_VARIANCE_WINDOW`/`MAR_VARIANCE_MIN` in `infer_demo.py`); a
  static open mouth now falls back to the next-best class.
- **Voice warnings silently failing/lagging after the first one** — was
  reusing one `pyttsx3` TTS engine instance across calls, a known-unreliable
  pattern on Windows; fixed by creating a fresh engine per utterance. Voice
  alerts were subsequently removed entirely per product decision (text-only
  warnings), so this is now moot, but noted in case voice is revisited later.
- **Continuous-run debounce could miss genuine sustained talking** — real
  talking has the mouth open and close rhythmically, so requiring *every*
  frame in the window to be non-attentive meant real talking could fail to
  trigger a warning at all. Replaced with a sliding-window majority vote
  (`STATUS_WINDOW`/`VIOLATION_FRACTION`/`RESET_FRACTION`).
- **False "looking-away"/"talking" from ordinary human noise** — breathing
  with a slightly open mouth, or a small involuntary head tilt, was being
  misclassified as a violation. Two contributing causes, both addressed:
  1. The synthetic "looking-at-screen" band was too tight (yaw/pitch std=7°,
     no blink or relaxed-mouth noise at all) relative to real human jitter.
     Widened it and added blink (~15%) and relaxed-mouth/breathing (~15%)
     noise into the synthetic attentive class, and pushed the other
     classes' onset ranges further out so there's a real margin instead of
     the bands nearly touching (`generate_synthetic_data.py`).
  2. Raw per-frame landmark estimates are themselves noisy (detection
     jitter, not just real movement). Pose/gaze/EAR features are now
     smoothed with a rolling mean (`SMOOTH_WINDOW` in `infer_demo.py`,
     `SMOOTHED_FEATURES` in `pose_estimation.py`) before being classified.
     `mar` is deliberately excluded from this smoothing since talking
     detection needs the raw signal's variance, not a blurred average.

These were all found and fixed using synthetic/simulated feature vectors,
without needing new real webcam data — but they're also a direct
illustration of why real data matters: every one of them is exactly the
kind of gap that shows up only when the model meets a real face, and there
will likely be more like it.

## Known Limitations

- **No real data yet** — see above. This is the single biggest gap before v1
  is trustworthy, and matters more now than before given the added
  eye-gaze class (gaze estimation from consumer webcams is inherently
  noisier than head pose).
- **Gaze/iris estimation is coarse** — `gaze_x`/`gaze_y` are derived from
  eye-socket-relative iris position on a standard webcam (no calibration
  step, no infrared, low resolution around the eye). This is fine for
  "eyes clearly averted" but will not reliably catch subtle glances, and
  may be noisy for people with smaller eye apertures, certain eyelid
  shapes, or partial occlusion from eyelashes/makeup — none of this has
  been validated against real users.
- **EAR/MAR are unvalidated for this population** — the "open eye" and
  "closed mouth" reference ranges in `generate_synthetic_data.py` are
  generic literature defaults, not calibrated to any real face. A real
  person's resting EAR/MAR could sit outside these assumptions.
- **Glasses / reflections**: not tested; likely to affect gaze/EAR more
  than head pose, since iris visibility is more sensitive to lens glare
  than the coarser face landmarks used for yaw/pitch.
- **Low light**: out of scope per the brief; expect degraded face and iris
  detection in dim rooms, not tested.
- **Side/extreme profiles**: a hard profile turn tends to produce
  `no-face-detected` rather than a confident `looking-away` — functionally
  fine here since both count as a violation, but it means yaw isn't really
  measured all the way to 90°.
- **`face_size` thresholds (`too-close`/`too-far`) are unvalidated
  guesses** (`FACE_SIZE_TOO_FAR = 0.12`, `FACE_SIZE_TOO_CLOSE = 0.75` in
  `infer_demo.py`) — picked without real measurements at typical laptop
  webcam distance. Expect to need tuning once you watch it against a real
  seated user.
- **Single frontal camera only** — no multi-camera or unusual placement
  handling, as scoped.
- **`num_faces >= 2` always flags `multiple-faces`** — someone briefly
  passing behind the student would trigger it. Acceptable for v1.
- **Warnings are text-only, no audio** — voice/TTS was built and tested,
  then removed per product decision (only written warnings wanted). The
  per-status `MESSAGES` map in `alerts.py` is unchanged and would still
  work as the source of truth if voice is added back later.

## Deployment Format Recommendation

**ONNX** (`models/gaze_classifier.onnx`, exported via `skl2onnx`). Reasoning:
- The classifier is a tiny scikit-learn pipeline (StandardScaler + MLP) —
  ONNX is a natural fit and already wired up in `train_classifier.py`.
- ONNX Runtime Web can run it client-side, matching the brief's "no extra
  server" browser option.
- If the team decides on the Python microservice route instead, the
  `.joblib` file (also saved) loads directly with no conversion needed.
- The MediaPipe FaceLandmarker side already ships a JS/WASM build
  (`@mediapipe/tasks-vision`) that reads the same `.task` file and supports
  iris landmarks, so the browser-only deployment path (landmarks + gaze +
  pose in JS, ONNX classifier in JS) is viable without a Python service in
  the loop at all — worth discussing before committing to the microservice
  architecture.
- Warnings are text-only (`alerts.py`'s `MESSAGES` map), which maps
  directly onto whatever the exam portal's existing violation-warning UI
  does for tab-switch/fullscreen-exit — no extra work needed there.

## Not Yet Decided (per brief)

- Browser (TFJS/ONNX Runtime Web) vs. Python microservice deployment.
- Wiring `report_violation()` in `src/infer_demo.py` to the real
  `POST /api/attempts/:attemptId/violations` endpoint (stubbed, prints
  instead of calling, since that endpoint doesn't exist yet).
- Whether `too-close`/`too-far`/`multiple-faces` should also be reported as
  `type: "look-away"` violations (current stub does this) or need their own
  violation types on the backend.
