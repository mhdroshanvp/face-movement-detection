# Face Movement & Gaze Detection

> Computer vision based attention and malpractice monitoring module for an online examination system.

A real-time face, eye-gaze, and mouth-movement monitoring system designed for the **Linux Basics Online Examination Portal**.

The system analyzes webcam frames and determines whether an exam candidate is:

- Looking at the screen
- Looking away
- Looking elsewhere with their eyes
- Talking
- Not visible in the camera
- Multiple people visible
- Sitting too close
- Sitting too far

The system uses **MediaPipe Face Landmarker** for facial landmark extraction and a **Scikit-learn MLP classifier** for attention-state classification.

The trained classifier is also exported to **ONNX**, making it possible to run the model directly inside a browser using **ONNX Runtime Web**.

---

## Features

### Face Monitoring

Detects whether the candidate's face is present in the webcam frame.

### Head Pose Detection

Tracks:

- Yaw
- Pitch
- Roll

to determine whether the candidate is turning or tilting their head away from the screen.

### Eye Gaze Detection

Tracks iris position to detect cases where:

> The candidate's head is facing the screen but their eyes are looking somewhere else.

### Eye Aspect Ratio

EAR is used as a feature for detecting eye openness and blinking.

### Mouth Aspect Ratio

MAR is used to detect mouth opening.

Talking detection additionally uses temporal mouth movement rather than relying only on a single open-mouth frame.

### Multiple Face Detection

If two or more faces are detected, the system reports:

`multiple-faces`

### Camera Distance Detection

The system estimates face size in the frame to identify:

- `too-close`
- `too-far`

### Temporal Violation Detection

The system does not immediately report a violation from a single bad frame.

Instead, it uses a sliding window and temporal smoothing to reduce false positives caused by:

- Blinking
- Small head movements
- Brief glances
- Landmark noise

The current demo uses approximately **2.5 seconds** of observation before triggering a warning.

---

# System Architecture

```text
                    Webcam
                       │
                       ▼
             MediaPipe FaceLandmarker
                       │
                       ▼
              Facial Landmarks
                       │
                       ▼
             Feature Extraction
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
   Head Pose        Eye Gaze        Mouth/Eye
 yaw/pitch/roll    iris position      EAR/MAR
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                 9 Features
                       │
                       ▼
              MLP Classifier
                       │
                       ▼
                  Status
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    Looking at screen          Non-attentive
          │                         │
          │                  Temporal filtering
          │                         │
          │                         ▼
          │                    Violation
          │                         │
          │                         ▼
          │                  Warning / Backend
          │
          ▼
        Normal