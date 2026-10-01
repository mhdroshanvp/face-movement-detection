"""
Shared face-geometry extraction: wraps MediaPipe FaceLandmarker to turn a BGR
webcam frame into a feature vector that the downstream classifier is trained
on:

  yaw, pitch, roll   head orientation (from the facial transformation matrix)
  dx, dy             face-center offset from frame center (off-frame / leaning)
  gaze_x, gaze_y     iris position within the eye socket (eye movement,
                     independent of head pose -- catches "head straight but
                     eyes elsewhere")
  ear                eye-aspect-ratio, average of both eyes (closed/squinting
                     eyes; classic blink-detection feature)
  mar                mouth-aspect-ratio (open mouth / talking)
  face_size          landmark bounding-box width, normalized -- proxy for
                     distance from camera (too close / too far -> "sit
                     properly")

This module does NOT decide "suspicious" itself beyond the trivial cases (0
or 2+ faces, or face_size out of range) -- pose/gaze/mouth classification is
the trained classifier's job (train_classifier.py / infer_demo.py).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

FEATURE_NAMES = ["yaw", "pitch", "roll", "dx", "dy", "gaze_x", "gaze_y", "ear", "mar"]

# Features smoothed (rolling mean) before being fed to the classifier at
# inference time, to reduce single-frame landmark jitter and transient noise
# (a blink, a momentary detection wobble) from being mistaken for a real
# sustained pose/gaze change. "mar" is deliberately excluded: talking
# detection needs the RAW per-frame mouth openness plus its variance over
# time (see infer_demo.py), which a smoothed/averaged value would blur out.
SMOOTHED_FEATURES = ["yaw", "pitch", "roll", "dx", "dy", "gaze_x", "gaze_y", "ear"]

# Standard MediaPipe 478-point face mesh landmark indices (well-documented,
# widely used for EAR/MAR/iris-gaze -- see mediapipe face mesh landmark map).
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # outer, top1, top2, inner, bot2, bot1
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
LEFT_EYE_H_CORNERS = (33, 133)    # outer, inner
RIGHT_EYE_H_CORNERS = (263, 362)  # outer, inner
LEFT_EYE_V_LIDS = (159, 145)      # top, bottom (for vertical gaze normalization)
RIGHT_EYE_V_LIDS = (386, 374)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
MOUTH_V = (13, 14)     # inner upper lip, inner lower lip
MOUTH_H = (78, 308)    # left corner, right corner
MIN_LANDMARKS_FOR_IRIS = 478  # iris points only present in the attention-mesh model


@dataclass
class FaceGeometry:
    num_faces: int
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    dx: Optional[float] = None       # face-center x offset from image center, normalized [-0.5, 0.5]
    dy: Optional[float] = None       # face-center y offset from image center, normalized [-0.5, 0.5]
    gaze_x: Optional[float] = None   # avg iris horizontal offset within eye socket, ~[-0.5, 0.5]
    gaze_y: Optional[float] = None   # avg iris vertical offset within eye socket
    ear: Optional[float] = None      # eye-aspect-ratio, lower = more closed
    mar: Optional[float] = None      # mouth-aspect-ratio, higher = more open
    face_size: Optional[float] = None  # landmark bbox width, normalized [0, 1]

    def as_feature_vector(self):
        """Returns feature vector matching FEATURE_NAMES, or None if no primary face."""
        if self.num_faces == 0 or self.yaw is None:
            return None
        return [self.yaw, self.pitch, self.roll, self.dx, self.dy,
                self.gaze_x, self.gaze_y, self.ear, self.mar]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rotation_matrix_to_euler(matrix4x4) -> tuple:
    """Decomposes MediaPipe's 4x4 facial transformation matrix into yaw/pitch/roll degrees.

    Note: this is MediaPipe's canonical-face-model -> camera-space rotation, not a
    metrologically calibrated head-pose angle. It's monotonic with real head rotation
    and consistent frame-to-frame, which is all the classifier needs.
    """
    R = np.asarray(matrix4x4)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
    return float(yaw), float(pitch), float(roll)


def _eye_aspect_ratio(pts) -> float:
    """Classic EAR (Soukupova & Cech 2016): (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _iris_gaze_offset(landmarks):
    """Average, over both eyes, of the iris center's position within the eye
    socket -- normalized so 0 = centered, negative/positive = toward one side.
    This is independent of head yaw/pitch: a head-on face with eyes glancing
    sideways will show a nonzero gaze offset here even though yaw/pitch stay near 0.
    """
    def offset_for_eye(iris_idx, h_corners, v_lids):
        iris = landmarks[iris_idx]
        outer = landmarks[h_corners[0]]
        inner = landmarks[h_corners[1]]
        top = landmarks[v_lids[0]]
        bottom = landmarks[v_lids[1]]

        eye_w = abs(inner.x - outer.x)
        eye_h = abs(bottom.y - top.y)
        x_min, x_max = min(outer.x, inner.x), max(outer.x, inner.x)
        y_min, y_max = min(top.y, bottom.y), max(top.y, bottom.y)

        gx = (iris.x - (x_min + x_max) / 2) / eye_w if eye_w > 1e-6 else 0.0
        gy = (iris.y - (y_min + y_max) / 2) / eye_h if eye_h > 1e-6 else 0.0
        return gx, gy

    lx, ly = offset_for_eye(LEFT_IRIS_CENTER, LEFT_EYE_H_CORNERS, LEFT_EYE_V_LIDS)
    rx, ry = offset_for_eye(RIGHT_IRIS_CENTER, RIGHT_EYE_H_CORNERS, RIGHT_EYE_V_LIDS)
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def _mouth_aspect_ratio(landmarks) -> float:
    top, bottom = landmarks[MOUTH_V[0]], landmarks[MOUTH_V[1]]
    left, right = landmarks[MOUTH_H[0]], landmarks[MOUTH_H[1]]
    horizontal = _dist((left.x, left.y), (right.x, right.y))
    if horizontal < 1e-6:
        return 0.0
    return _dist((top.x, top.y), (bottom.x, bottom.y)) / horizontal


class PoseEstimator:
    def __init__(self, model_path: Path = MODEL_PATH, max_faces: int = 2):
        base_options = BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def analyze(self, frame_bgr: np.ndarray) -> FaceGeometry:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        num_faces = len(result.face_landmarks)
        if num_faces == 0:
            return FaceGeometry(num_faces=0)

        # Primary face = first detected. If num_faces > 1 the caller treats the
        # frame as a violation regardless of this face's pose.
        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        dx, dy = cx - 0.5, cy - 0.5
        face_size = max(xs) - min(xs)

        matrix = result.facial_transformation_matrixes[0]
        yaw, pitch, roll = _rotation_matrix_to_euler(matrix)

        gaze_x = gaze_y = ear = mar = None
        if len(landmarks) >= MIN_LANDMARKS_FOR_IRIS:
            left_pts = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_EAR]
            right_pts = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_EAR]
            ear = (_eye_aspect_ratio(left_pts) + _eye_aspect_ratio(right_pts)) / 2.0
            gaze_x, gaze_y = _iris_gaze_offset(landmarks)
            mar = _mouth_aspect_ratio(landmarks)

        return FaceGeometry(
            num_faces=num_faces, yaw=yaw, pitch=pitch, roll=roll, dx=dx, dy=dy,
            gaze_x=gaze_x, gaze_y=gaze_y, ear=ear, mar=mar, face_size=face_size,
        )

    def close(self):
        self.detector.close()
