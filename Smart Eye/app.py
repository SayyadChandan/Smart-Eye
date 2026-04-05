import base64
import csv
import io
import math
import os
import random
import string
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.core import image as mp_image_module
from zoneinfo import ZoneInfo
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    Response,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash


###############################################################################
# Flask Application Setup
###############################################################################

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (BASE_DIR / "smart_eye.db").resolve()
# Backwards compatibility: keep using existing engagement.db if present.
LEGACY_DB_PATH = (BASE_DIR / "engagement.db").resolve()

_env_db_path = os.getenv("TE_DB_PATH")
if _env_db_path:
    DB_PATH = Path(_env_db_path)
    if not DB_PATH.is_absolute():
        DB_PATH = (BASE_DIR / DB_PATH).resolve()
else:
    DB_PATH = LEGACY_DB_PATH if LEGACY_DB_PATH.exists() else DEFAULT_DB_PATH

app = Flask(__name__)

# IMPORTANT: In production use a strong, secret key and keep it outside source
# control. For a local MCA project this is sufficient.
app.secret_key = os.getenv("TE_SECRET_KEY", "dev_only_change_me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("TE_SESSION_COOKIE_SECURE", "0") == "1"

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


def _utc_naive() -> datetime:
    """Current UTC time as naive datetime (matches existing SQLite DateTime columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_yyyy_mm_dd(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _admin_sessions_filtered_query(
    meet_app: str | None,
    meet_title: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    meet_time_range: str | None = None,
):
    q = SessionModel.query
    if meet_app:
        q = q.filter(SessionModel.meet_app == meet_app)
    if meet_title:
        like = f"%{meet_title.strip()}%"
        q = q.filter(SessionModel.meet_title.ilike(like))
    if meet_time_range:
        like_time = f"%{meet_time_range.strip()}%"
        q = q.filter(SessionModel.meet_time_range.ilike(like_time))
    if start_date:
        q = q.filter(SessionModel.login_time >= start_date)
    if end_date:
        q = q.filter(SessionModel.login_time < (end_date + timedelta(days=1)))
    return q


def _compute_session_stats(session_id: int):
    """Compute basic engagement stats for a single session."""

    sess = db.session.get(SessionModel, session_id)
    if not sess:
        return {
            "duration_seconds": 0,
            "engaged_seconds": 0,
            "disengaged_seconds": 0,
            "disengage_events": 0,
        }

    start = sess.login_time
    end = sess.logout_time or _utc_naive()
    if not start or end <= start:
        duration_seconds = 0
    else:
        # Keep as float (avoid truncating to 0 for the first seconds).
        duration_seconds = (end - start).total_seconds()

    events = (
        EngagementEvent.query.filter(EngagementEvent.session_id == session_id)
        .order_by(EngagementEvent.timestamp.asc())
        .all()
    )

    engaged_seconds = 0.0
    disengaged_seconds = 0.0
    disengage_events = 0

    if not events:
        return {
            "duration_seconds": duration_seconds,
            "engaged_seconds": 0,
            "disengaged_seconds": 0,
            "disengage_events": 0,
            "face_detection_rate": 0.0,
        }

    def _state_bucket(ev: EngagementEvent) -> str:
        """Map stored DB state into one of: engaged | disengaged | unknown | other."""
        state = getattr(ev, "engagement_state", None)
        if state is None:
            state = getattr(ev, "state", None)
        if state is None:
            # Backward compatibility with older code paths
            return "engaged" if bool(getattr(ev, "is_engaged", False)) else "other"

        state_s = str(state).lower()
        if state_s == "engaged":
            return "engaged"
        if state_s in {"disengaged", "not engaged", "not_engaged", "not_engaged"}:
            return "disengaged"
        if state_s in {"unknown", "calibrating"}:
            return "unknown"
        if state_s in {"no_face", "no face detected", "no-face"}:
            return "unknown"
        return "other"

    engaged_count = 0
    disengaged_count = 0
    buckets: list[str] = []

    for ev in events:
        bucket = _state_bucket(ev)
        buckets.append(bucket)
        if bucket == "engaged":
            engaged_count += 1
        elif bucket == "disengaged":
            disengaged_count += 1

    # Use expected capture interval so we don't get stuck with 0 seconds due
    # to timestamp granularity / truncation.
    engaged_seconds = engaged_count * SAMPLING_INTERVAL_SECONDS
    disengaged_seconds = disengaged_count * SAMPLING_INTERVAL_SECONDS

    # Disengagement events: count transitions engaged -> disengaged.
    for prev, curr in zip(buckets[:-1], buckets[1:]):
        if prev == "engaged" and curr == "disengaged":
            disengage_events += 1

    # Face detection rate (fairness metric)
    total_frames = len(events)
    no_face_frames = sum(
        1
        for ev in events
        if str(getattr(ev, "engagement_state", "")).lower()
        in {"no_face", "no face detected", "no-face"}
    )
    frames_with_face = total_frames - no_face_frames
    face_detection_rate = (
        (frames_with_face / total_frames) * 100.0 if total_frames > 0 else 0.0
    )

    if duration_seconds > 0:
        engaged_seconds = max(0.0, min(engaged_seconds, duration_seconds))
        disengaged_seconds = max(0.0, min(disengaged_seconds, duration_seconds))

    return {
        "duration_seconds": float(duration_seconds),
        "engaged_seconds": int(round(engaged_seconds)),
        "disengaged_seconds": int(round(disengaged_seconds)),
        "disengage_events": disengage_events,
        "face_detection_rate": round(float(face_detection_rate), 2),
    }


@app.template_filter("format_meet_time_range")
def format_meet_time_range(value):
    """Reformat meet_time_range string from YYYY-MM-DD to DD-MM-YYYY if needed."""
    if not value or not isinstance(value, str):
        return value
    
    # Split by " | "
    parts = value.split(" | ", 1)
    if len(parts) != 2:
        return value
    
    date_part, time_part = parts
    
    # If it's already DD-MM-YYYY, leave it
    try:
        datetime.strptime(date_part, "%d-%m-%Y")
        return value
    except ValueError:
        pass
        
    # If it's YYYY-MM-DD, convert it
    try:
        d_obj = datetime.strptime(date_part, "%Y-%m-%d")
        new_date = d_obj.strftime("%d-%m-%Y")
        return f"{new_date} | {time_part}"
    except ValueError:
        return value


@app.template_filter("ist")
def format_ist(dt, fmt="%d-%m-%Y %H:%M:%S"):
    """Format a datetime (stored as UTC naive/aware) in IST for display.

    - DB values are created as naive UTC via _utc_naive().
    - We interpret naive datetimes as UTC, then convert to Asia/Kolkata.
    """
    if dt is None:
        return "-"

    try:
        # Treat naive datetimes as UTC
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        dt_ist = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        return dt_ist.strftime(fmt)
    except Exception:
        # Fallback: fixed offset UTC+5:30
        try:
            from datetime import timezone

            ist = timezone(timedelta(hours=5, minutes=30))
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(ist).strftime(fmt)
        except Exception:
            return str(dt)


###############################################################################
# Database Models
###############################################################################


class User(db.Model):
    """User model storing login details and role ('admin' or 'student')."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)

    sessions = db.relationship("SessionModel", back_populates="student")
    created_teaching_sessions = db.relationship("TeachingSession", backref="creator", lazy=True)


class TeachingSession(db.Model):
    """Represents a class/meet session created by an admin."""

    __tablename__ = "teaching_sessions"

    id = db.Column(db.Integer, primary_key=True)
    session_code = db.Column(db.String(20), unique=True, nullable=False)
    meet_app = db.Column(db.String(50), nullable=False)
    meet_title = db.Column(db.String(200), nullable=False)
    date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    created_by_admin_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=_utc_naive)

    # Relationship to student sessions
    student_sessions = db.relationship("SessionModel", back_populates="teaching_session", lazy=True)


class SessionModel(db.Model):
    """Represents one login session for a student."""

    __tablename__ = "sessions"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    
    # Link to the admin-created teaching session
    teaching_session_id = db.Column(db.Integer, db.ForeignKey("teaching_sessions.id"), nullable=True)
    
    login_time = db.Column(db.DateTime, nullable=False)
    logout_time = db.Column(db.DateTime, nullable=True)

    # Meet/class metadata (filled when the student starts a session)
    # These can now be redundant if teaching_session_id is present, 
    # but kept for backward compatibility or fallback.
    meet_app = db.Column(db.String(50), nullable=True)
    meet_title = db.Column(db.String(200), nullable=True)
    meet_time_range = db.Column(db.String(100), nullable=True)

    # When the session row was created (UTC), useful for history and ordering
    created_at = db.Column(db.DateTime, nullable=False, default=_utc_naive)

    student = db.relationship("User", back_populates="sessions")
    teaching_session = db.relationship("TeachingSession", back_populates="student_sessions")
    events = db.relationship(
        "EngagementEvent", back_populates="session", order_by="EngagementEvent.timestamp"
    )


class EngagementEvent(db.Model):
    """Single engagement observation for a session.

    Each event corresponds to one processed frame from the student's webcam.
    engagement_state is either 'engaged' or 'disengaged'.
    """

    __tablename__ = "engagement_events"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    engagement_state = db.Column(db.String(20), nullable=False)
    score = db.Column(db.Float, nullable=False)
    # Explainability fields (stored per-frame so exports are complete)
    reason = db.Column(db.String(255), nullable=True)
    confidence = db.Column(db.Float, nullable=True)

    session = db.relationship("SessionModel", back_populates="events")


###############################################################################
# OpenCV Configuration and Engagement Logic
###############################################################################

# Haar cascade for face detection (no training required).
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Optional (more robust) OpenCV DNN face detector if model files are present.
# We ship the app without heavyweight model binaries; if the user adds the
# standard OpenCV Caffe SSD files under ./models, they'll be used automatically.
MODELS_DIR = BASE_DIR / "models"
DNN_PROTO = MODELS_DIR / "deploy.prototxt"
DNN_MODEL = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# MediaPipe Tasks face landmarker (Python 3.11+ wheels omit legacy `mp.solutions`).
FACE_LANDMARKER_TASK_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _load_dnn_face_net():
    if DNN_PROTO.exists() and DNN_MODEL.exists():
        try:
            return cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_MODEL))
        except Exception:
            return None
    return None


FACE_NET = _load_dnn_face_net()

_face_landmarker = None
_face_landmarker_lock = threading.Lock()


def _ensure_face_landmarker_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = FACE_LANDMARKER_TASK_PATH
    if path.exists() and path.stat().st_size > 10_000:
        return path
    part = path.with_name(path.name + ".part")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_TASK_URL, part)
        if part.stat().st_size < 10_000:
            raise OSError("Downloaded face landmarker file is too small.")
        part.replace(path)
    except Exception as e:
        if part.exists():
            part.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe face landmarker model. "
            f"Save face_landmarker.task into {MODELS_DIR} manually, or fix network access. "
            f"Source: {FACE_LANDMARKER_TASK_URL}"
        ) from e
    return path


def _detect_face_landmarks_rgb(image_rgb: np.ndarray):
    """Run MediaPipe face landmarks on an RGB uint8 image; returns landmark list or None."""
    global _face_landmarker
    if image_rgb is None or image_rgb.size == 0:
        return None
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    if not image_rgb.flags["C_CONTIGUOUS"]:
        image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp_image_module.Image(
        image_format=mp_image_module.ImageFormat.SRGB,
        data=image_rgb,
    )
    with _face_landmarker_lock:
        if _face_landmarker is None:
            model_path = _ensure_face_landmarker_model()
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_base_options.BaseOptions(model_asset_path=str(model_path)),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        result = _face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def _read_admin_passkey():
    """Read admin registration passkey.

    Priority:
      1) Environment variable TE_ADMIN_PASSKEY
      2) admin_passkey.txt in project control
      3) dev fallback ("admin123")
    """
    env_key = os.getenv("TE_ADMIN_PASSKEY")
    if env_key:
        return env_key.strip()

    passkey_path = BASE_DIR / "admin_passkey.txt"
    if passkey_path.exists():
        try:
            for line in passkey_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return line
        except Exception:
            pass

    # Local-dev fallback only
    return "admin123"

# Approximate sampling interval in seconds (should match JS capture interval).
SAMPLING_INTERVAL_SECONDS = 2.0

# Score threshold for declaring engagement.
ENGAGEMENT_THRESHOLD = 0.5

# Cache of previous grayscale frame per active session to estimate motion.
PREVIOUS_FRAMES = {}

# Cache for smoothing engagement scores (session_id -> last_score)
SMOOTHED_SCORES = {}

# Cache of previous face boxes for admin class view (simple tracking-lite)
PREVIOUS_FACE_BOXES = {}

# Cache of previous face "bbox area" (derived from landmark x/y spread) for occlusion guard.
PREVIOUS_FACE_AREAS = {}  # session_id -> float (px^2)

# Engagement temporal smoothing / eye-closure tracking
ENGAGEMENT_HISTORY = {}  # session_id -> deque of last N per-frame scores
EYE_CLOSED_COUNTERS = {}  # session_id -> consecutive closed-eye frames count
LOOKING_AWAY_COUNTERS = {}  # session_id -> consecutive looking-away frames count
LOOKING_DOWN_COUNTERS = {}  # session_id -> consecutive looking-down frames count
OCCLUSION_COUNTERS = {}  # session_id -> consecutive face-occluded frames count
# Per-user/per-session baseline calibration store for yaw/pitch deltas.
# Keyed by student user_id when available; otherwise falls back to session_id.
calibration_store = {}
FRAME_COUNTERS = {}  # session_id -> processed frame count (for debug output)

# EAR threshold and smoothing parameters
# Global EAR threshold used only before calibration finishes.
# (Student baseline uses personal EAR after calibration.)
EAR_THRESHOLD = 0.15
CLOSED_EAR_CONSEC_FRAMES = 3
SMOOTHING_WINDOW = 10

# When confidence is low (e.g., early frames) we return "Unknown" instead of forcing a label.
UNCERTAINTY_MIN_FRAMES = 5
CONFIDENCE_UNKNOWN_THRESHOLD = 0.5

# Fairness: brief head-pose glitches shouldn't instantly label someone disengaged.
# Looking-away must persist for N consecutive frames to be treated as disengaged.
LOOKING_AWAY_CONSEC_FRAMES = int(os.getenv("TE_LOOKING_AWAY_CONSEC_FRAMES", "3"))
# Occlusion (phone/object in front) requires persistence to reduce false positives.
OCCLUSION_CONSEC_FRAMES = int(os.getenv("TE_OCCLUSION_CONSEC_FRAMES", "4"))
LOOKING_DOWN_CONSEC_FRAMES = int(os.getenv("TE_LOOKING_DOWN_CONSEC_FRAMES", "2"))

# Neutral pose calibration (per session)
# 3 clean forward-facing frames is enough to establish a baseline.
BASELINE_MIN_FRAMES = min(int(os.getenv("TE_BASELINE_MIN_FRAMES", "3")), 3)
# Only update baseline when head is roughly forward, so calibration doesn't "learn away"
# from the engaged orientation.
BASELINE_FORWARD_YAW_MAX = float(os.getenv("TE_BASELINE_FORWARD_YAW_MAX", "45"))
BASELINE_FORWARD_PITCH_MAX = float(os.getenv("TE_BASELINE_FORWARD_PITCH_MAX", "30"))

# Looking-away thresholds (in degrees) applied either to raw pose or pose deltas from baseline.
YAW_LOOK_AWAY_THRESHOLD = float(os.getenv("TE_YAW_LOOK_AWAY_THRESHOLD", "20"))
PITCH_LOOK_AWAY_THRESHOLD = float(os.getenv("TE_PITCH_LOOK_AWAY_THRESHOLD", "12"))

# MediaPipe occlusion detection thresholds.
LOW_VIS_THRESHOLD = float(os.getenv("TE_LOW_VIS_THRESHOLD", "0.5"))
# Require a majority of landmarks to be low-visibility.
OCCLUSION_RATIO_THRESHOLD = float(os.getenv("TE_OCCLUSION_RATIO_THRESHOLD", "0.55"))
EXPECTED_LANDMARK_COUNT = 468

# Server-side frame downscaling for lower CPU latency.
# This does NOT store raw video; we only process and discard frames.
MAX_FRAME_DIM = int(os.getenv("TE_MAX_FRAME_DIM", "480"))


def decode_base64_image(data_url):
    """Decode a base64 image sent from the browser into a BGR OpenCV image."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Downscale large frames to keep MediaPipe + solvePnP fast.
        if MAX_FRAME_DIM > 0:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > MAX_FRAME_DIM:
                scale = MAX_FRAME_DIM / float(max_dim)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def calculate_ear(landmarks, image_shape):
    """Calculate Eye Aspect Ratio (EAR) for both eyes from MediaPipe landmarks.

    Returns (left_ear, right_ear, mean_ear) where values are floats.
    """
    h, w = image_shape[0], image_shape[1]
    # MediaPipe face mesh eye landmark indices (MediaPipe FaceMesh)
    # Corrected to avoid EAR collapsing to ~0 when eyes are actually open.
    #
    # LEFT_EYE:  p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    # RIGHT_EYE: p1=33,  p2=160, p3=158, p4=133, p5=153, p6=144
    left_idxs = [362, 385, 387, 263, 373, 380]
    right_idxs = [33, 160, 158, 133, 153, 144]

    def _eye_ear(idxs):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]
        p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
        # vertical distances
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        # horizontal distance
        C = np.linalg.norm(p1 - p4)
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return float(ear)

    left_ear = _eye_ear(left_idxs)
    right_ear = _eye_ear(right_idxs)
    mean_ear = float((left_ear + right_ear) / 2.0)
    return left_ear, right_ear, mean_ear


def estimate_head_pose(image, landmarks):
    """
    Returns:
        yaw   -> left/right
        pitch -> up/down
        roll  -> tilt
    """
    h, w, _ = image.shape

    # 3D face model points
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0),    # Right mouth corner
    ], dtype=np.float64)

    # MediaPipe landmarks -> 2D image points
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),  # Chin
        (landmarks[33].x * w, landmarks[33].y * h),    # Left eye corner
        (landmarks[263].x * w, landmarks[263].y * h),  # Right eye corner
        (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth corner
        (landmarks[291].x * w, landmarks[291].y * h),  # Right mouth corner
    ], dtype=np.float64)

    # Camera matrix
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rot_mat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

    # OpenCV gives [pitch, yaw, roll] from this decomposition
    raw_pitch = float(euler[0, 0])  # up/down
    raw_yaw = float(euler[1, 0])    # left/right
    raw_roll = float(euler[2, 0])   # tilt

    def normalize_angle(angle: float) -> float:
        # Normalize to [-180, 180]
        angle = math.fmod(angle, 360.0)
        if angle > 180.0:
            angle -= 360.0
        elif angle < -180.0:
            angle += 360.0
        return float(angle)

    yaw = normalize_angle(raw_yaw)
    pitch = normalize_angle(raw_pitch)
    roll = normalize_angle(raw_roll)

    # Fix flipped solvePnP outputs.
    if pitch > 90.0:
        pitch = 180.0 - pitch
    elif pitch < -90.0:
        pitch = -180.0 - pitch

    if abs(yaw) > 90.0:
        yaw = (180.0 - abs(yaw)) if yaw > 0 else -(180.0 - abs(yaw))

    print(
        f"ANGLES | raw_pitch={raw_pitch:.1f} -> normalized={pitch:.1f} | "
        f"raw_yaw={raw_yaw:.1f} -> normalized={yaw:.1f}"
    )

    # Required return order
    return yaw, pitch, roll


def get_engagement_status(session_id, image_bgr, landmarks, user_id=None):
    """Compute per-frame engagement score, decide status with temporal smoothing.

    Returns (engagement_score, status, reason, confidence, signals).
    """
    # If no landmarks, immediate not engaged
    if landmarks is None:
        signals = {
            "ear": 0.0,
            "ear_threshold": float(EAR_THRESHOLD),
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
            "baseline_ready": False,
            "calibration_progress": 0.0,
        }
        return 0.0, "Not Engaged", "No Face", 0.0, signals

    # Baseline calibration state for yaw/pitch deltas.
    # Keyed by student user_id when available (otherwise fall back to session_id).
    calib_key = user_id if user_id is not None else session_id
    CALIB_NEEDED = 3
    calib_state = calibration_store.get(calib_key)
    if calib_state is None:
        calib_state = {
            "baseline_yaw": None,
            "baseline_pitch": None,
            "baseline_ready": False,
            "calib_yaw_samples": [],
            "calib_pitch_samples": [],
        }
        calibration_store[calib_key] = calib_state

    # ---------------------------------------------------------------------
    # Baseline calibration rewrite (yaw/pitch only, no EAR/pose gating).
    # - Collect 3 forward-looking samples.
    # - During calibration, return "Setting up..." immediately.
    # - After baseline_ready, continue with the rest of the logic.
    # ---------------------------------------------------------------------
    CALIB_NEEDED = 3
    if not bool(calib_state.get("baseline_ready", False)):
        # Use EAR + head pose values for debug output and to build baseline.
        # (No EAR sanity checks during calibration.)
        try:
            ear_left_tmp, ear_right_tmp, mean_ear_tmp = calculate_ear(
                landmarks, image_bgr.shape
            )
        except Exception:
            ear_left_tmp, ear_right_tmp, mean_ear_tmp = 0.0, 0.0, 0.0

        yaw_tmp, pitch_tmp, _ = estimate_head_pose(image_bgr, landmarks)

        calib_state["calib_yaw_samples"].append(float(yaw_tmp))
        calib_state["calib_pitch_samples"].append(float(pitch_tmp))

        samples_len = len(calib_state["calib_yaw_samples"])

        if samples_len >= CALIB_NEEDED:
            baseline_yaw = float(
                sum(calib_state["calib_yaw_samples"]) / float(samples_len)
            )
            baseline_pitch = float(
                sum(calib_state["calib_pitch_samples"]) / float(samples_len)
            )
            calib_state["baseline_yaw"] = baseline_yaw
            calib_state["baseline_pitch"] = baseline_pitch
            calib_state["baseline_ready"] = True
            print(
                f"BASELINE DONE: yaw_ref={baseline_yaw:.1f} pitch_ref={baseline_pitch:.1f}"
            )
        else:
            status = "Setting up..."
            reason = "Calibrating... please look at your camera"
            confidence = 0.1

            baseline_str = f"{samples_len}/{CALIB_NEEDED}"
            print(
                "FRAME | face=YES | "
                f"baseline={baseline_str} | yaw={yaw_tmp:.1f} | "
                f"pitch={pitch_tmp:.1f} | ear={mean_ear_tmp:.2f} | "
                f"status={status}"
            )

            signals = {
                "ear": float(mean_ear_tmp),
                "ear_threshold": float(EAR_THRESHOLD),
                "yaw_delta": 0.0,
                "pitch_delta": 0.0,
                "baseline_ready": False,
                "calibration_progress": float(samples_len) / float(CALIB_NEEDED),
            }
            return 0.0, status, reason, confidence, signals

    # ----------------------------------------------------------------------------
    # Face occlusion detection (phone/object in front of face)
    #
    # We look at landmark reliability (MediaPipe FaceMesh visibility/presence).
    # When too many landmarks are low-visibility for several consecutive frames,
    # we immediately mark the student as disengaged.
    # ----------------------------------------------------------------------------
    landmark_list = landmarks
    landmark_len = len(landmark_list) if landmark_list is not None else 0

    low_vis_landmarks = 0
    occlusion_ratio = 0.0
    face_occluded_candidate = False

    # Compute occlusion using either visibility or presence, whichever is informative
    # for the MediaPipe version in this environment.
    vis_vals = [getattr(lm, "visibility", None) for lm in landmark_list]
    pres_vals = [getattr(lm, "presence", None) for lm in landmark_list]

    vis_numeric = [v for v in vis_vals if v is not None]
    pres_numeric = [v for v in pres_vals if v is not None]

    vis_non_uniform = (
        (max(vis_numeric) - min(vis_numeric)) > 1e-3 if len(vis_numeric) > 0 else False
    )
    pres_non_uniform = (
        (max(pres_numeric) - min(pres_numeric)) > 1e-3 if len(pres_numeric) > 0 else False
    )

    if vis_non_uniform:
        low_vis_landmarks = sum(1 for v in vis_vals if v is not None and v < LOW_VIS_THRESHOLD)
        occlusion_ratio = float(low_vis_landmarks) / float(max(landmark_len, 1))
        face_occluded_candidate = occlusion_ratio > OCCLUSION_RATIO_THRESHOLD
    elif pres_non_uniform:
        low_vis_landmarks = sum(
            1 for v in pres_vals if v is not None and v < LOW_VIS_THRESHOLD
        )
        occlusion_ratio = float(low_vis_landmarks) / float(max(landmark_len, 1))
        face_occluded_candidate = occlusion_ratio > OCCLUSION_RATIO_THRESHOLD
    else:
        # Fallback: if visibility/presence aren't informative (often all 1.0),
        # infer occlusion from abnormal depth between nose tip and cheeks.
        nose_z = float(getattr(landmark_list[4], "z", 0.0)) if len(landmark_list) > 4 else 0.0
        left_cheek_z = float(getattr(landmark_list[234], "z", 0.0)) if len(landmark_list) > 234 else 0.0
        right_cheek_z = float(getattr(landmark_list[454], "z", 0.0)) if len(landmark_list) > 454 else 0.0
        cheek_avg_z = (left_cheek_z + right_cheek_z) / 2.0
        z_variance = abs(nose_z - cheek_avg_z)
        face_occluded_candidate = z_variance > 0.08
        occlusion_ratio = 1.0 if face_occluded_candidate else 0.0
        low_vis_landmarks = 0

    # We apply the occlusion debounce/priority after EAR is computed,
    # so we can require the guard condition (EAR abnormal OR bbox shrink).

    # Calculate EAR
    try:
        ear_left, ear_right, mean_ear = calculate_ear(landmarks, image_bgr.shape)
    except Exception:
        ear_left, ear_right, mean_ear = 0.0, 0.0, 0.0

    # Estimate head pose early so all return paths can print yaw/pitch debug.
    # NOTE: `estimate_head_pose()` returns (yaw, pitch, roll).
    yaw, pitch, _ = estimate_head_pose(image_bgr, landmarks)

    # EAR sanity check: avoid false disengagement if landmark geometry is wrong.
    if mean_ear > 0.50 or mean_ear < 0.01:
        baseline_ready = bool(calib_state.get("baseline_ready", False))
        calib_samples_len = len(calib_state.get("calib_yaw_samples", []))
        calibration_progress = (
            1.0 if baseline_ready else float(calib_samples_len) / float(CALIB_NEEDED)
        )
        ear_threshold = float(EAR_THRESHOLD)
        # Reset occlusion evidence because landmarks seem unreliable this frame.
        OCCLUSION_COUNTERS[session_id] = 0

        final_status = "Unknown"
        final_reason = "Landmark error"
        confidence = 0.1

        signals = {
            "ear": float(mean_ear),
            "ear_threshold": ear_threshold,
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
            "baseline_ready": bool(baseline_ready),
            "calibration_progress": float(calibration_progress),
            "occlusion_ratio": float(occlusion_ratio),
        }

        baseline_str = "READY" if baseline_ready else f"{calib_samples_len}/{CALIB_NEEDED}"
        print(
            "FRAME | face=YES | "
            f"baseline={baseline_str} | "
            f"yaw={yaw:.1f} | pitch={pitch:.1f} | "
            f"ear={mean_ear:.2f} | status={final_status}"
        )

        return 0.0, final_status, final_reason, confidence, signals

    # ----------------------------------------------------------------------------
    # Occlusion guard (phone/object in front):
    # We only treat the face as occluded if:
    #   - landmark reliability indicates occlusion (occlusion_ratio threshold), AND
    #   - either EAR is abnormal OR the inferred landmark bbox area suddenly shrinks.
    # This prevents visibility-only false positives.
    # ----------------------------------------------------------------------------
    h, w, _ = image_bgr.shape
    xs = [float(lm.x) * w for lm in landmark_list]
    ys = [float(lm.y) * h for lm in landmark_list]
    bbox_w = float(max(xs) - min(xs)) if xs else 0.0
    bbox_h = float(max(ys) - min(ys)) if ys else 0.0
    bbox_area = max(0.0, bbox_w * bbox_h)

    prev_bbox_area = float(PREVIOUS_FACE_AREAS.get(session_id, 0.0))
    bbox_shrunk = (
        prev_bbox_area > 0.0 and bbox_area > 0.0 and (bbox_area / prev_bbox_area) < 0.7
    )
    PREVIOUS_FACE_AREAS[session_id] = float(bbox_area)

    ear_abnormal = float(mean_ear) < float(EAR_THRESHOLD)
    occlusion_candidate_guarded = bool(face_occluded_candidate) and (
        ear_abnormal or bbox_shrunk
    )

    occl_count = OCCLUSION_COUNTERS.get(session_id, 0)
    if occlusion_candidate_guarded:
        occl_count += 1
    else:
        occl_count = 0
    OCCLUSION_COUNTERS[session_id] = occl_count
    face_occluded_confirmed = occl_count >= OCCLUSION_CONSEC_FRAMES

    if face_occluded_confirmed:
        # Occlusion takes priority: do not run head-pose or eyes-closed checks.
        EYE_CLOSED_COUNTERS[session_id] = 0
        LOOKING_AWAY_COUNTERS[session_id] = 0
        LOOKING_DOWN_COUNTERS[session_id] = 0

        hist = deque(maxlen=SMOOTHING_WINDOW)
        hist.append(0.1)
        ENGAGEMENT_HISTORY[session_id] = hist
        avg_score = 0.1

        baseline_ready = bool(calib_state.get("baseline_ready", False))
        calibration_progress = (
            1.0
            if baseline_ready
            else float(len(calib_state.get("calib_yaw_samples", []))) / float(CALIB_NEEDED)
        )
        ear_threshold = float(EAR_THRESHOLD)

        final_reason = "Face blocked — possible phone/object use"
        final_status = "Not Engaged"
        confidence = 0.85

        signals = {
            "ear": float(mean_ear),
            "ear_threshold": ear_threshold,
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
            "baseline_ready": bool(baseline_ready),
            "calibration_progress": float(calibration_progress),
            "occlusion_ratio": float(occlusion_ratio),
        }

        baseline_str = (
            "READY"
            if baseline_ready
            else f"{len(calib_state.get('calib_yaw_samples', []))}/{CALIB_NEEDED}"
        )
        print(
            f"FRAME | face=YES | baseline={baseline_str} | "
            f"yaw={yaw:.1f} | pitch={pitch:.1f} | ear={mean_ear:.2f} | "
            f"status={final_status}"
        )

        return avg_score, final_status, final_reason, confidence, signals

    eyes_closed = False

    # Baseline is ready (we either already completed it above during calibration,
    # or it was seeded as ready).
    baseline_ready = bool(calib_state.get("baseline_ready", False))
    yaw_ref = float(calib_state.get("baseline_yaw", 0.0)) if baseline_ready else 0.0
    pitch_ref = float(calib_state.get("baseline_pitch", 0.0)) if baseline_ready else 0.0

    # Looking-away uses delta (not absolute pose).
    yaw_delta = abs(yaw - yaw_ref)
    pitch_delta_signed = float(pitch - pitch_ref)  # + = looking down, - = looking up
    pitch_delta = abs(pitch_delta_signed)

    # Directional gaze checks.
    yaw_turn = yaw_delta > YAW_LOOK_AWAY_THRESHOLD
    looking_down_pose = pitch_delta_signed > PITCH_LOOK_AWAY_THRESHOLD
    looking_up_pose = (not looking_down_pose) and (pitch_delta > PITCH_LOOK_AWAY_THRESHOLD)

    # Pose "candidate" for looking-away (debounced below).
    looking_away_pose = yaw_turn or looking_down_pose or looking_up_pose

    # EAR threshold is constant after baseline (no EAR gating during calibration).
    active_ear_threshold = float(EAR_THRESHOLD)

    if not baseline_ready:
        # Don't accumulate disengagement evidence while calibrating.
        EYE_CLOSED_COUNTERS[session_id] = 0
        LOOKING_AWAY_COUNTERS[session_id] = 0
        LOOKING_DOWN_COUNTERS[session_id] = 0
        eyes_closed = False
    else:
        closed_count = EYE_CLOSED_COUNTERS.get(session_id, 0)
        if mean_ear < active_ear_threshold:
            closed_count += 1
        else:
            closed_count = 0
        EYE_CLOSED_COUNTERS[session_id] = closed_count
        eyes_closed = closed_count >= CLOSED_EAR_CONSEC_FRAMES

    # Debounce looking-away to avoid unfair short spikes.
    # Downward gaze (phone) is treated as a stronger signal with fewer consecutive frames.
    if not baseline_ready:
        looking_away_confirmed = False
        looking_down_confirmed = False
    else:
        # Downward debounce.
        down_count = LOOKING_DOWN_COUNTERS.get(session_id, 0)
        if looking_down_pose:
            down_count += 1
        else:
            down_count = 0
        LOOKING_DOWN_COUNTERS[session_id] = down_count
        looking_down_confirmed = down_count >= LOOKING_DOWN_CONSEC_FRAMES

        # General debounce for yaw or upward gaze.
        general_pose = yaw_turn or looking_up_pose
        la_count = LOOKING_AWAY_COUNTERS.get(session_id, 0)
        if general_pose:
            la_count += 1
        else:
            la_count = 0
        LOOKING_AWAY_COUNTERS[session_id] = la_count
        looking_away_confirmed = looking_down_confirmed or (la_count >= LOOKING_AWAY_CONSEC_FRAMES)

    # Per-frame scoring.
    # Order matters for fairness/explainability:
    # 1) If head-pose indicates looking-away (debounced), force disengaged.
    # 2) Only if head pose is OK, then apply EAR/eyes-closed logic.
    if baseline_ready and looking_away_confirmed:
        frame_score = 0.0
        reason = (
            "Looking down — possible phone use"
            if looking_down_confirmed
            else ("Looking away (up)" if looking_up_pose else "Looking Away")
        )
    elif eyes_closed:
        frame_score = 0.0
        reason = f"Eyes closed (EAR: {mean_ear:.2f})"
    else:
        frame_score = 1.0
        reason = "OK"

    # Append to history deque for smoothing
    hist = ENGAGEMENT_HISTORY.get(session_id)
    if hist is None:
        hist = deque(maxlen=SMOOTHING_WINDOW)
        ENGAGEMENT_HISTORY[session_id] = hist
    hist.append(frame_score)

    # Average score over smoothing window
    avg_score = float(np.mean(list(hist))) if hist else float(frame_score)

    # Final status determination (baseline is already ready because we early-return while calibrating).
    # Head-pose confirmed looking-away always forces Not Engaged; EAR/eyes-closed cannot override it.
    if looking_away_confirmed or eyes_closed or avg_score < 0.5:
        final_status = "Not Engaged"
    else:
        final_status = "Engaged"

    # Explainable multi-factor reason text (based on current signals).
    if baseline_ready and looking_away_confirmed:
        if looking_down_confirmed:
            final_reason = "Looking down — possible phone use"
        elif yaw_turn and looking_up_pose:
            final_reason = (
                f"Looking left/right (yaw delta: {yaw_delta:.1f}°) + Looking away (up)"
            )
        elif yaw_turn:
            final_reason = f"Looking left/right (yaw delta: {yaw_delta:.1f}°)"
        elif looking_up_pose:
            final_reason = f"Looking away (up) (pitch delta: {pitch_delta:.1f}°)"
        else:
            final_reason = "Looking Away"
    else:
        ear_low = mean_ear < float(EAR_THRESHOLD)
        if ear_low:
            final_reason = f"Eyes closed (EAR: {mean_ear:.2f})"
        else:
            final_reason = "Low Attention" if final_status == "Not Engaged" else "OK"

    calibration_progress = (
        1.0
        if baseline_ready
        else float(len(calib_state.get("calib_yaw_samples", []))) / float(CALIB_NEEDED)
    )

    # Confidence is highest when rules are decisive (eyes closed / looking away)
    # and when we have enough samples to stabilize the smoothing window.
    hist_len = len(hist) if hist is not None else 0
    attention_strength = float(abs(avg_score - 0.5) * 2.0)  # 0..1
    if eyes_closed:
        base_conf = 0.95
    else:
        base_conf = 0.40 + 0.60 * attention_strength  # 0.40..1.00

    window_factor = min(hist_len / float(SMOOTHING_WINDOW), 1.0) if SMOOTHING_WINDOW > 0 else 1.0
    confidence = float(np.clip(base_conf * window_factor, 0.0, 1.0))

    signals = {
        "ear": float(mean_ear),
        "ear_threshold": float(EAR_THRESHOLD),
        "yaw_delta": float(yaw_delta),
        "pitch_delta": float(pitch_delta),
        "baseline_ready": bool(baseline_ready),
        "calibration_progress": float(calibration_progress),
    }

    baseline_samples_len = len(calib_state.get("calib_yaw_samples", []))
    baseline_str = "READY" if baseline_ready else f"{baseline_samples_len}/{CALIB_NEEDED}"
    print(
        "FRAME | face=YES | "
        f"baseline={baseline_str} | yaw={yaw:.1f} | pitch={pitch:.1f} | "
        f"ear={mean_ear:.2f} | status={final_status}"
    )

    return avg_score, final_status, final_reason, confidence, signals


def compute_engagement_for_frame(session_id, image_bgr, user_id=None):
    """Compute engagement for a single frame using EAR + head-pose + smoothing.

    Returns:
      - engagement_score: float (0..1) or None when no face is visible
      - status: "Engaged"|"Not Engaged"|"Unknown"|"No Face Detected"
      - reason: explanation string
      - confidence: float (0..1)
      - signals: dict or None
    """
    if image_bgr is None:
        return None, "No Face Detected", "Face not visible", 0.0, None

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = _detect_face_landmarks_rgb(image_rgb)

    if not landmarks:
        # No face detected in the current frame.
        # Do NOT silently skip it; return a dedicated status so the UI
        # can warn the user and analytics can exclude it fairly.
        EYE_CLOSED_COUNTERS.pop(session_id, None)
        LOOKING_AWAY_COUNTERS.pop(session_id, None)
        LOOKING_DOWN_COUNTERS.pop(session_id, None)
        OCCLUSION_COUNTERS.pop(session_id, None)

        # Debug print for "face=NO" frames.
        CALIB_NEEDED = 3
        calib_key = user_id if user_id is not None else session_id
        calib_state = calibration_store.get(calib_key)
        if calib_state is None:
            calib_state = {
                "baseline_yaw": None,
                "baseline_pitch": None,
                "baseline_ready": False,
                "calib_yaw_samples": [],
                "calib_pitch_samples": [],
            }
            calibration_store[calib_key] = calib_state
        baseline_ready = bool(calib_state.get("baseline_ready", False))
        baseline_str = (
            "READY"
            if baseline_ready
            else f"{len(calib_state.get('calib_yaw_samples', []))}/{CALIB_NEEDED}"
        )
        print(
            "FRAME | face=NO | "
            f"baseline={baseline_str} | yaw=0.0 | pitch=0.0 | ear=0.00 | "
            "status=No Face Detected"
        )
        # Keep ENGAGEMENT_HISTORY so we don't "punish" the user by clearing
        # smoothing state; unknown/no-face are excluded from disengaged time.
        return None, "No Face Detected", "Face not visible", 0.0, None

    score, status, reason, confidence, signals = get_engagement_status(
        session_id, image_bgr, landmarks, user_id=user_id
    )
    return float(score), status, reason, float(confidence), signals


def _detect_faces(gray, image_bgr=None):
    """Return list of faces as (x, y, w, h). Uses DNN if available.

    DNN tends to be more robust under angles/lighting than Haar cascades.
    We keep Haar as a zero-dependency fallback.
    """
    if FACE_NET is not None and image_bgr is not None:
        try:
            h, w = image_bgr.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            FACE_NET.setInput(blob)
            detections = FACE_NET.forward()
            faces = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < 0.5:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
        except Exception:
            # fall back to Haar
            pass

    return FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))


def compute_class_engagement_for_frame(session_id, image_bgr):
    """Compute engagement for multiple faces in a frame (admin/class dashboard).

    Returns:
      - class_engagement: float (0..1)
      - students: list[dict] with id/status/engagement_score

    Notes:
      - This is still heuristic and lightweight (no heavy ML weights required).
      - Uses per-face center + sharpness + per-face motion proxy.
      - Adds temporal smoothing to reduce flicker.
    """
    if image_bgr is None:
        return 0.0, []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_center = np.array([w / 2.0, h / 2.0])

    faces = list(_detect_faces(gray, image_bgr=image_bgr))
    if not faces:
        return 0.0, []

    # Global motion for this "session" (admin feed)
    prev_gray = PREVIOUS_FRAMES.get(session_id)
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_gray)
        global_motion = float(np.clip(np.mean(diff) / 50.0, 0.0, 1.0))
    else:
        global_motion = 0.2
    PREVIOUS_FRAMES[session_id] = gray

    # Previous boxes to approximate per-face motion
    prev_boxes = PREVIOUS_FACE_BOXES.get(session_id, [])

    def box_center(b):
        x, y, bw, bh = b
        return np.array([x + bw / 2.0, y + bh / 2.0])

    students = []
    scores = []

    # Sort by size (largest first) so IDs are stable-ish between frames
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for idx, (x, y, bw, bh) in enumerate(faces_sorted[:10], start=1):
        face_center = np.array([x + bw / 2.0, y + bh / 2.0])
        dist = np.linalg.norm(face_center - img_center)
        max_dist = np.linalg.norm(img_center)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        # Sharpness proxy (engaged faces tend to be sharper & frontal)
        roi = gray[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        if roi.size:
            lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            sharpness = float(np.clip(lap_var / 200.0, 0.0, 1.0))
        else:
            sharpness = 0.0

        # Per-face motion: nearest previous face center distance
        per_face_motion = global_motion
        if prev_boxes:
            c = box_center((x, y, bw, bh))
            dists = [float(np.linalg.norm(c - box_center(pb))) for pb in prev_boxes]
            if dists:
                # normalize by face size (so moving ~face_width counts similarly)
                per_face_motion = float(np.clip((min(dists) / max(bw, 1)) / 0.8, 0.0, 1.0))

        # Compose score (lightweight heuristic)
        score = 0.50 * 1.0 + 0.25 * center_score + 0.15 * sharpness + 0.10 * per_face_motion
        score = float(np.clip(score, 0.0, 1.0))

        # Mild temporal smoothing for stability
        key = f"{session_id}:{idx}"
        prev_s = SMOOTHED_SCORES.get(key)
        if prev_s is not None:
            score = 0.7 * float(prev_s) + 0.3 * score
        SMOOTHED_SCORES[key] = score

        status = "Engaged" if score >= ENGAGEMENT_THRESHOLD else "Not Engaged"
        students.append({"id": f"Student {idx}", "status": status, "engagement_score": score})
        scores.append(score)

    PREVIOUS_FACE_BOXES[session_id] = [(x, y, bw, bh) for (x, y, bw, bh) in faces_sorted[:10]]
    class_engagement = float(np.mean(scores)) if scores else 0.0
    return class_engagement, students


def compute_class_engagement_for_frame_mediapipe(session_id, image_bgr):
    """Compute engagement for multiple faces using MediaPipe per detected face.

    This is heavier than the original heuristic, but it aligns the admin dashboard
    with the same core signals used on the student endpoint:
      - Eye Aspect Ratio (EAR) for eye-closed detection
      - Head pose (yaw/pitch) for looking-away detection
      - Temporal smoothing inside `get_engagement_status()`
    """
    if image_bgr is None:
        return 0.0, []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = list(_detect_faces(gray, image_bgr=image_bgr))
    if not faces:
        return 0.0, []

    h, w = gray.shape
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    max_faces = int(os.getenv("TE_ADMIN_MAX_FACES", "5"))

    students: list[dict] = []
    scores: list[float] = []

    for idx, (x, y, bw, bh) in enumerate(faces_sorted[:max_faces], start=1):
        # Add padding so MediaPipe sees the full face (forehead + chin).
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        face_crop = image_bgr[y1:y2, x1:x2]
        face_key = f"{session_id}:face{idx}"

        if face_crop is None or face_crop.size == 0:
            score, status, reason, confidence, _signals = (
                None,
                "No Face Detected",
                "Face not visible",
                0.0,
                None,
            )
        else:
            score, status, reason, confidence, _signals = compute_engagement_for_frame(
                face_key, face_crop
            )

        score_val = float(score) if score is not None else 0.0

        students.append(
            {
                "id": f"Student {idx}",
                "status": status,
                "engagement_score": score_val,
                "reason": reason,
                "confidence": float(confidence),
            }
        )
        scores.append(score_val)

    class_engagement = float(np.mean(scores)) if scores else 0.0
    return class_engagement, students


def get_head_pose(image, landmarks):
    """
    Estimate head pose (pitch, yaw, roll) from MediaPipe landmarks.
    Returns: (pitch, yaw, roll) in degrees.
    """
    h, w, _ = image.shape
    
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points from landmarks
    # MP indices: Nose=1, Chin=152, L-Eye-L=33, R-Eye-R=263, L-Mouth=61, R-Mouth=291
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h)
    ], dtype="double")

    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Combine to projection matrix
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    
    # Decompose projection matrix to Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    
    raw_pitch, raw_yaw, raw_roll = [float(v) for v in euler_angles.flatten()[:3]]

    def normalize_angle(angle: float) -> float:
        angle = math.fmod(angle, 360.0)
        if angle > 180.0:
            angle -= 360.0
        elif angle < -180.0:
            angle += 360.0
        return float(angle)

    yaw = normalize_angle(raw_yaw)
    pitch = normalize_angle(raw_pitch)
    roll = normalize_angle(raw_roll)

    if pitch > 90.0:
        pitch = 180.0 - pitch
    elif pitch < -90.0:
        pitch = -180.0 - pitch

    if abs(yaw) > 90.0:
        yaw = (180.0 - abs(yaw)) if yaw > 0 else -(180.0 - abs(yaw))

    print(
        f"ANGLES | raw_pitch={raw_pitch:.1f} -> normalized={pitch:.1f} | "
        f"raw_yaw={raw_yaw:.1f} -> normalized={yaw:.1f}"
    )

    return pitch, yaw, roll


###############################################################################
# Authentication Helpers
###############################################################################


def create_user(username, password, role):
    """Create a user with hashed password and role ('admin' or 'student')."""
    from sqlalchemy.exc import IntegrityError

    existing = User.query.filter_by(username=username).first()
    if existing is not None:
        return existing

    user = User(
        username=username,
        password_hash=generate_password_hash(password),
        role=role,
    )
    db.session.add(user)
    try:
        db.session.commit()
        return user
    except IntegrityError:
        # If another code path seeded/inserted the same username, return it.
        db.session.rollback()
        return User.query.filter_by(username=username).first()


def verify_user(username, password, role=None):
    """Verify username/password and optional role filter."""
    query = User.query.filter_by(username=username)
    if role:
        query = query.filter_by(role=role)
    user = query.first()
    if user and check_password_hash(user.password_hash, password):
        return user
    return None


def require_role(required_role):
    """Decorator factory enforcing that the current user has the given role."""

    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = session.get("user_id")
            role = session.get("role")
            if not user_id or role != required_role:
                flash("Please log in with the correct role.", "warning")
                if required_role == "admin":
                    return redirect(url_for("admin_login"))
                return redirect(url_for("student_login"))
            return func(*args, **kwargs)

        return wrapper

    return decorator


###############################################################################
# Public Routes
###############################################################################


@app.route("/")
def home():
    """Simple landing page with Login / Register buttons."""
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register either a student or an admin account."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()
        role = request.form.get("role", "student")
        admin_passkey = request.form.get("admin_passkey", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif role not in {"admin", "student"}:
            flash("Invalid role selected.", "danger")
        elif role == "admin":
            # Require a passkey to prevent students from self-registering as admin.
            # Configure it via environment variable TE_ADMIN_PASSKEY or admin_passkey.txt.
            required_passkey = _read_admin_passkey()
            if not admin_passkey or admin_passkey != required_passkey:
                flash("Invalid admin passkey.", "danger")
            elif User.query.filter_by(username=username).first():
                flash("Username already exists. Choose another one.", "danger")
            else:
                create_user(username, password, role)
                flash(f"Registered {role} '{username}'. Please log in.", "success")
                return redirect(url_for("admin_login"))
        elif User.query.filter_by(username=username).first():
            flash("Username already exists. Choose another one.", "danger")
        else:
            create_user(username, password, role)
            flash(f"Registered {role} '{username}'. Please log in.", "success")
            return redirect(url_for("student_login"))

    return render_template("register.html")


###############################################################################
# Student Login, Dashboard & Engagement API
###############################################################################


@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    """Student login.

    Important: we do NOT create a SessionModel row here anymore.
    Creating a session row now happens after the student submits meet details
    on `/student/start-session`.
    """
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = verify_user(username, password, role="student")
        if user:
            session["user_id"] = user.id
            session["role"] = "student"

            flash(f"Welcome, {user.username}!", "success")
            return redirect(url_for("student_join_session"))

        flash("Invalid student credentials.", "danger")

    return render_template("student_login.html")


@app.route("/student/home")
@require_role("student")
def student_home():
    """Student landing page showing history and join button."""
    user_id = session.get("user_id")
    user = db.session.get(User, user_id)
    
    # Fetch recent sessions
    sessions = (
        SessionModel.query.filter_by(student_id=user_id)
        .order_by(SessionModel.created_at.desc())
        .limit(10)
        .all()
    )
    
    return render_template("student_home.html", username=user.username, sessions=sessions)



@app.route("/student/start-session", methods=["GET", "POST"])
@require_role("student")
def student_start_session():
    """Start the session. Uses the pending teaching session ID from join-session."""
    
    pending_ts_id = session.get("pending_teaching_session_id")
    
    if pending_ts_id:
        # Auto-start logic if we have a pending teaching session
        teaching_session = db.session.get(TeachingSession, pending_ts_id)
        if not teaching_session:
            session.pop("pending_teaching_session_id", None)
            flash("Session expired or invalid.", "danger")
            return redirect(url_for("student_join_session"))
            
        # Create the student session immediately
        new_sess = SessionModel(
            student_id=session["user_id"],
            login_time=_utc_naive(),
            meet_app=teaching_session.meet_app,
            meet_title=teaching_session.meet_title,
            # Format time range string for backward compatibility
            meet_time_range=f"{teaching_session.date.strftime('%d-%m-%Y')} | {teaching_session.start_time.strftime('%I:%M %p')} - {teaching_session.end_time.strftime('%I:%M %p')}",
            teaching_session_id=teaching_session.id
        )
        db.session.add(new_sess)
        db.session.commit()
        
        session["session_id"] = new_sess.id
        # Clear the pending ID
        session.pop("pending_teaching_session_id", None)
        
        return redirect(url_for("student_dashboard"))

    # If no pending session, redirect to join page
    return redirect(url_for("student_join_session"))


@app.route("/student/join-session", methods=["GET", "POST"])
@require_role("student")
def student_join_session():
    """Student enters a session code to join a class."""
    if request.method == "POST":
        code = request.form.get("session_code", "").strip()
        teaching_session = TeachingSession.query.filter_by(session_code=code).first()
        
        if not teaching_session:
            flash("Invalid Session ID. Please check and try again.", "danger")
            return redirect(url_for("student_join_session"))
            
        # Store teaching_session_id in session to be used in start_session
        session["pending_teaching_session_id"] = teaching_session.id
        
        # Redirect to start session logic
        return redirect(url_for("student_start_session"))
        
    return render_template("student_join_session.html")


@app.route("/student/dashboard")
@require_role("student")
def student_dashboard():
    """Student dashboard; browser sends frames to /api/engagement."""
    session_id = session.get("session_id")
    current_session = None
    if session_id:
        current_session = db.session.get(SessionModel, session_id)
        
    return render_template("student_dashboard.html", current_session=current_session)


@app.route("/student/session/<int:session_id>")
@require_role("student")
def student_session_detail(session_id):
    """Detailed analysis view for a specific student session."""
    user_id = session.get("user_id")
    sess = db.get_or_404(SessionModel, session_id)

    # Ensure the session belongs to the logged-in student
    if sess.student_id != user_id:
        flash("You do not have permission to view this session.", "danger")
        return redirect(url_for("student_home"))

    stats = _compute_session_stats(session_id)
    
    # Calculate percentages for charts
    total = stats["duration_seconds"]
    engaged_pct = 0
    disengaged_pct = 0
    if total > 0:
        engaged_pct = round((stats["engaged_seconds"] / total) * 100, 1)
        disengaged_pct = round((stats["disengaged_seconds"] / total) * 100, 1)

    return render_template(
        "student_session_detail.html",
        session_obj=sess,
        stats=stats,
        engaged_pct=engaged_pct,
        disengaged_pct=disengaged_pct
    )


@app.route("/student/logout")
@require_role("student")
def student_logout():
    """Close the current session and log the student out."""
    session_id = session.get("session_id")
    if session_id is not None:
        session_model = db.session.get(SessionModel, session_id)
        if session_model and session_model.logout_time is None:
            session_model.logout_time = _utc_naive()
            db.session.commit()

    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/admin/logout")
@require_role("admin")
def admin_logout():
    """Log the admin out by clearing the session."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/api/engagement", methods=["POST"])
@require_role("student")
def api_engagement():
    """Receive webcam frame from student and log engagement event.

    Request JSON:
        {"image": "<base64 data URL>"}

    Response JSON:
        {"engagement_state": "engaged"|"disengaged", "score": 0.73}
    """
    session_id = session.get("session_id")
    if session_id is None:
        return jsonify({"error": "No active session."}), 400

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field."}), 400

    try:
        img = decode_base64_image(data["image"])
        score, state, reason, confidence, signals = compute_engagement_for_frame(
            session_id, img, user_id=session.get("user_id")
        )
    except Exception as e:
        return jsonify(
            {
                "error": "Engagement processing failed.",
                "detail": str(e),
                "engagement_score": None,
                "status": "Unknown",
                "reason": "Server vision pipeline error",
                "confidence": 0.0,
                "calibration_progress": 0.0,
                "baseline_ready": False,
                "signals": None,
                "stats": _compute_session_stats(session_id),
            }
        ), 503

    baseline_ready = bool(signals.get("baseline_ready")) if signals else False
    calibration_progress = (
        float(signals.get("calibration_progress", 0.0)) if signals else 0.0
    )

    # Persist the observation so analytics can compute fair time + detection rate.
    # State mapping:
    #   Engaged      -> engaged
    #   Not Engaged  -> disengaged
    #   Unknown      -> unknown (excluded from engaged/disengaged totals)
    #   No Face Detected -> no_face (excluded from engaged/disengaged totals)
    if state == "Engaged":
        db_state = "engaged"
        score_to_store = float(score)
    elif state == "Not Engaged":
        db_state = "disengaged"
        score_to_store = float(score)
    elif state == "No Face Detected":
        db_state = "no_face"
        score_to_store = 0.0
    else:
        db_state = "unknown"
        score_to_store = float(score) if score is not None else 0.0

    event = EngagementEvent(
        session_id=session_id,
        timestamp=_utc_naive(),
        engagement_state=db_state,
        score=float(score_to_store),
        reason=reason,
        confidence=float(confidence) if confidence is not None else None,
    )
    db.session.add(event)
    db.session.commit()

    # Debug: verify events are persisting + how many we have so far.
    try:
        saved_count = (
            EngagementEvent.query.filter_by(session_id=session_id).count()
        )
        print(f"SAVED EVENT: {state} score={float(score_to_store) if score_to_store is not None else 'n/a'}")
        print(f"TOTAL EVENTS IN DB FOR THIS SESSION: {saved_count}")
    except Exception:
        # Debug should never break the API.
        pass

    # Compute cumulative stats for this session to send back to the student dashboard
    stats = _compute_session_stats(session_id)

    return jsonify({
        "engagement_score": (
            None
            if str(state).lower() == "no face detected"
            or str(state).lower() == "unknown"
            or str(state).lower().startswith("setting up")
            else round(float(score), 3)
        ),
        "status": state,
        "reason": reason,
        "confidence": round(float(confidence), 3),
        "calibration_progress": round(float(calibration_progress), 3),
        "baseline_ready": baseline_ready,
        "signals": signals,
        "stats": stats,
    })


@app.route("/api/engagement/reset_baseline", methods=["POST"])
@require_role("student")
def api_reset_baseline():
    """Reset baseline calibration for the current student session."""
    session_id = session.get("session_id")
    user_id = session.get("user_id")
    if session_id is None:
        return jsonify({"error": "No active session."}), 400

    calib_key = user_id if user_id is not None else session_id
    calib_state = calibration_store.get(calib_key)
    if calib_state is None:
        calib_state = {}
    calib_state["baseline_yaw"] = None
    calib_state["baseline_pitch"] = None
    calib_state["baseline_ready"] = False
    calib_state["calib_yaw_samples"] = []
    calib_state["calib_pitch_samples"] = []
    calibration_store[calib_key] = calib_state

    # Clear per-frame counters so evidence doesn't leak into the next baseline.
    EYE_CLOSED_COUNTERS.pop(session_id, None)
    LOOKING_AWAY_COUNTERS.pop(session_id, None)
    LOOKING_DOWN_COUNTERS.pop(session_id, None)
    OCCLUSION_COUNTERS.pop(session_id, None)

    # Reset smoothing to avoid any carryover from previous baseline state.
    ENGAGEMENT_HISTORY.pop(session_id, None)

    return jsonify({"status": "reset"})


@app.route("/api/session/<int:session_id>/export", methods=["GET"])
@require_role("admin")
def api_session_export(session_id: int):
    """Export a single student's session engagement events as CSV."""
    events = (
        EngagementEvent.query.filter(EngagementEvent.session_id == session_id)
        .order_by(EngagementEvent.timestamp.asc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "status", "engagement_score", "reason", "confidence"])

    for ev in events:
        ts = ev.timestamp.isoformat() if getattr(ev, "timestamp", None) else ""
        status = ev.engagement_state
        engagement_score = ev.score
        reason = ev.reason if getattr(ev, "reason", None) else ""
        confidence = ev.confidence if getattr(ev, "confidence", None) is not None else ""
        writer.writerow([ts, status, engagement_score, reason, confidence])

    csv_bytes = output.getvalue().encode("utf-8")
    output.close()

    filename = f"session_{session_id}_report.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


###############################################################################
# Admin Real-time Dashboard (class view)
###############################################################################


@app.route("/dashboard")
@require_role("admin")
def dashboard():
    """Admin real-time class dashboard (webcam based)."""
    user_id = session.get("user_id")
    user = db.session.get(User, user_id) if user_id else None
    return render_template("dashboard.html", username=(user.username if user else "admin"))


@app.route("/detect", methods=["POST"])
@require_role("admin")
def detect():
    """Receive a webcam frame and return per-face engagement (admin/class view)."""
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field."}), 400

    try:
        img = decode_base64_image(data["image"])
        # Use a stable ID for admin realtime smoothing
        admin_session_id = f"admin:{session.get('user_id', 'unknown')}"
        class_engagement, students = compute_class_engagement_for_frame_mediapipe(
            admin_session_id, img
        )
    except Exception as e:
        return jsonify({"error": "Detection failed.", "detail": str(e)}), 503

    return jsonify(
        {
            "class_engagement": round(float(class_engagement), 3),
            "total_students": int(len(students)),
            "students": [
                {
                    "id": s["id"],
                    "status": s["status"],
                    "engagement_score": round(float(s["engagement_score"]), 3),
                    "reason": s.get("reason", ""),
                    "confidence": round(float(s.get("confidence", 0.0)), 3),
                }
                for s in students
            ],
        }
    )


###############################################################################
# Admin Login & Analytics
###############################################################################


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Admin login route."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = verify_user(username, password, role="admin")
        if user:
            session["user_id"] = user.id
            session["role"] = "admin"
            flash(f"Welcome, admin {user.username}!", "success")
            return redirect(url_for("admin_dashboard"))

        flash("Invalid admin credentials.", "danger")

    return render_template("admin_login.html")


def compute_session_stats(session_model):
    """Approximate engagement statistics for a session based on its events."""
    events = session_model.events
    if not events:
        return {
            "duration_seconds": 0.0,
            "engaged_seconds": 0.0,
            "disengaged_seconds": 0.0,
            "engaged_to_disengaged": 0,
            "disengaged_to_engaged": 0,
        }

    engaged_count = sum(1 for e in events if e.engagement_state == "engaged")
    unknown_count = sum(
        1 for e in events if str(e.engagement_state).lower() == "unknown"
    )
    no_face_count = sum(
        1
        for e in events
        if str(e.engagement_state).lower()
        in {"no_face", "no face detected", "no-face"}
    )
    disengaged_count = len(events) - engaged_count - unknown_count - no_face_count

    engaged_seconds = engaged_count * SAMPLING_INTERVAL_SECONDS
    disengaged_seconds = disengaged_count * SAMPLING_INTERVAL_SECONDS

    engaged_to_disengaged = 0
    disengaged_to_engaged = 0
    for prev, curr in zip(events[:-1], events[1:]):
        prev_state = str(prev.engagement_state).lower()
        curr_state = str(curr.engagement_state).lower()
        if prev_state == "engaged" and curr_state == "disengaged":
            engaged_to_disengaged += 1
        elif prev_state == "disengaged" and curr_state == "engaged":
            disengaged_to_engaged += 1

    end_time = session_model.logout_time or _utc_naive()
    duration_seconds = max((end_time - session_model.login_time).total_seconds(), 0.0)

    return {
        "duration_seconds": duration_seconds,
        "engaged_seconds": engaged_seconds,
        "disengaged_seconds": disengaged_seconds,
        "engaged_to_disengaged": engaged_to_disengaged,
        "disengaged_to_engaged": disengaged_to_engaged,
    }


@app.route("/admin/dashboard")
@require_role("admin")
def admin_dashboard():
    """Admin dashboard showing grouped meet sessions."""
    # Get current user for welcome message
    user_id = session.get("user_id")
    user = db.session.get(User, user_id)
    username = user.username if user else "Admin"

    # Default to 'today' if no filter provided
    filter_type = request.args.get("filter", "today").lower()
    start_date_str = request.args.get("start_date", "")
    end_date_str = request.args.get("end_date", "")
    
    now = _utc_naive()
    start_date = None
    end_date = None
    
    # Apply quick filters or custom date range
    if filter_type == "today":
        start_date = datetime(now.year, now.month, now.day)
        end_date = now + timedelta(days=1) # Include full day
    elif filter_type == "week":
        start_date = now - timedelta(days=7)
        end_date = now + timedelta(days=1)
    elif filter_type == "month":
        start_date = now - timedelta(days=30)
        end_date = now + timedelta(days=1)
    elif filter_type == "all":
        start_date = None
        end_date = None
    elif start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            flash("Invalid date format. Use YYYY-MM-DD.", "warning")
            start_date = None
            end_date = None
    
    # Base query
    q = SessionModel.query
    if start_date:
        q = q.filter(SessionModel.login_time >= start_date)
    if end_date:
        q = q.filter(SessionModel.login_time < end_date)
        
    all_sessions = q.order_by(SessionModel.login_time.desc()).all()
    
    # Group by (meet_title, meet_time_range, meet_app)
    grouped_meets = {}
    
    for sess in all_sessions:
        title = sess.meet_title or "Untitled Meet"
        time_range = sess.meet_time_range or "Unknown Time"
        app_name = sess.meet_app or "Unknown App"
        
        key = (title, time_range, app_name)
        
        if key not in grouped_meets:
            grouped_meets[key] = {
                "title": title,
                "time_range": time_range,
                "app": app_name,
                "sessions": [],
                "student_count": 0,
                "avg_engagement": 0.0,
                "total_duration": 0.0,
                "date": sess.created_at
            }
        
        grouped_meets[key]["sessions"].append(sess)
        
    # Compute aggregates
    meets_list = []
    for key, data in grouped_meets.items():
        sessions_in_group = data["sessions"]
        data["student_count"] = len(sessions_in_group)
        
        total_eng_pct = 0.0
        total_dur = 0.0
        
        for s in sessions_in_group:
            stats = _compute_session_stats(s.id)
            dur = stats["duration_seconds"]
            if dur > 0:
                eng_pct = (stats["engaged_seconds"] / dur) * 100.0
            else:
                eng_pct = 0.0
            
            total_eng_pct += eng_pct
            total_dur += dur
            
        if data["student_count"] > 0:
            data["avg_engagement"] = total_eng_pct / data["student_count"]
        
        data["total_duration"] = total_dur
        meets_list.append(data)
        
    meets_list.sort(key=lambda x: x["date"], reverse=True)

    return render_template(
        "admin_dashboard.html",
        meets=meets_list,
        filter_type=filter_type,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        username=username
    )


@app.route("/admin/meet-details")
@require_role("admin")
def admin_meet_details():
    """Show student details for a specific meet group."""
    title = request.args.get("title")
    time_range = request.args.get("time_range")
    
    if not title or not time_range:
        flash("Invalid meet details.", "danger")
        return redirect(url_for("admin_dashboard"))
        
    sessions = (
        SessionModel.query
        .filter(SessionModel.meet_title == title)
        .filter(SessionModel.meet_time_range == time_range)
        .all()
    )
    
    rows = []
    for sess in sessions:
        stats = _compute_session_stats(sess.id)
        duration_min = stats["duration_seconds"] / 60.0 if stats["duration_seconds"] else 0.0
        engaged_min = stats["engaged_seconds"] / 60.0
        disengaged_min = stats["disengaged_seconds"] / 60.0
        engagement_pct = (
            100.0 * stats["engaged_seconds"] / stats["duration_seconds"]
            if stats["duration_seconds"] > 0
            else 0.0
        )
        
        rows.append({
            "session": sess,
            "student": sess.student,
            "stats": stats,
            "duration_min": duration_min,
            "engaged_min": engaged_min,
            "disengaged_min": disengaged_min,
            "engagement_pct": engagement_pct,
        })
        
    return render_template(
        "admin_meet_details.html",
        meet_title=title,
        meet_time_range=time_range,
        rows=rows
    )


@app.route("/admin/session/<int:session_id>")
@require_role("admin")
def admin_session_detail(session_id):
    """Detailed view for a single session's engagement timeline."""
    session_model = db.get_or_404(SessionModel, session_id)
    stats = compute_session_stats(session_model)
    return render_template("admin_session_detail.html", session_model=session_model, stats=stats)


@app.route("/admin/teaching-session/<int:id>")
@require_role("admin")
def admin_teaching_session_details(id):
    """Show analytics for a specific teaching session."""
    teaching_session = db.get_or_404(TeachingSession, id)
    
    # Get all student sessions linked to this teaching session
    sessions = SessionModel.query.filter_by(teaching_session_id=id).all()
    
    rows = []
    for sess in sessions:
        stats = _compute_session_stats(sess.id)
        duration_min = stats["duration_seconds"] / 60.0 if stats["duration_seconds"] else 0.0
        engaged_min = stats["engaged_seconds"] / 60.0
        disengaged_min = stats["disengaged_seconds"] / 60.0
        engagement_pct = (
            100.0 * stats["engaged_seconds"] / stats["duration_seconds"]
            if stats["duration_seconds"] > 0
            else 0.0
        )
        
        rows.append({
            "session": sess,
            "student": sess.student,
            "stats": stats,
            "duration_min": duration_min,
            "engaged_min": engaged_min,
            "disengaged_min": disengaged_min,
            "engagement_pct": engagement_pct,
        })
        
    return render_template(
        "admin_teaching_session_details.html",
        teaching_session=teaching_session,
        rows=rows
    )


@app.route("/admin/sessions")
@require_role("admin")
def admin_sessions():
    """Full sessions history with filters for meet_app/title and date range."""
    meet_app = (request.args.get("meet_app") or "").strip() or None
    meet_title = (request.args.get("meet_title") or "").strip()
    start_date_str = (request.args.get("start_date") or "").strip()
    end_date_str = (request.args.get("end_date") or "").strip()

    start_date = _parse_yyyy_mm_dd(start_date_str)
    end_date = _parse_yyyy_mm_dd(end_date_str)

    q = _admin_sessions_filtered_query(meet_app, meet_title or None, start_date, end_date)
    q = q.join(User).filter(User.role == "student")
    q = q.order_by(SessionModel.login_time.desc())

    sessions = q.all()
    rows = []
    for s in sessions:
        rows.append(
            {
                "student": s.student,
                "session": s,
                "stats": _compute_session_stats(s.id),
            }
        )

    return render_template(
        "admin_sessions.html",
        rows=rows,
        meet_app=meet_app or "",
        meet_title=meet_title,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )


@app.route("/admin/sessions/export.csv")
@require_role("admin")
def admin_sessions_export_csv():
    """Export sessions (matching filters) as CSV."""
    meet_app = (request.args.get("meet_app") or "").strip() or None
    meet_title = (request.args.get("meet_title") or "").strip()
    start_date_str = (request.args.get("start_date") or "").strip()
    end_date_str = (request.args.get("end_date") or "").strip()

    start_date = _parse_yyyy_mm_dd(start_date_str)
    end_date = _parse_yyyy_mm_dd(end_date_str)

    q = _admin_sessions_filtered_query(meet_app, meet_title or None, start_date, end_date)
    q = q.join(User).filter(User.role == "student")
    q = q.order_by(SessionModel.login_time.desc())
    sessions = q.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "student_username",
            "meet_app",
            "meet_title",
            "login_time_utc",
            "logout_time_utc",
            "duration_seconds",
            "engaged_seconds",
            "disengaged_seconds",
            "disengage_events",
            "session_id",
        ]
    )

    for s in sessions:
        stats = _compute_session_stats(s.id)
        writer.writerow(
            [
                s.student.username,
                s.meet_app or "",
                s.meet_title or "",
                s.login_time.isoformat() if s.login_time else "",
                s.logout_time.isoformat() if s.logout_time else "",
                int(stats.get("duration_seconds") or 0),
                int(stats.get("engaged_seconds") or 0),
                int(stats.get("disengaged_seconds") or 0),
                int(stats.get("disengage_events") or 0),
                s.id,
            ]
        )

    csv_bytes = output.getvalue().encode("utf-8")
    output.close()

    # Friendly filename based on filters
    safe_app = (meet_app or "all").replace(" ", "_")
    safe_title = (meet_title or "all").strip().replace(" ", "_")
    filename = f"sessions_{safe_app}_{safe_title}.csv"

    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/admin/session/<int:session_id>/delete", methods=["POST"])
@require_role("admin")
def admin_delete_session(session_id):
    """Delete a session and all its related engagement events.
    
    This route safely deletes:
        1. All EngagementEvent records linked to this session_id
        2. The SessionModel record itself
    
    Uses POST to prevent accidental deletion via GET requests.
    """
    session_model = db.get_or_404(SessionModel, session_id)
    student_username = session_model.student.username
    
    # Delete all engagement events for this session
    EngagementEvent.query.filter_by(session_id=session_id).delete()
    
    # Delete the session itself
    db.session.delete(session_model)
    db.session.commit()
    
    flash(f"Session {session_id} for student '{student_username}' has been deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/student/<int:student_id>/delete_all", methods=["POST"])
@require_role("admin")
def admin_delete_all_student_sessions(student_id):
    """Delete all sessions and engagement events for a specific student.
    
    This route safely deletes:
        1. All EngagementEvent records for all sessions of this student
        2. All SessionModel records for this student
    
    Uses POST to prevent accidental deletion via GET requests.
    """
    student = db.get_or_404(User, student_id)
    if student.role != "student":
        flash("Invalid student ID.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    # Get all sessions for this student
    sessions = SessionModel.query.filter_by(student_id=student_id).all()
    session_ids = [s.id for s in sessions]
    
    if not session_ids:
        flash(f"No sessions found for student '{student.username}'.", "info")
        return redirect(url_for("admin_dashboard"))
    
    # Delete all engagement events for these sessions
    EngagementEvent.query.filter(EngagementEvent.session_id.in_(session_ids)).delete()
    
    # Delete all sessions
    SessionModel.query.filter_by(student_id=student_id).delete()
    db.session.commit()
    
    flash(f"All sessions for student '{student.username}' have been deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/analytics")
@require_role("admin")
def admin_analytics():
    """Analytics page with charts showing engagement insights.
    
    This route aggregates data from sessions and engagement_events to provide:
        1. Daily/weekly summary: unique students per day/week
        2. Engagement vs disengagement: total time across all sessions
        3. Student engagement distribution: engagement % per student
    
    All aggregations are computed using SQLAlchemy queries for efficiency.
    """
    # Parse date filter (same logic as dashboard)
    filter_type = request.args.get("filter", "all").lower()
    start_date_str = request.args.get("start_date", "")
    end_date_str = request.args.get("end_date", "")
    
    now = _utc_naive()
    start_date = None
    end_date = None
    
    if filter_type == "today":
        start_date = datetime(now.year, now.month, now.day)
        end_date = now
    elif filter_type == "week":
        start_date = now - timedelta(days=7)
        end_date = now
    elif filter_type == "month":
        start_date = now - timedelta(days=30)
        end_date = now
    elif start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            start_date = None
            end_date = None
    
    # 1. Daily summary: unique students per day
    daily_query = (
        db.session.query(
            func.date(SessionModel.login_time).label("date"),
            func.count(func.distinct(SessionModel.student_id)).label("unique_students"),
        )
        .join(User)
        .filter(User.role == "student")
    )
    if start_date:
        daily_query = daily_query.filter(SessionModel.login_time >= start_date)
    if end_date:
        daily_query = daily_query.filter(SessionModel.login_time <= end_date)
    daily_data = daily_query.group_by(func.date(SessionModel.login_time)).order_by(func.date(SessionModel.login_time)).all()
    
    daily_labels = []
    daily_counts = []
    for row in daily_data:
        date_val = row.date
        if hasattr(date_val, "strftime"):
            label = date_val.strftime("%Y-%m-%d")
        else:
            # Try to parse ISO-like string (e.g. "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS")
            try:
                parsed = datetime.fromisoformat(str(date_val))
                label = parsed.strftime("%Y-%m-%d")
            except Exception:
                # Fallback to the string representation
                label = str(date_val)
        daily_labels.append(label)
        daily_counts.append(row.unique_students)
    
    # 2. Engagement vs disengagement: total time across all sessions
    session_query = SessionModel.query.join(User).filter(User.role == "student")
    if start_date:
        session_query = session_query.filter(SessionModel.login_time >= start_date)
    if end_date:
        session_query = session_query.filter(SessionModel.login_time <= end_date)
    
    sessions = session_query.all()
    total_engaged_seconds = 0.0
    total_disengaged_seconds = 0.0
    
    for sess in sessions:
        stats = compute_session_stats(sess)
        total_engaged_seconds += stats["engaged_seconds"]
        total_disengaged_seconds += stats["disengaged_seconds"]
    
    engagement_labels = ["Engaged", "Disengaged"]
    engagement_data = [
        round(total_engaged_seconds / 3600.0, 2),  # Convert to hours
        round(total_disengaged_seconds / 3600.0, 2),
    ]
    
    # 3. Student engagement distribution: engagement % per student
    students = User.query.filter_by(role="student").all()
    student_names = []
    student_engagement_pcts = []
    
    for student in students:
        student_sessions_query = session_query.filter(SessionModel.student_id == student.id)
        last_session = student_sessions_query.order_by(SessionModel.login_time.desc()).first()
        
        if last_session:
            stats = compute_session_stats(last_session)
            if stats["duration_seconds"] > 0:
                engagement_pct = 100.0 * stats["engaged_seconds"] / stats["duration_seconds"]
                student_names.append(student.username)
                student_engagement_pcts.append(round(engagement_pct, 1))
    
    return render_template(
        "admin_analytics.html",
        filter_type=filter_type,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        daily_labels=daily_labels,
        daily_counts=daily_counts,
        engagement_labels=engagement_labels,
        engagement_data=engagement_data,
        student_names=student_names,
        student_engagement_pcts=student_engagement_pcts,
    )


###############################################################################
# Backwards‑compatible /login route
###############################################################################


@app.route("/login", methods=["GET", "POST"])
def legacy_login():
    """Keep /login working by redirecting to the student login form."""
    if request.method == "POST":
        return student_login()
    return redirect(url_for("student_login"))


###############################################################################
# Entry Point & DB Init
###############################################################################


def init_db():
    """Create all database tables on first run and apply lightweight migrations.

    This project uses a single SQLite file without Alembic migrations.
    To keep existing `engagement.db` working when we add new columns, we do a
    small, safe migration step that adds missing columns.
    """
    db.create_all()
    _ensure_sqlite_columns()

    # Seed default login accounts (only if missing).
    # This ensures the app remains usable even after first-run migrations.
    try:
        if not User.query.filter_by(username="student1").first():
            u = User(
                username="student1",
                password_hash=generate_password_hash("test123"),
                role="student",
            )
            db.session.add(u)

        if not User.query.filter_by(username="admin1").first():
            a = User(
                username="admin1",
                password_hash=generate_password_hash("admin123"),
                role="admin",
            )
            db.session.add(a)

        db.session.commit()
    except Exception:
        db.session.rollback()


def _ensure_sqlite_columns():
    """Add newly introduced columns to existing SQLite databases.

    SQLite supports `ALTER TABLE ... ADD COLUMN`.
    We use it to add `meet_app`, `meet_title`, and `created_at` if they are missing.
    """
    try:
        engine = db.engine
        if engine.dialect.name != "sqlite":
            return

        def has_column(table, col):
            rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).fetchall()
            existing = {r[1] for r in rows}
            return col in existing

        alters = []
        if not has_column("sessions", "meet_app"):
            alters.append("ALTER TABLE sessions ADD COLUMN meet_app VARCHAR(50)")
        if not has_column("sessions", "meet_title"):
            alters.append("ALTER TABLE sessions ADD COLUMN meet_title VARCHAR(200)")
        if not has_column("sessions", "created_at"):
            alters.append("ALTER TABLE sessions ADD COLUMN created_at DATETIME")

        if not has_column("engagement_events", "reason"):
            alters.append("ALTER TABLE engagement_events ADD COLUMN reason VARCHAR(255)")
        if not has_column("engagement_events", "confidence"):
            alters.append("ALTER TABLE engagement_events ADD COLUMN confidence FLOAT")

        for stmt in alters:
            db.session.execute(db.text(stmt))
        if alters:
            # Backfill created_at for existing rows (best-effort)
            db.session.execute(
                db.text("UPDATE sessions SET created_at = COALESCE(created_at, login_time)")
            )
            db.session.commit()
    except Exception:
        # Don't hard-fail app boot if migration can't run.
        db.session.rollback()


def _generate_session_code():
    """Generate a random 8-character alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=8))


@app.route("/admin/teaching-sessions")
@require_role("admin")
def admin_teaching_sessions():
    """List all admin-created teaching sessions."""
    sessions = TeachingSession.query.order_by(TeachingSession.date.desc(), TeachingSession.start_time.desc()).all()
    return render_template("admin_teaching_sessions.html", sessions=sessions)


@app.route("/admin/teaching-sessions/create", methods=["GET", "POST"])
@require_role("admin")
def admin_create_teaching_session():
    """Create a new teaching session."""
    if request.method == "POST":
        meet_app = request.form.get("meet_app")
        meet_title = request.form.get("meet_title")
        date_str = request.form.get("date")
        start_time_str = request.form.get("start_time")
        end_time_str = request.form.get("end_time")
        
        # Handle "Others" case for meet_app
        if meet_app == "Others":
            custom_app = request.form.get("custom_meet_app", "").strip()
            if custom_app:
                meet_app = custom_app
        
        if not all([meet_app, meet_title, date_str, start_time_str, end_time_str]):
            flash("All fields are required.", "danger")
            return redirect(url_for("admin_create_teaching_session"))
            
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            start_time_obj = datetime.strptime(start_time_str, "%H:%M").time()
            end_time_obj = datetime.strptime(end_time_str, "%H:%M").time()
        except ValueError:
            flash("Invalid date or time format.", "danger")
            return redirect(url_for("admin_create_teaching_session"))
            
        # Generate unique code
        code = _generate_session_code()
        while TeachingSession.query.filter_by(session_code=code).first():
            code = _generate_session_code()
            
        new_session = TeachingSession(
            session_code=code,
            meet_app=meet_app,
            meet_title=meet_title,
            date=date_obj,
            start_time=start_time_obj,
            end_time=end_time_obj,
            created_by_admin_id=session.get("user_id")
        )
        
        db.session.add(new_session)
        db.session.commit()
        
        flash(f"Session created successfully! Code: {code}", "success")
        return redirect(url_for("admin_teaching_sessions"))
        
    return render_template("admin_create_teaching_session.html")


if __name__ == "__main__":
    with app.app_context():
        init_db()
    print(f"[SmartEye] Database location: {DB_PATH}")
    print(
        "[SmartEye] Default login — student: student1/test123 | admin: admin1/admin123"
    )
    print(f"[SmartEye] EAR_THRESHOLD set to: {EAR_THRESHOLD:.2f}")
    app.run(debug=True)


