## APPENDIX — CODE ATTACHMENT (Smart Eye)

This appendix provides the important source files and key code excerpts used in the Smart Eye project.

### A.1 Project Dependencies
File: `requirements.txt`

```text
Flask==3.0.3
Flask-SQLAlchemy==3.1.1
opencv-python==4.10.0.84
numpy==1.26.4
mediapipe
pytest==8.3.4
```

---

### A.2 Backend (Flask) — `app.py`
File: `Smart Eye/app.py`

#### A.2.1 Database Path (Persistent SQLite)
Ensures DB location is stable across restarts.

```python
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (BASE_DIR / "smart_eye.db").resolve()
LEGACY_DB_PATH = (BASE_DIR / "engagement.db").resolve()

_env_db_path = os.getenv("TE_DB_PATH")
if _env_db_path:
    DB_PATH = Path(_env_db_path)
    if not DB_PATH.is_absolute():
        DB_PATH = (BASE_DIR / DB_PATH).resolve()
else:
    DB_PATH = LEGACY_DB_PATH if LEGACY_DB_PATH.exists() else DEFAULT_DB_PATH

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
```

#### A.2.2 Models
Key tables: users, sessions, engagement events.

```python
class EngagementEvent(db.Model):
    __tablename__ = "engagement_events"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    engagement_state = db.Column(db.String(20), nullable=False)  # engaged/disengaged/unknown/no_face
    score = db.Column(db.Float, nullable=False)
    reason = db.Column(db.String(255), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
```

#### A.2.3 Eye Aspect Ratio (EAR)
MediaPipe Face Mesh indices used for EAR.

```python
left_idxs = [362, 385, 387, 263, 373, 380]
right_idxs = [33, 160, 158, 133, 153, 144]
```

#### A.2.4 Head Pose Normalization (solvePnP flip fix)

```python
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
```

#### A.2.5 Baseline Calibration (3 frames)
Baseline is stored in `calibration_store` keyed by student user id.

```python
calibration_store = {}  # keyed by user_id (fallback to session_id)
CALIB_NEEDED = 3

if not calib_state["baseline_ready"]:
    calib_state["calib_yaw_samples"].append(yaw)
    calib_state["calib_pitch_samples"].append(pitch)
    if len(calib_state["calib_yaw_samples"]) >= CALIB_NEEDED:
        calib_state["baseline_yaw"] = sum(calib_state["calib_yaw_samples"]) / CALIB_NEEDED
        calib_state["baseline_pitch"] = sum(calib_state["calib_pitch_samples"]) / CALIB_NEEDED
        calib_state["baseline_ready"] = True
```

#### A.2.6 Reset Calibration Endpoint

```python
@app.route("/api/engagement/reset_baseline", methods=["POST"])
@require_role("student")
def api_reset_baseline():
    session_id = session.get("session_id")
    user_id = session.get("user_id")
    calib_key = user_id if user_id is not None else session_id
    calibration_store[calib_key] = {
        "baseline_yaw": None,
        "baseline_pitch": None,
        "baseline_ready": False,
        "calib_yaw_samples": [],
        "calib_pitch_samples": [],
    }
    return jsonify({"status": "reset"})
```

#### A.2.7 Saving Engagement Events + Live Stats

```python
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

stats = _compute_session_stats(session_id)
```

---

### A.3 Student Frontend — `templates/student_dashboard.html`
Contains:
- Webcam video
- Calibration progress bar
- Reset calibration button
- Engagement time counters

---

### A.4 Student Frontend JS — `static/js/student_engagement.js`
Key features:
- Captures webcam frames
- Sends frames to `/api/engagement`
- Updates status badge, gauge, and time stats
- Reset baseline button handler

---

### A.5 Admin Dashboard — `templates/dashboard.html` + `static/js/realtime.js`
Key features:
- Captures frames and sends to `/detect`
- Renders student cards per detected face
- Class engagement summary bar (green/yellow/red)
- Low attention alert (rolling last 5 scores, beep + badge)

---

### A.6 How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run:
```bash
python app.py
```

3. Open:
- `http://127.0.0.1:5000/`

