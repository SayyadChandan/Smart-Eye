import numpy as np


def _dummy_landmarks(n: int = 300):
    # Only used as a non-None placeholder; landmark access is mocked in tests.
    class L:
        x = 0.5
        y = 0.5

    return [L() for _ in range(n)]


def test_no_face_returns_not_engaged_and_zero_confidence(monkeypatch):
    import app as teacher_eye_app

    teacher_eye_app.EYE_CLOSED_COUNTERS.clear()
    teacher_eye_app.ENGAGEMENT_HISTORY.clear()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    score, status, reason, confidence, _signals = teacher_eye_app.get_engagement_status(
        "s_no_face", img, None
    )

    assert score == 0.0
    assert status == "Not Engaged"
    assert reason == "No Face"
    assert confidence == 0.0


def test_eyes_closed_triggers_not_engaged(monkeypatch):
    import app as teacher_eye_app

    teacher_eye_app.EYE_CLOSED_COUNTERS.clear()
    teacher_eye_app.ENGAGEMENT_HISTORY.clear()
    teacher_eye_app.LOOKING_AWAY_COUNTERS.clear()
    teacher_eye_app.LOOKING_DOWN_COUNTERS.clear()
    teacher_eye_app.calibration_store.clear()

    # Mock EAR to be consistently below threshold.
    def fake_calculate_ear(landmarks, image_shape):
        return 0.0, 0.0, teacher_eye_app.EAR_THRESHOLD - 0.05

    # Mock head pose to be looking forward.
    def fake_estimate_head_pose(image, landmarks):
        return 0.0, 0.0, 0.0  # yaw, pitch, roll

    monkeypatch.setattr(teacher_eye_app, "calculate_ear", fake_calculate_ear)
    monkeypatch.setattr(teacher_eye_app, "estimate_head_pose", fake_estimate_head_pose)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = _dummy_landmarks()

    # Pre-seed baseline as "forward/engaged" so we are past calibration.
    teacher_eye_app.calibration_store["s_eyes_closed"] = {
        "baseline_yaw": 0.0,
        "baseline_pitch": 0.0,
        "baseline_ready": True,
        "calib_yaw_samples": [0.0, 0.0, 0.0],
        "calib_pitch_samples": [0.0, 0.0, 0.0],
    }

    # UNCERTAINTY_MIN_FRAMES is 5, so on the 5th frame we should not return "Unknown".
    for i in range(5):
        score, status, reason, confidence, _signals = teacher_eye_app.get_engagement_status(
            "s_eyes_closed", img, landmarks
        )

    assert status == "Not Engaged"
    # Mean EAR is EAR_THRESHOLD - 0.05 => EAR: 0.15 (with defaults).
    assert reason.startswith("Eyes closed (EAR:")


def test_looking_away_triggers_not_engaged(monkeypatch):
    import app as teacher_eye_app

    teacher_eye_app.EYE_CLOSED_COUNTERS.clear()
    teacher_eye_app.ENGAGEMENT_HISTORY.clear()
    teacher_eye_app.LOOKING_AWAY_COUNTERS.clear()
    teacher_eye_app.LOOKING_DOWN_COUNTERS.clear()
    teacher_eye_app.calibration_store.clear()

    # Mock EAR to be open (above threshold).
    def fake_calculate_ear(landmarks, image_shape):
        return 0.0, 0.0, teacher_eye_app.EAR_THRESHOLD + 0.2

    # Mock head pose to look away.
    def fake_estimate_head_pose(image, landmarks):
        return 30.0, 0.0, 0.0  # yaw, pitch, roll

    monkeypatch.setattr(teacher_eye_app, "calculate_ear", fake_calculate_ear)
    monkeypatch.setattr(teacher_eye_app, "estimate_head_pose", fake_estimate_head_pose)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = _dummy_landmarks()

    # Pre-seed baseline as "forward/engaged" so yaw_delta is large.
    teacher_eye_app.calibration_store["s_looking_away"] = {
        "baseline_yaw": 0.0,
        "baseline_pitch": 0.0,
        "baseline_ready": True,
        "calib_yaw_samples": [0.0, 0.0, 0.0],
        "calib_pitch_samples": [0.0, 0.0, 0.0],
    }

    for i in range(5):
        score, status, reason, confidence, _signals = teacher_eye_app.get_engagement_status(
            "s_looking_away", img, landmarks
        )

    # With the updated rule, head pose looking-away must force Not Engaged
    # even when eyes are open.
    assert status == "Not Engaged"


def test_low_attention_when_recent_history_has_looking_away(monkeypatch):
    import app as teacher_eye_app

    teacher_eye_app.EYE_CLOSED_COUNTERS.clear()
    teacher_eye_app.ENGAGEMENT_HISTORY.clear()
    teacher_eye_app.LOOKING_AWAY_COUNTERS.clear()
    teacher_eye_app.LOOKING_DOWN_COUNTERS.clear()
    teacher_eye_app.calibration_store.clear()

    # Mock EAR to be open.
    def fake_calculate_ear(landmarks, image_shape):
        return 0.0, 0.0, teacher_eye_app.EAR_THRESHOLD + 0.2

    # Alternate yaw: first 4 frames look away, 5th frame looks forward.
    state = {"frame": 0}

    def fake_estimate_head_pose(image, landmarks):
        state["frame"] += 1
        if state["frame"] <= 4:
            return 30.0, 0.0, 0.0  # yaw, pitch, roll
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(teacher_eye_app, "calculate_ear", fake_calculate_ear)
    monkeypatch.setattr(teacher_eye_app, "estimate_head_pose", fake_estimate_head_pose)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = _dummy_landmarks()

    # Pre-seed baseline as "forward/engaged" so yaw_delta is computed relative to 0.
    teacher_eye_app.calibration_store["s_low_attention"] = {
        "baseline_yaw": 0.0,
        "baseline_pitch": 0.0,
        "baseline_ready": True,
        "calib_yaw_samples": [0.0, 0.0, 0.0],
        "calib_pitch_samples": [0.0, 0.0, 0.0],
    }

    status = None
    reason = None
    for i in range(5):
        score, status, reason, confidence, _signals = teacher_eye_app.get_engagement_status(
            "s_low_attention", img, landmarks
        )

    # With open eyes, looking-away should not force Not Engaged.
    assert status == "Engaged"


def test_looking_down_pitch_forces_not_engaged(monkeypatch):
    import app as teacher_eye_app

    teacher_eye_app.EYE_CLOSED_COUNTERS.clear()
    teacher_eye_app.ENGAGEMENT_HISTORY.clear()
    teacher_eye_app.LOOKING_AWAY_COUNTERS.clear()
    teacher_eye_app.LOOKING_DOWN_COUNTERS.clear()
    teacher_eye_app.calibration_store.clear()

    # Mock EAR to be open (above threshold).
    def fake_calculate_ear(landmarks, image_shape):
        return 0.0, 0.0, teacher_eye_app.EAR_THRESHOLD + 0.2

    # Mock head pose to look down.
    def fake_estimate_head_pose(image, landmarks):
        return 0.0, 20.0, 0.0  # yaw, pitch, roll

    monkeypatch.setattr(teacher_eye_app, "calculate_ear", fake_calculate_ear)
    monkeypatch.setattr(teacher_eye_app, "estimate_head_pose", fake_estimate_head_pose)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = _dummy_landmarks()

    # Pre-seed baseline as "forward/engaged" so pitch_delta is large.
    teacher_eye_app.calibration_store["s_phone_down"] = {
        "baseline_yaw": 0.0,
        "baseline_pitch": 0.0,
        "baseline_ready": True,
        "calib_yaw_samples": [0.0, 0.0, 0.0],
        "calib_pitch_samples": [0.0, 0.0, 0.0],
    }

    # LOOKING_DOWN_CONSEC_FRAMES defaults to 2, so on the 2nd frame it should disengage.
    status = None
    for _ in range(2):
        _score, status, _reason, _confidence, _signals = teacher_eye_app.get_engagement_status(
            "s_phone_down", img, landmarks
        )

    assert status == "Not Engaged"

