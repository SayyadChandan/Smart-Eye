import pytest

import os
from datetime import datetime, timedelta, timezone


@pytest.fixture()
def app_instance(tmp_path):
    """Import the Flask app with a temporary SQLite DB.

    This keeps tests isolated from the project's real `engagement.db`.
    """
    os.environ["TE_TESTING"] = "1"
    os.environ["TE_DB_PATH"] = str(tmp_path / "test.db")

    import app as teacher_eye_app

    # Redirect DB (for safety) after import, too.
    teacher_eye_app.app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
    )

    with teacher_eye_app.app.app_context():
        teacher_eye_app.db.drop_all()
        teacher_eye_app.db.create_all()

        # Create a couple of users
        teacher_eye_app.create_user("admin1", "pass", "admin")
        teacher_eye_app.create_user("student1", "pass", "student")

        # Create a teaching session so students can join and start sessions.
        # This matches the current application flow:
        #   /student/join-session -> sets pending_teaching_session_id
        #   /student/start-session -> creates SessionModel row and redirects to dashboard
        admin_user = teacher_eye_app.User.query.filter_by(username="admin1").first()
        assert admin_user is not None

        now_utc = datetime.now(timezone.utc)
        teaching_session = teacher_eye_app.TeachingSession(
            session_code="TESTCODE",
            meet_app="Zoom",
            meet_title="Math Class",
            date=now_utc.date(),
            start_time=now_utc.time(),
            end_time=(now_utc + timedelta(minutes=30)).time(),
            created_by_admin_id=admin_user.id,
        )
        teacher_eye_app.db.session.add(teaching_session)
        teacher_eye_app.db.session.commit()

    yield teacher_eye_app.app


@pytest.fixture()
def client(app_instance):
    return app_instance.test_client()


def test_home_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_register_loads(client):
    resp = client.get("/register")
    assert resp.status_code == 200


def test_admin_login_loads(client):
    resp = client.get("/admin/login")
    assert resp.status_code == 200


def test_student_login_loads(client):
    resp = client.get("/student/login")
    assert resp.status_code == 200


def test_protected_routes_redirect_when_logged_out(client):
    # student dashboard
    resp = client.get("/student/dashboard", follow_redirects=False)
    assert resp.status_code in (301, 302)

    # admin dashboard
    resp = client.get("/admin/dashboard", follow_redirects=False)
    assert resp.status_code in (301, 302)


def test_admin_realtime_dashboard_requires_admin(client):
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code in (301, 302)


def test_admin_register_requires_passkey(client, monkeypatch):
    # Set required passkey for this test
    # We use env override to avoid depending on local admin_passkey.txt.
    monkeypatch.setenv("TE_ADMIN_PASSKEY", "secretpass")

    # Without passkey -> should not redirect to admin login
    resp = client.post(
        "/register",
        data={
            "username": "admin2",
            "role": "admin",
            "password": "pass",
            "confirm_password": "pass",
            "admin_passkey": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 200

    # Wrong passkey -> should also stay on page
    resp = client.post(
        "/register",
        data={
            "username": "admin3",
            "role": "admin",
            "password": "pass",
            "confirm_password": "pass",
            "admin_passkey": "wrong",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 200

    # Correct passkey -> should redirect to admin login
    resp = client.post(
        "/register",
        data={
            "username": "admin4",
            "role": "admin",
            "password": "pass",
            "confirm_password": "pass",
            "admin_passkey": "secretpass",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (301, 302)
    assert "/admin/login" in resp.headers.get("Location", "")


def test_admin_register_uses_passkey_file(client, tmp_path, monkeypatch):
    # Ensure env doesn't override
    monkeypatch.delenv("TE_ADMIN_PASSKEY", raising=False)

    # Write a temporary passkey file and point app's BASE_DIR to it by monkeypatching path read.
    # We can't easily change BASE_DIR, so instead we create the file in the real project root.
    # This is safe in CI/local because it's just a test value and gitignored.
    passkey_path = os.path.join(os.path.dirname(__file__), "..", "admin_passkey.txt")
    passkey_path = os.path.abspath(passkey_path)
    with open(passkey_path, "w", encoding="utf-8") as f:
        f.write("# test key\nfilekey\n")

    resp = client.post(
        "/register",
        data={
            "username": "admin_file",
            "role": "admin",
            "password": "pass",
            "confirm_password": "pass",
            "admin_passkey": "filekey",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (301, 302)
    assert "/admin/login" in resp.headers.get("Location", "")


def test_student_login_redirects_to_start_session(client):
    resp = client.post(
        "/student/login",
        data={"username": "student1", "password": "pass"},
        follow_redirects=False,
    )
    assert resp.status_code in (301, 302)
    assert "/student/join-session" in resp.headers.get("Location", "")


def test_student_start_session_creates_session(client):
    # Login first
    resp = client.post(
        "/student/login",
        data={"username": "student1", "password": "pass"},
        follow_redirects=False,
    )
    assert resp.status_code in (301, 302)

    # Join the teaching session (sets pending_teaching_session_id in the session)
    resp2 = client.post(
        "/student/join-session",
        data={"session_code": "TESTCODE"},
        follow_redirects=False,
    )
    assert resp2.status_code in (301, 302)

    # Now start the session (pending_teaching_session_id is present)
    resp3 = client.post("/student/start-session", data={}, follow_redirects=False)
    assert resp3.status_code in (301, 302)
    assert "/student/dashboard" in resp3.headers.get("Location", "")


def test_admin_sessions_requires_login(client):
    resp = client.get("/admin/sessions", follow_redirects=False)
    assert resp.status_code in (301, 302)
    assert "/admin/login" in resp.headers.get("Location", "")


def test_admin_sessions_csv_requires_login(client):
    resp = client.get("/admin/sessions/export.csv", follow_redirects=False)
    assert resp.status_code in (301, 302)
    assert "/admin/login" in resp.headers.get("Location", "")
