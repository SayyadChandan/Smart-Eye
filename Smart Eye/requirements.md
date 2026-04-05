# Project: Teacher Eye – AI‑Powered Student Engagement Monitor

## Goal
Build a Python Flask web application that monitors student engagement in online classes using computer vision. It should run locally on my machine.

## Tech Stack
- Backend: Python 3, Flask
- Frontend: HTML, CSS, JavaScript (no React)
- CV/ML: MediaPipe Face Mesh (landmarks), OpenCV (solvePnP + optional face boxes), NumPy
- Database: SQLite via Flask-SQLAlchemy (users, sessions, engagement events)

## Features

### 1. Authentication (Simple)
- Pages: Register, Login, Logout
- Store users in SQLite with hashed passwords
- After login, redirect to Dashboard

### 2. Dashboard (Main Page)
- Show:
  - Live webcam feed (from browser)
  - Class engagement percentage (0–100%)
  - Number of students detected
  - List of each detected face with:
    - ID
    - Engagement status: "Engaged" or "Not Engaged"
    - Engagement score (0–1, float)

### 3. Engagement Detection Logic
- Student stream:
  - Student dashboard captures frames periodically (every ~2 seconds) and POSTs to `/api/engagement`
  - Backend decodes the image, runs **MediaPipe Face Mesh** landmarks, and computes:
    - **EAR** (eye-closed detection)
    - **Head pose** (yaw/pitch solvePnP) for looking-away detection
    - Temporal smoothing + deterministic scoring
  - Response includes `engagement_score`, `status` (`Engaged|Not Engaged|Unknown`), `reason`, and `confidence`
- Admin/class stream:
  - Admin realtime webcam captures frames periodically (every ~1 second) and POSTs to `/detect`
  - Backend detects face boxes using OpenCV and then runs the same MediaPipe engagement logic per face crop
  - Response includes `class_engagement` (mean score) and per-face student statuses

### 4. Frontend UI
- Use Bootstrap or clean CSS
- Layout:
  - Left: video feed
  - Right: stats card (class engagement, number of students)
  - Below stats: list of students with status and score
- Color coding:
  - Engaged: green background
  - Not Engaged: red/pink background

### 5. Project Structure
- `app.py` – Flask app with routes:
  - `/` – home/landing page
  - `/register` – register page
  - `/login` – legacy redirect to student login
  - `/student/login`, `/student/join-session`, `/student/start-session`, `/student/dashboard`, `/student/logout`
  - `/admin/login`, `/admin/dashboard`, `/admin/analytics`, `/admin/logout`
  - `/dashboard` – admin real-time class webcam view
  - `/api/engagement` – student frame endpoint
  - `/detect` – admin per-frame endpoint
- `templates/`:
  - `base.html`, `home.html`, `register.html`, `login.html`, `dashboard.html`
- `static/css/style.css` – basic styling
- `static/js/realtime.js` – webcam capture + AJAX to `/detect`
- `users.json` – stores users
- `requirements.txt` – all dependencies

### 6. Constraints
- The app must run with:
  - `pip install -r requirements.txt`
  - `python app.py`
- No external paid APIs
- Keep the ML part simple and deterministic (no model training required)

## Deliverables
- All code files
- Clear instructions in README: how to install and run
- Clean, commented code suitable for an MCA 4th sem ML/DL project
