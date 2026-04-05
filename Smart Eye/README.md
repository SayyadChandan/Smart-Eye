# Smart-Eye

Repository: [github.com/SayyadChandan/Smart-Eye](https://github.com/SayyadChandan/Smart-Eye)

## Teacher Eye – AI‑Powered Student Engagement Monitor

Teacher Eye is a **Python Flask** web application that demonstrates how **computer vision** can be used to monitor student engagement in online classes. It uses a **webcam feed in the browser**, sends frames to the backend using **base64 images**, and applies **MediaPipe Face Mesh (landmarks)** + **head pose** + **EAR (eye aspect ratio)** to estimate engagement (privacy-friendly: raw video is not stored).

This project is designed to be a **clean, well‑commented MCA 4th semester ML/DL mini‑project** with a simple but complete end‑to‑end pipeline.

---

### 1. Project Features

- **Authentication (Students + Admins)**
  - **Register / Login / Logout** backed by **SQLite** (`engagement.db`) via **Flask-SQLAlchemy**
  - Passwords are stored as **secure hashes** (not plain text)
  - Admin self-registration can be protected with an **admin passkey** (`admin_passkey.txt` or `TE_ADMIN_PASSKEY`)
- **Dashboards**
  - **Student dashboard** shows the student's **current engagement**
  - **Admin analytics dashboard** shows historical sessions & engagement summaries
  - Optional **admin real-time class view** renders a webcam dashboard using `/dashboard` + `/detect`
- **Engagement Detection Logic**
  - **Student endpoint:** `/api/engagement` (POST)
    - Receives frames as **base64**
    - Returns `{"engagement_score": ..., "status": "Engaged|Not Engaged|Unknown", "reason": "...", "confidence": ..., "stats": {...}}`
  - **Admin/class endpoint:** `/detect` (POST)
    - Receives frames as **base64**
    - Detects faces, runs the same MediaPipe-based engagement logic per face, and returns `{"students": [...], "class_engagement": 0.78, "total_students": 3}`
- **Frontend UI**
  - Built with **HTML + Bootstrap + CSS + JavaScript**
  - Layout:
    - **Left:** webcam video feed
    - **Right:** class stats
    - **Below:** list of students with status and score
  - Color coding:
    - **Green** background → Engaged
    - **Red / Pink** background → Not Engaged

---

### 2. Project Structure

- `app.py` – Flask app with routes:
  - `/` – Home / landing page
  - `/register` – Register page
  - `/student/login` – Student login
  - `/student/dashboard` – Student dashboard (requires student login)
  - `/student/logout` – Student logout
  - `/admin/login` – Admin login
  - `/admin/dashboard` – Admin sessions dashboard (requires admin login)
  - `/admin/analytics` – Admin analytics
  - `/admin/logout` – Admin logout
  - `/dashboard` – Admin real-time class dashboard (requires admin login)
  - `/detect` – POST endpoint used by the real-time admin class dashboard
  - `/api/engagement` – POST endpoint used by student dashboard
- `templates/`
  - `base.html` – Base template with navbar & layout
  - `home.html` – Project introduction and links to login/register
  - `register.html` – User registration form
  - `login.html` – User login form
  - `dashboard.html` – Main live engagement dashboard
- `static/css/style.css` – Custom styling on top of Bootstrap
- `static/js/realtime.js` – Webcam capture + AJAX to `/detect`
- `engagement.db` – SQLite database storing users/sessions/engagement events
- `requirements.txt` – Python dependencies

---

### 3. Prerequisites

- **Operating System:** Windows 10/11 (tested), but should work on Linux/macOS too
- **Python:** 3.9 or newer recommended
- **Pip:** comes with Python (for installing packages)
- A **webcam** (built‑in or external)
- A **modern browser** (Chrome / Edge / Firefox) with webcam support

---

### 4. Create and Activate a Virtual Environment (Windows)

It is **strongly recommended** to use a virtual environment so that project dependencies do not conflict with other Python projects on your system.

Open **PowerShell** in the project folder (for example: `D:\Smart Eye`) and run:

```bash
cd "D:\Smart Eye"

# Create virtual environment named "venv"
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

After activation, your prompt should show `(venv)` at the beginning.  
Whenever you want to work on this project in the future, **activate the venv again** with:

```bash
cd "D:\Smart Eye"
.venv\Scripts\activate
```

To **deactivate** the virtual environment later:

```bash
deactivate
```

---

### 5. Install Dependencies

With the virtual environment **activated**, install the required libraries from `requirements.txt`:

```bash
cd "D:\Smart Eye"
.venv\Scripts\activate   # if not already active

pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

- `Flask` – backend web framework
- `opencv-python` – OpenCV (computer vision)
- `numpy` – numerical library used by OpenCV

---

### 6. Run the Project

With the virtual environment active and dependencies installed:

```bash
cd "D:\Smart Eye"
.venv\Scripts\activate   # if needed

python app.py
```

You should see output similar to:

```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000/  (Press CTRL+C to quit)
```

---

### 7. Open in Browser

After running `python app.py`, open your browser and go to:

- **URL:** `http://127.0.0.1:5000/`

Flow to test:

1. Go to `http://127.0.0.1:5000/`
2. Click **Register** and create a test user.
3. Login with that user.
4. You will be redirected to **Dashboard**.
5. Allow the browser to **access your webcam**.
6. You should start seeing:
   - The webcam feed on the left
   - Engagement percentage and student count on the right
   - List of detected faces, with **Engaged / Not Engaged** status and scores.

---

### 8. ML / DL / CV Explanation (for Viva / Report)

- **Landmark model:**
  - **MediaPipe Face Mesh** with `refine_landmarks=True` (returns dense facial landmarks for geometry).
- **Signals extracted (rule-based, deterministic):**
  - **Eye state:** compute **EAR (Eye Aspect Ratio)** from eye landmarks.
    - If EAR stays below `EAR_THRESHOLD` for `CLOSED_EAR_CONSEC_FRAMES` consecutive frames → **Eyes Closed**
  - **Head pose:** estimate **yaw/pitch/roll** using **solvePnP** (3D face model + key landmarks), then threshold:
    - `abs(yaw) > 20°` or `abs(pitch) > 15°` → **Looking Away**
- **Engagement decision:**
  - No face landmarks → **Not Engaged** (`reason: "No Face"`)
  - Eyes closed → **Not Engaged** (`reason: "Eyes Closed"`)
  - Looking away → **Not Engaged** (`reason: "Looking Away"`)
  - Otherwise compute a per-frame score in `[0,1]` and apply **temporal smoothing** over the last `SMOOTHING_WINDOW` observations.
- **Confidence / Unknown:**
  - Early frames (low confidence) are returned as `status: "Unknown"` (grey UI) and are **not stored** in SQLite analytics.
- **Admin class view:**
  - OpenCV detects face bounding boxes in the admin frame.
  - MediaPipe engagement is computed per detected face crop.
  - `class_engagement` is the mean of per-face scores.

This approach is still easy to explain in viva: the ML component is the landmark detector, while the engagement output is an interpretable ruleset with smoothing.

---

### 9. Where to Look in Code

- **Flask app & routes:** `app.py`
  - Authentication helpers
  - `/detect` endpoint
  - Engagement score computation with MediaPipe Face Mesh (EAR + head pose)
- **Templates (HTML):** `templates/`
  - `dashboard.html` – where video and stats are rendered
- **Frontend JS (webcam + AJAX):** `static/js/realtime.js`
- **Styling:** `static/css/style.css`
- **User storage:** SQLite (`engagement.db`) via Flask-SQLAlchemy

You can extend this project by:

- Storing engagement history per session
- Plotting engagement over time (charts)
- Adding more advanced ML models for emotion or attention detection

---

### 10. Quick Summary for Running

1. **Create venv (once):**

   ```bash
   cd D:\ML
   python -m venv venv
   ```

2. **Activate venv:**

   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Flask app:**

   ```bash
   python app.py
   ```

5. **Open in browser:**
   - `http://127.0.0.1:5000/`

