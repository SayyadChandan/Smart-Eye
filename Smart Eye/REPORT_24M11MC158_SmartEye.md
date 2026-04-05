## SMART EYE: AI‑Powered Student Engagement Monitoring System (Web Application)

**A project report submitted in partial fulfillment of the requirements for the award of the Degree of**

**MASTER OF COMPUTER APPLICATIONS**

**Submitted by**  
**Sayyad Chandan**  
**24M11MC158**

**Under the Guidance of**  
**Mr. K. Nagababu**

Department of Computer Applications  
School of Engineering  
Aditya University, Surampalem, Andhra Pradesh, India  
2024–2026

---

### CERTIFICATE
This is to certify that the project work entitled **SMART EYE: AI‑Powered Student Engagement Monitoring System (Web Application)** has been submitted by **Sayyad Chandan (24M11MC158)** in partial fulfillment of the requirements for the award of **Master of Computer Applications** degree.

**PROJECT GUIDE**: Mr. K. Nagababu  
**HEAD OF THE DEPARTMENT**: ______________________

---

### DECLARATION
I hereby declare that the project entitled **SMART EYE: AI‑Powered Student Engagement Monitoring System (Web Application)** is original and has been submitted to **Aditya University, Surampalem**, in partial fulfillment of the requirements for the award of **Master of Computer Applications** degree. I further declare that this project work has not been submitted in full or in part for the award of any degree in this or any other institution and confirm that this work complies with the institutional Academic Integrity and AI Usage Policy.

**Place**: Surampalem  
**Date**: ___________  
**Sayyad Chandan**  
**24M11MC158**

---

### ACKNOWLEDGEMENT
I express my sincere gratitude to the Department of Computer Applications, Aditya University, for providing the academic environment and resources required to complete this project.

I would like to thank my project guide **Mr. K. Nagababu** for his guidance, suggestions, and continuous support throughout the development of this project. His feedback helped in improving the system design and implementation quality.

I also thank the faculty members and technical staff of the department for their help. Finally, I thank my parents and friends for their encouragement during the project.

**Sayyad Chandan**

---

### ABSTRACT
Online learning environments require new ways to understand whether students are attentive and participating effectively. Traditional approaches such as manual observation or simple attendance tracking do not capture the quality of engagement during a live class. **Smart Eye** is a computer‑vision‑based web application that estimates student engagement in real time using webcam frames, without storing raw video. The system uses **MediaPipe Face Mesh** to extract facial landmarks and derives interpretable signals such as **Eye Aspect Ratio (EAR)** for eye state, **head pose (yaw/pitch)** for gaze direction, and an **occlusion indicator** for phone/object interference. A lightweight rule‑based model converts these signals into an **engagement status**, a numeric **engagement score**, an explanation **reason**, and a **confidence** value.

Smart Eye is implemented using **Python Flask** for the backend, **OpenCV** for image decoding and geometric operations, and **SQLite (Flask‑SQLAlchemy)** for persistent storage of users, sessions, and engagement events. The frontend is built with **HTML + Bootstrap + JavaScript**, capturing frames via `getUserMedia()` and updating dashboards using JSON APIs. The system includes student and admin workflows, session analytics, CSV export, fairness handling for “No Face Detected” frames, calibration reset, and an admin real‑time class dashboard that processes multiple faces independently.

---

### CONTENTS
Chapter 1: INTRODUCTION  
1.1 Brief Information about the Project  
1.2 Motivation and Contribution of Project  
1.3 Objective of the Project  
1.4 Scope of the Project  

Chapter 2: LITERATURE SURVEY  
2.1 Overview of the Existing Systems  
2.2 Related Work  
2.3 Research Gap Identified  
2.4 Contribution Over Existing Work  

Chapter 3: SYSTEM ANALYSIS  
3.1 Existing System  
3.2 Proposed System  
3.3 Feasibility Study  

Chapter 4: SYSTEM DESIGN  
4.1 Introduction  
4.2 Architectural Design  
4.3 Functional Modules  
4.4 Data Flow Diagram (Textual)  
4.5 Use Case Diagram (Textual)  

Chapter 5: SYSTEM IMPLEMENTATION  
5.1 Overview  
5.2 Technologies Used  
5.3 Modules and Code Integration  

Chapter 6: SYSTEM TESTING  
6.1 Overview  
6.2 Types of Testing Performed  
6.3 Test Cases and Results  
6.4 Bug Fixes and Optimizations  
6.5 Conclusion  

Chapter 7: RESULTS AND DISCUSSION  
7.1 Introduction  
7.2 Sample Test Case Specification  
7.3 Sample Output Screens (Description)  
7.4 Response Time and Performance  
7.5 Strengths of the System  
7.6 Observations  
7.7 Discussion  
7.8 Future Improvements  
7.9 Sample Outputs (Example JSON)  

Chapter 8: CONCLUSION AND FUTURE WORK  
8.1 Conclusion  
8.2 Limitations  
8.3 Future Enhancement  

Chapter 9: REFERENCES  

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Brief Information about the Project
Smart Eye is a web application that estimates whether a student appears engaged during an online class. The application uses a browser webcam stream to capture frames and sends them to a Flask backend. The backend performs face landmark detection and computes engagement indicators. The system returns:

- **Status**: Engaged / Not Engaged / Setting up… / No Face Detected / Unknown  
- **Engagement score**: numeric score in \([0,1]\) when applicable  
- **Reason**: human‑readable explanation (e.g., looking down, eyes closed, face blocked)  
- **Confidence**: score showing how reliable the output is  
- **Live stats**: engaged seconds, disengaged seconds, face detection rate

The project supports two roles:
- **Student**: starts a session, views live engagement status, receives explainable feedback, and can reset calibration if needed.  
- **Admin**: views class dashboards, monitors multiple faces in a single frame, downloads session CSV reports, and views analytics.

### 1.2 Motivation and Contribution of Project
**Motivation**  
In virtual classes, instructors cannot easily measure student attention. Attendance alone cannot indicate whether students are following the class. Also, unfair systems can penalize students due to lighting, camera placement, face‑detection failures, or physiological differences.

**Contribution**  
This project provides an end‑to‑end solution with:
- A privacy‑friendly pipeline (frames are processed transiently; raw video is not stored)
- Interpretable, explainable engagement decisions (reasons + signals)
- Fair handling of camera failures (No Face Detected is not treated as disengagement)
- Per‑session calibration reset for quick recovery during demo/viva
- Admin real‑time multi‑face analysis with per‑face independent processing

### 1.3 Objective of the Project
The objectives of Smart Eye are:
- To create a usable student engagement monitoring system for online classes
- To compute engagement using reliable facial landmarks and geometry (EAR + pose)
- To provide explainability (reason + confidence) instead of a black‑box label
- To store engagement events in a database and compute session analytics
- To implement admin dashboards and session export for reporting

### 1.4 Scope of the Project
The scope includes:
- Student registration/login, join session workflow, and live dashboard
- Admin login, session analytics pages, and real‑time class dashboard
- Engagement API endpoints for real‑time detection and data persistence
- Session export to CSV (timestamp, state, score, reason, confidence)

Out of scope:
- Emotion recognition or identity matching across frames as a production‑grade feature
- Cloud deployment hardening beyond local university demo needs

---

## CHAPTER 2 — LITERATURE SURVEY

### 2.1 Overview of the Existing Systems
Existing engagement monitoring approaches include:
- **Manual observation**: teacher visually checks students; not scalable for large classes.
- **Attendance‑based systems**: track presence but not attention quality.
- **Basic webcam heuristics**: simple face presence checks; can be unfair and noisy.
- **Deep learning emotion classification**: requires datasets, model training, and may be biased.

### 2.2 Related Work
Common computer vision techniques related to this project:
- **Facial landmark models** such as MediaPipe Face Mesh for stable landmark detection.
- **Eye Aspect Ratio (EAR)** widely used in drowsiness detection to infer eye closure.
- **Head pose estimation** using solvePnP (3D model points + 2D face points) to infer yaw/pitch.
- **Temporal smoothing** techniques to reduce flicker in real‑time classification.

### 2.3 Research Gap Identified
Many systems focus on one of the following, but not all simultaneously:
- Real‑time performance in a lightweight web app
- Explainability for students (why not engaged)
- Fairness when the camera feed is poor or face is not detected
- Persistent session analytics and reporting for admin review

### 2.4 Contribution Over Existing Work
Smart Eye improves practical usability by:
- Adding **calibration** for stable head‑pose reference (baseline)
- Handling **occlusion/phone** scenarios with additional checks
- Returning **reason + confidence + signals**, supporting transparency
- Computing **face detection rate** and excluding “no face” frames from penalties

---

## CHAPTER 3 — SYSTEM ANALYSIS

### 3.1 Existing System
In many classes, engagement is evaluated informally (teacher observation) or by attendance. These methods cannot reliably quantify attention over time and do not produce reports. Additionally, they do not help the student understand how to improve.

### 3.2 Proposed System
The proposed system uses a web camera feed and processes frames periodically:
- Detects face landmarks
- Computes EAR and head pose angles
- Uses deterministic rules and smoothing to output engagement
- Stores events in SQLite and computes session statistics

### 3.3 Feasibility Study
**Technical feasibility**  
The project uses open‑source libraries that run on normal laptops:
- Flask, OpenCV, MediaPipe, NumPy, SQLite

**Economic feasibility**  
No paid APIs are required; it is suitable for academic submission.

**Operational feasibility**  
User‑friendly dashboards are available for student and admin, including a reset calibration button.

---

## CHAPTER 4 — SYSTEM DESIGN

### 4.1 Introduction
Smart Eye is designed as a client‑server web system where the browser captures frames and the server evaluates engagement. The system is modular: authentication, engagement computation, storage, analytics, and UI dashboards.

### 4.2 Architectural Design
**Client (Browser)**
- Uses `getUserMedia()` to access webcam
- Captures frames with `<canvas>`
- Sends base64 JPEG frames to backend endpoints
- Updates UI (status badge, score gauge, calibration bar, time counters)

**Server (Flask)**
- Decodes base64 image
- Runs MediaPipe Face Mesh on RGB frames
- Computes EAR and head pose
- Applies calibration and decision rules
- Stores EngagementEvent in SQLite
- Returns JSON for live UI updates

**Database (SQLite)**
- Users, teaching sessions, student sessions
- Engagement events per student session

### 4.3 Functional Modules
1. **Authentication Module**
   - Register/login/logout for student and admin
   - Password hashing using Werkzeug

2. **Student Session Module**
   - Join a teaching session using session code
   - Start session and open dashboard

3. **Engagement Detection Module**
   - Face landmark detection using MediaPipe
   - EAR computation
   - Head pose estimation (solvePnP + angle normalization)
   - Baseline calibration for yaw/pitch
   - Occlusion handling for phone/object

4. **Data Persistence Module**
   - Stores engagement events per session with timestamp, state, score, reason, confidence

5. **Analytics Module**
   - Engaged time, disengaged time, face detection rate
   - Export report CSV per session

6. **Admin Real‑Time Dashboard Module**
   - Processes multiple faces independently in one frame
   - Shows per‑face cards “Student 1, Student 2 …”
   - Shows class engagement summary bar

### 4.4 Data Flow Diagram (Textual)
**Student flow**
1. Browser captures frame → 2. POST `/api/engagement` → 3. Backend computes status/score → 4. Save event in DB → 5. Compute stats → 6. Return JSON → 7. Frontend updates UI.

**Admin flow**
1. Browser captures frame → 2. POST `/detect` → 3. OpenCV finds face crops → 4. MediaPipe per crop → 5. Return students + class score → 6. Frontend renders cards.

### 4.5 Use Case Diagram (Textual)
**Actors**: Student, Admin

Student use cases:
- Register / Login
- Join session and start monitoring
- View live engagement status and explanation
- Reset calibration
- End session and view session summary

Admin use cases:
- Login
- Create teaching sessions with session code
- View student sessions list and analytics
- Export session CSV report
- Live class monitoring via webcam

---

## CHAPTER 5 — SYSTEM IMPLEMENTATION

### 5.1 Overview
The system is implemented using Flask for routing and API handling. MediaPipe Face Mesh produces facial landmarks. Engagement is determined using deterministic rules based on:
- EAR (eye closure)
- Head pose deltas from baseline (yaw/pitch)
- Occlusion detection heuristics
- Debouncing and smoothing for stable output

### 5.2 Technologies Used
**Backend**
- Python 3.x
- Flask
- Flask‑SQLAlchemy
- SQLite
- OpenCV (`opencv-python`)
- NumPy
- MediaPipe

**Frontend**
- HTML templates (Jinja2)
- Bootstrap + custom CSS
- JavaScript (`fetch`, `getUserMedia`, canvas)

**Testing**
- pytest (unit + smoke tests)

### 5.3 Modules and Code Integration
Important integration points:
- `/api/engagement` saves each event to DB and returns live stats
- `_compute_session_stats()` computes engaged/disengaged seconds and face detection rate
- Student dashboard shows calibration progress and time counters
- Admin dashboard processes multiple faces independently

**Note**: The attached code appendix contains the major implementation files and key code excerpts.

---

## CHAPTER 6 — SYSTEM TESTING

### 6.1 Overview
Testing includes route smoke tests and core engagement logic validation using pytest. The goal is to ensure stable behavior after changes to thresholds, calibration, and fairness logic.

### 6.2 Types of Testing Performed
- **Unit testing**: engagement logic behavior under mocked EAR/pose values  
- **Integration testing**: Flask routes reachable and proper redirects for role protection  
- **Manual UI testing**: webcam permissions, calibration, reset button, and admin dashboard updates

### 6.3 Test Cases and Results
Representative test cases:
- Home page loads successfully
- Student login redirects to join session workflow
- Engagement logic returns Not Engaged for closed eyes
- Looking down triggers Not Engaged after debounce

### 6.4 Bug Fixes and Optimizations
Examples of resolved issues (documented during development):
- False disengagement from pitch flip (solvePnP normalization)
- Calibration stuck in “Setting up…” resolved via simplified 3‑frame baseline
- Fairness improvements for No Face Detected and Unknown frames
- Stats counters fixed to compute time based on sampling interval

### 6.5 Conclusion
Tests and manual runs confirm that Smart Eye works as expected for demo/viva usage, with readable explanations and persistent session logs.

---

## CHAPTER 7 — RESULTS AND DISCUSSION

### 7.1 Introduction
This chapter summarizes observed system outputs, stability of status, and dashboard feedback.

### 7.2 Sample Test Case Specification
**Case 1 — Student faces camera**
- Expected: Engaged, score near 1.0, reason “OK”

**Case 2 — Student looks down**
- Expected: Not Engaged, reason “Looking down — possible phone use”

**Case 3 — Face not visible**
- Expected: Status “No Face Detected”, score null, not penalized in analytics

### 7.3 Sample Output Screens (Description)
- Student dashboard shows:
  - Calibration bar for first 3 frames
  - Circular score gauge and status badge
  - Engaged/Disengaged time counters
  - “Why not engaged” explainability card

### 7.4 Response Time and Performance
Performance is improved by:
- Client‑side canvas capture at fixed resolution (avoids black bars)
- Server‑side downscaling (`MAX_FRAME_DIM`)
- Using lightweight geometric logic after landmarks are extracted

### 7.5 Strengths of the System
- Interpretable decisions (reasons + signals)
- Works in real time on standard hardware
- Admin multi‑face processing for class view
- Persistent session data and CSV reporting

### 7.6 Observations
- Lighting and camera angle affect landmark reliability; warnings are displayed.
- Downward gaze is treated as a stronger disengagement signal.
- Occlusion/phone scenarios can be detected using landmark reliability heuristics.

### 7.7 Discussion
The system is suitable for an academic demonstration because it combines:
web development + database + computer vision + explainable rules.

### 7.8 Future Improvements
- Add engagement timeline charts (Chart.js) for per‑session visualization
- Add face tracking across frames for stable identity mapping in admin view
- Improve occlusion detection robustness using segmentation models

### 7.9 Sample Outputs (Example JSON)
Example response from `/api/engagement`:
- `status`, `engagement_score`, `reason`, `confidence`, `signals`, and `stats` are returned.

---

## CHAPTER 8 — CONCLUSION AND FUTURE WORK

### 8.1 Conclusion
Smart Eye demonstrates a complete and practical engagement monitoring system using Flask and computer vision. It supports student and admin workflows, fair analytics, explainable decisions, and reporting.

### 8.2 Limitations
- Engagement is inferred from visual cues only; it cannot fully represent learning.
- Variations in cameras/lighting can impact accuracy.
- Multi‑person identity tracking is simplified (Student 1, Student 2, …).

### 8.3 Future Enhancement
- Add session engagement charts, better tracking, and optional deployment improvements.

---

## CHAPTER 9 — REFERENCES
- Flask Documentation  
- Flask‑SQLAlchemy Documentation  
- OpenCV Documentation  
- MediaPipe Face Mesh Documentation  
- NumPy Documentation  
- MDN Web Docs (`getUserMedia`, Canvas, Fetch API)

---

### APPENDIX
- See `APPENDIX_CODE_24M11MC158.md` for code attachment and key excerpts.

