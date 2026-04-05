// ============================================================================
// realtime.js
// ============================================================================
// Frontend logic for:
//   - Accessing the user's webcam from the browser
//   - Periodically capturing frames using a hidden <canvas>
//   - Sending frames to the Flask backend (/detect) as base64 images
//   - Updating the dashboard UI with engagement analytics
//
// This demonstrates how JavaScript in the browser can work together with
// Python + OpenCV on the server for a simple real-time ML/DL application.
// ============================================================================

const video = document.getElementById("video");
const canvas = document.getElementById("capture-canvas");
const videoStatus = document.getElementById("video-status");

const engagementPercentageEl = document.getElementById("engagement-percentage");
const totalStudentsEl = document.getElementById("total-students");
const studentsListEl = document.getElementById("students-list");
const classEngagementBarEl = document.getElementById("class-engagement-progress");
const classEngagementLabelEl = document.getElementById("class-engagement-label");

// Low attention tracking (rolling window of last 5 scores per student card).
const SCORE_WINDOW_SIZE = 5;
const scoreWindowByStudent = {}; // studentId -> [score1, score2, ...]
const lowAttentionAlertedByStudent = {}; // studentId -> boolean

function playSoftBeepOnce() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();

    oscillator.type = "sine";
    oscillator.frequency.value = 440;
    gain.gain.value = 0.0001;

    oscillator.connect(gain);
    gain.connect(ctx.destination);

    // Soft attack/decay so it isn't harsh.
    const now = ctx.currentTime;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.08, now + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.30);

    oscillator.start(now);
    oscillator.stop(now + 0.31);

    // Cleanup.
    oscillator.onended = () => {
      try {
        ctx.close();
      } catch (e) {}
    };
  } catch (e) {
    // Audio might be blocked unless there was a user gesture.
  }
}

// Capture interval in milliseconds (1000 ms = 1 second)
const CAPTURE_INTERVAL_MS = 1000;
// Cap canvas size to reduce payload + server CPU.
const MAX_CANVAS_DIM = 480;

let captureIntervalId = null;

// ----------------------------------------------------------------------------
// Helper: Initialize webcam stream using getUserMedia
// ----------------------------------------------------------------------------

async function initWebcam() {
  try {
    // Ask browser for permission to access webcam
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    videoStatus.textContent = "Webcam active. Analyzing engagement every second...";

    // Once video is playing, start periodic capture
    video.addEventListener("loadedmetadata", () => {
      startCapturing();
    });
  } catch (err) {
    console.error("Error accessing webcam:", err);
    videoStatus.textContent =
      "Unable to access webcam. Please allow camera permission and reload the page.";
  }
}

// ----------------------------------------------------------------------------
// Helper: Start periodic frame capture and send to backend
// ----------------------------------------------------------------------------

function startCapturing() {
  if (!video.videoWidth || !video.videoHeight) {
    // If metadata isn't ready yet, wait a bit and try again
    setTimeout(startCapturing, 200);
    return;
  }

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  // Use the exact video dimensions to avoid capturing black letterbox padding.
  canvas.width = vw;
  canvas.height = vh;

  // Clear any existing interval (safety)
  if (captureIntervalId) {
    clearInterval(captureIntervalId);
  }

  captureIntervalId = setInterval(captureFrameAndSend, CAPTURE_INTERVAL_MS);
}

// ----------------------------------------------------------------------------
// Helper: Capture current frame, convert to base64, send to /detect
// ----------------------------------------------------------------------------

async function captureFrameAndSend() {
  try {
    const ctx = canvas.getContext("2d");
    // Draw the current frame from video into the canvas
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Convert canvas image to a compressed JPEG base64 data URL
    const dataUrl = canvas.toDataURL("image/jpeg", 0.55);

    // Send POST request to Flask backend with JSON body {image: dataUrl}
    const response = await fetch("/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    if (!response.ok) {
      console.error("Backend responded with error", response.status);
      return;
    }

    const result = await response.json();
    updateDashboard(result);
  } catch (err) {
    console.error("Error during capture/send:", err);
  }
}

// ----------------------------------------------------------------------------
// Helper: Update dashboard UI using JSON from backend
// ----------------------------------------------------------------------------

function updateDashboard(data) {
  const classEngagement = data.class_engagement || 0.0;
  const totalStudents = data.total_students || 0;
  const students = data.students || [];

  // Convert from 0..1 to percentage (0..100) and show one decimal place
  const percentage = Math.round(classEngagement * 1000) / 10; // e.g., 0.783 -> 78.3

  engagementPercentageEl.textContent = `${percentage.toFixed(1)}%`;
  totalStudentsEl.textContent = totalStudents.toString();

  // Summary bar coloring: green if > 60, yellow if 40..60, red if < 40
  const pctNum = Math.max(0, Math.min(100, percentage));
  let barColor = "#ef4444"; // red default
  if (pctNum >= 60) barColor = "#22c55e";
  else if (pctNum >= 40) barColor = "#f59e0b";

  if (classEngagementLabelEl) {
    classEngagementLabelEl.textContent = `Class Engagement: ${pctNum.toFixed(1)}%`;
  }
  if (classEngagementBarEl) {
    classEngagementBarEl.style.width = `${pctNum.toFixed(1)}%`;
    classEngagementBarEl.style.backgroundColor = barColor;
  }

  // Build list of students
  if (students.length === 0) {
    studentsListEl.innerHTML =
      '<p class="text-muted mb-0">No faces detected in the current frame.</p>';
    return;
  }

  studentsListEl.innerHTML = "";

  students.forEach((student) => {
    const studentId = student.id || "Unknown";

    const scoreRaw = student.engagement_score;
    const score =
      typeof scoreRaw === "number" ? scoreRaw : Number(scoreRaw) || 0.0;

    // Update rolling window.
    if (!scoreWindowByStudent[studentId]) scoreWindowByStudent[studentId] = [];
    const win = scoreWindowByStudent[studentId];
    win.push(score);
    while (win.length > SCORE_WINDOW_SIZE) win.shift();

    const hasEnough = win.length === SCORE_WINDOW_SIZE;
    const avg = hasEnough ? win.reduce((a, b) => a + b, 0) / win.length : 0.0;

    const lowAttention = hasEnough && avg < 0.4;

    const card = document.createElement("div");
    card.classList.add("student-card");

    const statusLower = String(student.status || "").toLowerCase();
    if (statusLower === "engaged") {
      card.classList.add("engaged");
    } else if (statusLower === "unknown" || statusLower === "no face detected") {
      card.classList.add("unknown");
    } else {
      card.classList.add("not-engaged");
    }

    // Optional: add a hover tooltip with explanation/confidence.
    const reason = student.reason ? String(student.reason) : "";
    const confidence =
      typeof student.confidence === "number" ? Number(student.confidence) : null;
    const confText = confidence !== null ? ` (conf: ${confidence.toFixed(2)})` : "";
    card.title = `${student.status || "N/A"}${confText}${reason ? ` - ${reason}` : ""}`;

    // Left side: id + status
    const left = document.createElement("div");

    const idSpan = document.createElement("span");
    idSpan.classList.add("student-id");
    idSpan.textContent = student.id || "Unknown";

    const statusSpan = document.createElement("span");
    statusSpan.classList.add("student-status", "ms-2");
    statusSpan.textContent = `(${student.status || "N/A"})`;

    left.appendChild(idSpan);
    left.appendChild(statusSpan);

    // Low attention badge + alert styling.
    const alreadyAlerted = Boolean(lowAttentionAlertedByStudent[studentId]);
    if (lowAttention) {
      card.classList.add("low-attention");

      const badge = document.createElement("span");
      badge.classList.add("badge", "bg-danger", "ms-2");
      badge.textContent = "⚠️ Low Attention";
      left.appendChild(badge);

      // Play beep only once per alert until average recovers.
      if (!alreadyAlerted) {
        lowAttentionAlertedByStudent[studentId] = true;
        playSoftBeepOnce();
      }
    } else {
      card.classList.remove("low-attention");
      lowAttentionAlertedByStudent[studentId] = false;
    }

    // Right side: score
    const scoreSpan = document.createElement("span");
    scoreSpan.classList.add("student-score");
    const scoreValue = score.toFixed(3);
    scoreSpan.textContent = scoreValue;

    card.appendChild(left);
    card.appendChild(scoreSpan);

    studentsListEl.appendChild(card);
  });
}

// ----------------------------------------------------------------------------
// Entry point: start when page is loaded
// ----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    videoStatus.textContent =
      "Your browser does not support webcam access (getUserMedia API). Try a modern browser.";
    return;
  }

  initWebcam();
});


