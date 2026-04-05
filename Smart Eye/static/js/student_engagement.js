// ============================================================================
// student_engagement.js
// ============================================================================
// Frontend logic for the STUDENT dashboard:
//   - Access webcam via getUserMedia
//   - Capture frames every few seconds using a hidden <canvas>
//   - Send frames to Flask backend at /api/engagement
//   - Display current engagement state ("engaged" / "disengaged") and score
// ============================================================================

const video = document.getElementById("video");
const canvas = document.getElementById("capture-canvas");
const videoStatus = document.getElementById("video-status");

const engagementIndicator = document.getElementById("engagement-indicator");
const scoreGaugeEl = document.getElementById("score-gauge");
const scoreGaugeTextEl = document.getElementById("score-gauge-text");

// Capture interval in milliseconds
const CAPTURE_INTERVAL_MS = 2000; // 2 seconds
// Cap canvas size to reduce payload + server CPU.
const MAX_CANVAS_DIM = 480;

let captureIntervalId = null;

function showCalibrationBar() {
  const calibrationWrap = document.getElementById("calibration-progress-wrap");
  const calibrationBar = document.getElementById("calibration-progress-bar");
  const whyCard = document.getElementById("why-not-engaged-card");
  const warningEl = document.getElementById("face-warning-banner");

  if (whyCard) whyCard.classList.add("d-none");
  if (warningEl) warningEl.classList.add("d-none");

  if (calibrationWrap) calibrationWrap.classList.remove("d-none");
  if (calibrationBar) {
    calibrationBar.style.width = "0%";
    calibrationBar.setAttribute("aria-valuenow", "0");
  }
}

async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    video.srcObject = stream;
    videoStatus.textContent = "Webcam active. Analyzing every 2 seconds...";

    video.addEventListener("loadedmetadata", () => {
      startCapturing();
    });
  } catch (err) {
    console.error("Error accessing webcam:", err);
    videoStatus.textContent =
      "Unable to access webcam. Please allow camera permission and reload.";
  }
}

function startCapturing() {
  if (!video.videoWidth || !video.videoHeight) {
    setTimeout(startCapturing, 200);
    return;
  }

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  // Ensure aspect ratio is not distorted (helps MediaPipe landmark accuracy).
  video.style.width = "100%";
  video.style.height = "auto";
  // Use the exact video dimensions to avoid capturing black letterbox padding.
  canvas.width = vw;
  canvas.height = vh;

  if (captureIntervalId) {
    clearInterval(captureIntervalId);
  }

  captureIntervalId = setInterval(captureFrameAndSend, CAPTURE_INTERVAL_MS);
}

async function captureFrameAndSend() {
  try {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Poor lighting guard: compute average brightness from canvas pixels.
    // brightness = (sum of r+g+b) / (totalPixels * 3) => 0..255
    const lightingWarningEl = document.getElementById(
      "lighting-warning-banner"
    );
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    let rgbSum = 0;
    // Step by 4 bytes (R,G,B,A)
    for (let i = 0; i < data.length; i += 4) {
      rgbSum += data[i] + data[i + 1] + data[i + 2];
    }
    const numPixels = data.length / 4;
    const brightness = rgbSum / (numPixels * 3);

    if (brightness < 60) {
      if (lightingWarningEl) lightingWarningEl.classList.remove("d-none");
      console.warn(
        `Frame skipped: low brightness (${brightness.toFixed(1)})`
      );
      return; // Skip this frame (do not send to backend)
    }
    if (lightingWarningEl) lightingWarningEl.classList.add("d-none");

    const dataUrl = canvas.toDataURL("image/jpeg", 0.55);

    const response = await fetch("/api/engagement", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    if (!response.ok) {
      console.error("Backend responded with error", response.status);
      return;
    }

    const result = await response.json();
    updateIndicator(result);
  } catch (err) {
    console.error("Error during capture/send:", err);
  }
}

function updateIndicator(data) {
  // Prefer new API keys (engagement_score, status, reason), fall back to legacy keys
  let score = null;
  if (data.engagement_score === null) {
    score = null;
  } else if (typeof data.engagement_score === "number") {
    score = data.engagement_score;
  } else if (typeof data.score === "number") {
    score = data.score;
  }

  const state = data.status || data.engagement_state || (score >= 0.5 ? "engaged" : "disengaged");
  const reason = data.reason || "";
  const signals = data.signals || null;
  const confidence =
    typeof data.confidence === "number" ? data.confidence : null;

  const warningEl = document.getElementById("face-warning-banner");
  const whyCard = document.getElementById("why-not-engaged-card");
  const whyReasonEl = document.getElementById("why-not-engaged-reason");
  const whySignalsEl = document.getElementById("why-not-engaged-signals");
  const calibrationWrap = document.getElementById("calibration-progress-wrap");
  const calibrationBar = document.getElementById("calibration-progress-bar");

  const baselineReady =
    typeof data.baseline_ready === "boolean" ? data.baseline_ready : false;
  const calibrationProgress =
    typeof data.calibration_progress === "number"
      ? data.calibration_progress
      : 0.0;

  if (whyCard) whyCard.classList.add("d-none");

  // Update calibration progress UI
  const stateLowerRaw = String(state || "").toLowerCase();
  const isNoFace = stateLowerRaw === "no face detected";
  if (calibrationWrap && !baselineReady && !isNoFace) {
    calibrationWrap.classList.remove("d-none");
  }
  if (calibrationWrap && (baselineReady || isNoFace)) {
    calibrationWrap.classList.add("d-none");
  }
  if (calibrationBar && calibrationWrap && !calibrationWrap.classList.contains("d-none")) {
    const pct = Math.max(0, Math.min(1, calibrationProgress)) * 100;
    calibrationBar.style.width = `${pct.toFixed(0)}%`;
    calibrationBar.setAttribute("aria-valuenow", pct.toFixed(0));
  }

  engagementIndicator.innerHTML = "";
  const badge = document.createElement("span");
  badge.classList.add("badge", "px-3", "py-2");

  const stateLower = String(state).toLowerCase();
  const confText = confidence !== null ? ` (conf: ${Number(confidence).toFixed(2)})` : "";

  // Update score gauge (conic-gradient ring).
  // Score is in [0..1]. Render as percentage in the center.
  if (scoreGaugeEl && scoreGaugeTextEl) {
    if (score === null || stateLower === "setting up..." || stateLower.startsWith("setting up")) {
      scoreGaugeEl.style.setProperty("--pct", "0");
      scoreGaugeEl.style.setProperty("--gauge-color", "#9ca3af"); // gray
      scoreGaugeTextEl.textContent = "—";
    } else if (score === null) {
      scoreGaugeEl.style.setProperty("--pct", "0");
      scoreGaugeEl.style.setProperty("--gauge-color", "#9ca3af"); // gray
      scoreGaugeTextEl.textContent = "—";
    } else {
      const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
      let color = "#9ca3af";
      if (score >= 0.6) color = "#22c55e"; // green
      else if (score >= 0.3) color = "#f59e0b"; // yellow
      else color = "#ef4444"; // red

      scoreGaugeEl.style.setProperty("--pct", String(pct));
      scoreGaugeEl.style.setProperty("--gauge-color", color);
      scoreGaugeTextEl.textContent = `${pct}%`;
    }
  }

  // Explainability card: only show when the server says "Not Engaged".
  if (stateLower === "not engaged") {
    if (whyCard) whyCard.classList.remove("d-none");

    let prettyWhy = reason ? String(reason) : "Not enough information yet.";
    if (signals) {
      const ear = Number(signals.ear);
      const earTh = Number(signals.ear_threshold);
      const baselineReady = Boolean(signals.baseline_ready);

      const earDesc = Number.isFinite(ear) && Number.isFinite(earTh)
        ? (ear < earTh
          ? `Your eyes look closed (EAR ${ear.toFixed(2)} < ${earTh.toFixed(2)}).`
          : `Your eyes look open (EAR ${ear.toFixed(2)} >= ${earTh.toFixed(2)}).`)
        : "EAR not available for this frame.";

      let headDesc = "";
      const r = String(reason || "");
      if (r.includes("Eyes closed + head turned")) {
        headDesc = "Your head pose suggests you are looking away.";
      } else if (r.includes("Looking left/right")) {
        headDesc = "Your head pose suggests a left/right turn.";
      } else if (r.includes("Looking up/down")) {
        headDesc = "Your head pose suggests an up/down tilt.";
      } else if (r.includes("Low Attention")) {
        headDesc = "No strong eye/head trigger, but the attention score was low.";
      } else {
        headDesc = "Head/pose signals were not strong enough for a precise label.";
      }

      const baselineText = baselineReady
        ? "Baseline ready: yes."
        : "Baseline ready: no (calibrating).";
      const yawTxt = Number.isFinite(Number(signals.yaw_delta))
        ? `yaw delta: ${Number(signals.yaw_delta).toFixed(1)}°`
        : "yaw delta: -";
      const pitchTxt = Number.isFinite(Number(signals.pitch_delta))
        ? `pitch delta: ${Number(signals.pitch_delta).toFixed(1)}°`
        : "pitch delta: -";

      if (whyReasonEl) whyReasonEl.textContent = prettyWhy;
      if (whySignalsEl) {
        whySignalsEl.innerHTML =
          `${earDesc}<br/>${headDesc}<br/>${baselineText}<br/>EAR: ${ear.toFixed(2)} (threshold ${earTh.toFixed(2)}).<br/>${yawTxt}, ${pitchTxt}.`;
      }
    } else {
      if (whyReasonEl) whyReasonEl.textContent = prettyWhy;
      if (whySignalsEl) whySignalsEl.textContent = "Signals not available for this frame.";
    }
  }

  if (stateLower === "setting up..." || stateLower.startsWith("setting up")) {
    if (warningEl) warningEl.classList.add("d-none");
    badge.classList.add("bg-secondary");
    badge.textContent = `Setting up...${confText}`;
  } else if (stateLower === "unknown") {
    if (warningEl) warningEl.classList.add("d-none");
    badge.classList.add("bg-secondary");
    badge.textContent = `Calibrating${confText}`;
  } else if (stateLower === "engaged") {
    if (warningEl) warningEl.classList.add("d-none");
    badge.classList.add("bg-success");
    badge.textContent = `You are Engaged (${Number(score).toFixed(2)})${confText}`;
  } else if (stateLower === "no face detected") {
    if (warningEl) warningEl.classList.remove("d-none");
    badge.classList.add("bg-secondary");
    badge.textContent = `Face not visible`;
  } else {
    if (warningEl) warningEl.classList.add("d-none");
    badge.classList.add("bg-danger");
    const reasonText = reason ? ` - ${reason}` : "";
    badge.textContent = `You appear Disengaged (${Number(score).toFixed(2)})${reasonText}${confText}`;
  }

  engagementIndicator.appendChild(badge);
  
  // (Score is shown in gauge; no separate score text anymore.)

  // Update stats if provided by backend
  if (data.stats) {
    const eTime = document.getElementById("engaged-time");
    const dTime = document.getElementById("disengaged-time");
    if (eTime) eTime.textContent = formatDuration(data.stats.engaged_seconds);
    if (dTime) dTime.textContent = formatDuration(data.stats.disengaged_seconds);
  }
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

function endSession() {
  // 1. Stop capturing
  if (captureIntervalId) {
    clearInterval(captureIntervalId);
    captureIntervalId = null;
  }

  // 2. Stop webcam
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }

  // 3. Update UI
  if (videoStatus) videoStatus.textContent = "Session ended.";

  // Hide modal
  const modalEl = document.getElementById("endSessionModal");
  // Check if bootstrap is defined (it should be via base.html)
  if (typeof bootstrap !== "undefined" && modalEl) {
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) {
      modal.hide();
    }
  }

  // Replace video with summary message
  const videoContainer = document.querySelector(".video-container");
  if (videoContainer) {
    videoContainer.innerHTML = `
      <div class="d-flex flex-column align-items-center justify-content-center h-100 bg-light rounded-3 p-4 text-center">
        <h3 class="text-primary fw-bold mb-3">Session Ended</h3>
        <p class="text-muted mb-4">Your engagement stats have been recorded.</p>
        <a href="/student/home" class="btn btn-primary px-4 py-2 fw-bold">Back to Home</a>
      </div>
    `;
  }

  // Update status badge
  if (engagementIndicator) {
    engagementIndicator.innerHTML =
      '<span class="badge bg-secondary px-3 py-2">Session Completed</span>';
  }

  // Remove the End Session button
  const endBtn = document.querySelector('button[data-bs-target="#endSessionModal"]');
  if (endBtn) endBtn.remove();
}

// Expose to window
window.endSession = endSession;

// Start
initWebcam();

// Baseline reset button (lets users fix a bad baseline without restarting).
const resetBtn = document.getElementById("reset-baseline-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/engagement/reset_baseline", { method: "POST" });
      showCalibrationBar();
      console.log("Baseline reset — recalibrating...");
    } catch (e) {
      console.error("Baseline reset failed:", e);
    }
  });
}


