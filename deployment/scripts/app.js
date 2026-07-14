import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const WINDOW_SIZE = 7;
const CALIBRATION_FRAMES = 15;
const SAMPLE_INTERVAL_MS = 1000 / 15;
const HYBRID_THRESHOLD = 0.5;
const DROWSY_HISTORY_SIZE = 15;
const DROWSY_VOTES_REQUIRED = 10;
const TARGET_BLINKS_PER_MIN = 12;
const VERY_LOW_BLINKS_PER_MIN = 7;
const LONG_NO_BLINK_MS = 15000;
const BLINK_RATE_WINDOW_MS = 60000;
const TREND_WINDOW_MS = 180000;
const NOTIFICATION_COOLDOWN_MS = 60000;

const LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380];

const state = {
    faceLandmarker: null,
    running: false,
    stream: null,
    animationId: null,
    lastVideoTime: -1,
    lastSampleTime: 0,
    calibrationEars: [],
    currentMax: 0.23,
    calibrated: false,
    slidingWindow: [],
    predHistory: [],
    blinkEvents: [],
    totalWindows: 0,
    skippedMl: 0,
    blinkCount: 0,
    blinkCooldown: 0,
    sessionStartTime: null,
    lastBlinkTime: null,
    longestNoBlinkMs: 0,
    drowsyEvents: 0,
    wasDrowsy: false,
    notificationsEnabled: false,
    lastNotificationAt: {},
    health: {
        blinkRate: 0,
        blinkTrend: 0,
        noBlinkSeconds: 0,
        sessionSeconds: 0,
        warning: ""
    },
    latest: {
        rawEar: null,
        normEar: null,
        prediction: -1,
        status: "Idle",
        cpuMode: "N/A",
        faceDetected: false,
        drowsy: false
    }
};

const els = {
    video: document.getElementById("camera"),
    canvas: document.getElementById("overlay"),
    startBtn: document.getElementById("start-btn"),
    stopBtn: document.getElementById("stop-btn"),
    resetBtn: document.getElementById("reset-btn"),
    notifyBtn: document.getElementById("notify-btn"),
    notificationStatus: document.getElementById("notification-status"),
    status: document.getElementById("status"),
    rawEar: document.getElementById("raw-ear"),
    normEar: document.getElementById("norm-ear"),
    prediction: document.getElementById("prediction"),
    calibration: document.getElementById("calibration"),
    cpuSaved: document.getElementById("cpu-saved"),
    cpuMode: document.getElementById("cpu-mode"),
    blinkCount: document.getElementById("blink-count"),
    blinkRate: document.getElementById("blink-rate"),
    blinkTarget: document.getElementById("blink-target"),
    blinkTrend: document.getElementById("blink-trend"),
    noBlinkStreak: document.getElementById("no-blink-streak"),
    sessionTime: document.getElementById("session-time"),
    drowsyEvents: document.getElementById("drowsy-events"),
    alert: document.getElementById("alert"),
    message: document.getElementById("message")
};

function distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.hypot(dx, dy);
}

function computeEar(landmarks, indices) {
    const points = indices.map((idx) => landmarks[idx]);
    const v1 = distance(points[1], points[5]);
    const v2 = distance(points[2], points[4]);
    const h = distance(points[0], points[3]);
    return h === 0 ? 0 : (v1 + v2) / (2 * h);
}

function extractAdvancedFeatures(w) {
    const minVal = Math.min(...w);
    const maxVal = Math.max(...w);
    const meanVal = w.reduce((sum, val) => sum + val, 0) / w.length;
    const variance = w.reduce((sum, val) => sum + Math.pow(val - meanVal, 2), 0) / w.length;
    const stdVal = Math.sqrt(variance);
    const ratioCenter = (w[0] + w[6]) / (2 * w[3] + 1e-5);

    let kurtosis = 0.0;
    if (stdVal > 1e-5) {
        const moment4 = w.reduce((sum, val) => sum + Math.pow((val - meanVal) / stdVal, 4), 0);
        kurtosis = moment4 / w.length - 3.0;
    }

    return [...w, minVal, maxVal, stdVal, ratioCenter, kurtosis];
}

function resetPipeline() {
    const now = performance.now();
    state.lastSampleTime = 0;
    state.calibrationEars = [];
    state.currentMax = 0.23;
    state.calibrated = false;
    state.slidingWindow = [];
    state.predHistory = [];
    state.blinkEvents = [];
    state.totalWindows = 0;
    state.skippedMl = 0;
    state.blinkCount = 0;
    state.blinkCooldown = 0;
    state.sessionStartTime = now;
    state.lastBlinkTime = null;
    state.longestNoBlinkMs = 0;
    state.drowsyEvents = 0;
    state.wasDrowsy = false;
    state.health = {
        blinkRate: 0,
        blinkTrend: 0,
        noBlinkSeconds: 0,
        sessionSeconds: 0,
        warning: ""
    };
    state.latest = {
        rawEar: null,
        normEar: null,
        prediction: -1,
        status: "Calibrating",
        cpuMode: "N/A",
        faceDetected: false,
        drowsy: false
    };
    renderHud();
}

function pushLimited(buffer, value, maxLen) {
    buffer.push(value);
    if (buffer.length > maxLen) {
        buffer.shift();
    }
}

function recordBlink(timestamp) {
    state.blinkCount += 1;
    state.lastBlinkTime = timestamp;
    state.blinkEvents.push(timestamp);
    state.blinkEvents = state.blinkEvents.filter((eventTime) => timestamp - eventTime <= TREND_WINDOW_MS);
}

function updateEyeHealth(timestamp) {
    const sessionStart = state.sessionStartTime ?? timestamp;
    const sessionMs = Math.max(0, timestamp - sessionStart);
    const sessionSeconds = sessionMs / 1000;
    const recentBlinks = state.blinkEvents.filter((eventTime) => timestamp - eventTime <= BLINK_RATE_WINDOW_MS);
    const trendBlinks = state.blinkEvents.filter((eventTime) => timestamp - eventTime <= TREND_WINDOW_MS);
    const trendMinutes = Math.max(1 / 60, Math.min(3, sessionSeconds / 60));
    const blinkTrend = trendBlinks.length / trendMinutes;

    let noBlinkMs = 0;
    if (state.calibrated) {
        const referenceTime = state.lastBlinkTime ?? sessionStart;
        noBlinkMs = Math.max(0, timestamp - referenceTime);
        state.longestNoBlinkMs = Math.max(state.longestNoBlinkMs, noBlinkMs);
    }

    let warning = "";
    if (state.latest.drowsy) {
        warning = "Drowsiness warning";
    } else if (noBlinkMs >= LONG_NO_BLINK_MS) {
        warning = "Long no-blink streak";
    } else if (sessionSeconds >= 60 && recentBlinks.length < VERY_LOW_BLINKS_PER_MIN) {
        warning = "Very low blink rate";
    } else if (sessionSeconds >= 60 && recentBlinks.length < TARGET_BLINKS_PER_MIN) {
        warning = "Low blink rate";
    }

    state.health = {
        blinkRate: recentBlinks.length,
        blinkTrend,
        noBlinkSeconds: noBlinkMs / 1000,
        sessionSeconds,
        warning
    };

    maybeSendWarningNotification(warning, timestamp);
}

function processRawEar(rawEar, timestamp) {
    state.latest.rawEar = rawEar;

    if (!state.calibrated) {
        state.calibrationEars.push(rawEar);
        state.latest.status = "Calibrating";
        state.latest.prediction = -1;
        state.latest.normEar = null;

        if (state.calibrationEars.length >= CALIBRATION_FRAMES) {
            const initMax = Math.max(...state.calibrationEars);
            state.currentMax = Number.isNaN(initMax) || initMax < 0.18 ? 0.23 : initMax;
            state.calibrated = true;
            state.latest.status = "Collecting window";
            state.sessionStartTime = timestamp;
            state.lastBlinkTime = timestamp;
        }
        updateEyeHealth(timestamp);
        return;
    }

    if (rawEar > state.currentMax) {
        state.currentMax = rawEar;
    }

    const earNorm = Math.min(Math.max(rawEar / state.currentMax, 0.0), 1.0);
    state.latest.normEar = earNorm;
    pushLimited(state.slidingWindow, earNorm, WINDOW_SIZE);

    if (state.slidingWindow.length < WINDOW_SIZE) {
        state.latest.status = "Collecting window";
        state.latest.prediction = -1;
        updateEyeHealth(timestamp);
        return;
    }

    state.totalWindows += 1;
    let prediction;

    if (Math.min(...state.slidingWindow) >= HYBRID_THRESHOLD) {
        prediction = 0;
        state.skippedMl += 1;
        state.latest.cpuMode = "Filter";
    } else {
        const features = extractAdvancedFeatures(state.slidingWindow);
        prediction = predictPipeline(features);
        state.latest.cpuMode = "SVM";
    }

    state.latest.prediction = prediction;
    pushLimited(state.predHistory, prediction, DROWSY_HISTORY_SIZE);

    const drowsyVotes = state.predHistory.filter((value) => value === 2).length;
    state.latest.drowsy = drowsyVotes >= DROWSY_VOTES_REQUIRED;
    if (state.latest.drowsy && !state.wasDrowsy) {
        state.drowsyEvents += 1;
    }
    state.wasDrowsy = state.latest.drowsy;

    if (state.latest.drowsy) {
        state.latest.status = "Drowsy";
    } else if (prediction === 1) {
        state.latest.status = "Blinking";
    } else {
        state.latest.status = "Normal";
    }

    if (state.blinkCooldown > 0) {
        state.blinkCooldown -= 1;
    }
    if (prediction === 1 && state.blinkCooldown === 0) {
        recordBlink(timestamp);
        state.blinkCooldown = 3;
    }
    updateEyeHealth(timestamp);
}

function drawLandmarks(landmarks) {
    const canvas = els.canvas;
    const video = els.video;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const color = state.latest.drowsy ? "#ff3b30" : "#32d74b";
    ctx.fillStyle = color;

    const eyeIndices = [...LEFT_EYE_EAR_IDX, ...RIGHT_EYE_EAR_IDX];
    for (const idx of eyeIndices) {
        const point = landmarks[idx];
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 2.5, 0, Math.PI * 2);
        ctx.fill();
    }
}

function clearCanvas() {
    const ctx = els.canvas.getContext("2d");
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
}

function predictionLabel(value) {
    if (value === 0) return "0 - Open";
    if (value === 1) return "1 - Blink";
    if (value === 2) return "2 - Sleep";
    return "Waiting";
}

function notificationPermission() {
    if (!("Notification" in window)) {
        return "unsupported";
    }
    return Notification.permission;
}

function updateNotificationUi() {
    const permission = notificationPermission();

    if (permission === "unsupported") {
        els.notifyBtn.disabled = true;
        els.notificationStatus.textContent = "Unsupported";
        return;
    }

    state.notificationsEnabled = permission === "granted";
    els.notifyBtn.disabled = permission === "denied";
    els.notifyBtn.textContent = permission === "granted" ? "Notifications On" : "Enable Notifications";
    els.notificationStatus.textContent = permission === "granted"
        ? "On"
        : permission === "denied"
            ? "Blocked"
            : "Off";
}

async function requestNotifications() {
    if (!("Notification" in window)) {
        els.message.textContent = "This browser does not support system notifications.";
        updateNotificationUi();
        return;
    }

    const permission = await Notification.requestPermission();
    state.notificationsEnabled = permission === "granted";
    els.message.textContent = permission === "granted"
        ? "System notifications are enabled."
        : "Notifications were not enabled. Web alerts will still appear.";
    updateNotificationUi();
}

function maybeSendWarningNotification(warning, timestamp) {
    if (!warning || !state.notificationsEnabled || notificationPermission() !== "granted") {
        return;
    }

    const notificationKey = warning;
    const lastSentAt = state.lastNotificationAt[notificationKey] ?? -Infinity;
    if (timestamp - lastSentAt < NOTIFICATION_COOLDOWN_MS) {
        return;
    }

    state.lastNotificationAt[notificationKey] = timestamp;

    const { title, body } = notificationContent(warning);
    const notification = new Notification(title, {
        body,
        tag: notificationKey,
        renotify: false,
        silent: false
    });

    if ("vibrate" in navigator) {
        navigator.vibrate([160, 80, 160]);
    }

    window.setTimeout(() => notification.close(), 8000);
}

function notificationContent(warning) {
    if (warning === "Long no-blink streak") {
        return {
            title: "BlinkGuard: chớp mắt một chút nhé",
            body: `Bạn đã không chớp mắt ${state.health.noBlinkSeconds.toFixed(0)} giây. Chớp mắt vài lần và nhìn xa khỏi màn hình.`
        };
    }

    if (warning === "Very low blink rate" || warning === "Low blink rate") {
        return {
            title: "BlinkGuard: blink rate đang thấp",
            body: `60 giây vừa rồi có ${state.health.blinkRate} lần chớp mắt. Mục tiêu hiện tại là ${TARGET_BLINKS_PER_MIN}/phút.`
        };
    }

    return {
        title: "BlinkGuard: cảnh báo buồn ngủ",
        body: "Bạn có dấu hiệu nhắm mắt lâu. Hãy nghỉ ngắn hoặc kiểm tra lại trạng thái tỉnh táo."
    };
}

function renderHud() {
    const latest = state.latest;
    const cpuSaved = state.totalWindows === 0 ? 0 : (state.skippedMl / state.totalWindows) * 100;
    const calibrationProgress = Math.min(state.calibrationEars.length, CALIBRATION_FRAMES);
    const health = state.health;
    const warning = health.warning;
    const statusText = warning || latest.status;

    els.status.textContent = statusText;
    els.status.dataset.state = warning ? "danger" : latest.status.toLowerCase();
    els.rawEar.textContent = latest.rawEar == null ? "N/A" : latest.rawEar.toFixed(3);
    els.normEar.textContent = latest.normEar == null ? "N/A" : latest.normEar.toFixed(3);
    els.prediction.textContent = predictionLabel(latest.prediction);
    els.calibration.textContent = state.calibrated ? "Ready" : `${calibrationProgress}/${CALIBRATION_FRAMES}`;
    els.cpuSaved.textContent = `${cpuSaved.toFixed(1)}%`;
    els.cpuMode.textContent = latest.cpuMode;
    els.blinkCount.textContent = String(state.blinkCount);
    els.blinkRate.textContent = `${health.blinkRate}/min`;
    els.blinkTarget.textContent = `${TARGET_BLINKS_PER_MIN}/min`;
    els.blinkTrend.textContent = `${health.blinkTrend.toFixed(1)}/min`;
    els.noBlinkStreak.textContent = `${health.noBlinkSeconds.toFixed(1)}s`;
    els.sessionTime.textContent = formatDuration(health.sessionSeconds);
    els.drowsyEvents.textContent = String(state.drowsyEvents);
    els.alert.hidden = !warning;
    els.alert.textContent = warning.toUpperCase();
    updateNotificationUi();
}

function formatDuration(totalSeconds) {
    const seconds = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
    const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
    return `${minutes}:${seconds}`;
}

async function initFaceLandmarker() {
    if (state.faceLandmarker) {
        return;
    }

    els.message.textContent = "Loading MediaPipe FaceLandmarker...";
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);

    const options = {
        baseOptions: {
            modelAssetPath: MODEL_URL
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    };

    try {
        state.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            ...options,
            baseOptions: {
                ...options.baseOptions,
                delegate: "GPU"
            }
        });
    } catch (error) {
        console.warn("GPU delegate failed, falling back to CPU.", error);
        state.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, options);
    }
}

async function startCamera() {
    await initFaceLandmarker();

    state.stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user"
        },
        audio: false
    });

    els.video.srcObject = state.stream;
    await els.video.play();

    resetPipeline();
    state.running = true;
    els.startBtn.disabled = true;
    els.stopBtn.disabled = false;
    els.resetBtn.disabled = false;
    els.message.textContent = "Camera is running. Keep eyes open during the first 15 samples.";
    loop();
}

function stopCamera() {
    state.running = false;
    if (state.animationId != null) {
        cancelAnimationFrame(state.animationId);
    }
    if (state.stream) {
        for (const track of state.stream.getTracks()) {
            track.stop();
        }
    }
    state.stream = null;
    els.video.srcObject = null;
    clearCanvas();
    els.startBtn.disabled = false;
    els.stopBtn.disabled = true;
    els.resetBtn.disabled = true;
    els.message.textContent = "Camera stopped.";
}

function loop() {
    if (!state.running) {
        return;
    }

    const now = performance.now();
    const video = els.video;

    if (video.currentTime !== state.lastVideoTime) {
        state.lastVideoTime = video.currentTime;
        const result = state.faceLandmarker.detectForVideo(video, now);
        const landmarks = result.faceLandmarks && result.faceLandmarks[0];

        if (landmarks) {
            state.latest.faceDetected = true;
            drawLandmarks(landmarks);

            if (now - state.lastSampleTime >= SAMPLE_INTERVAL_MS) {
                const leftEar = computeEar(landmarks, LEFT_EYE_EAR_IDX);
                const rightEar = computeEar(landmarks, RIGHT_EYE_EAR_IDX);
                const rawEar = Math.min(leftEar, rightEar);
                processRawEar(rawEar, now);
                state.lastSampleTime = now;
            }
        } else {
            state.latest.faceDetected = false;
            state.latest.status = "No face";
            state.latest.prediction = -1;
            state.latest.drowsy = false;
            state.wasDrowsy = false;
            state.slidingWindow = [];
            state.predHistory = [];
            updateEyeHealth(now);
            clearCanvas();
        }
        renderHud();
    }

    state.animationId = requestAnimationFrame(loop);
}

els.startBtn.addEventListener("click", async () => {
    try {
        await startCamera();
    } catch (error) {
        console.error(error);
        if (state.stream) {
            stopCamera();
        }
        state.running = false;
        els.startBtn.disabled = false;
        els.stopBtn.disabled = true;
        els.resetBtn.disabled = true;
        els.message.textContent = `Cannot start camera: ${error.message}`;
    }
});

els.stopBtn.addEventListener("click", stopCamera);
els.notifyBtn.addEventListener("click", requestNotifications);

els.resetBtn.addEventListener("click", () => {
    resetPipeline();
    els.message.textContent = "Calibration reset. Keep eyes open for 15 samples.";
});

window.addEventListener("beforeunload", stopCamera);

updateNotificationUi();
renderHud();
