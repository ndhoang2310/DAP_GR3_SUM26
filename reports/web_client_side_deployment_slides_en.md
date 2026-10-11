# Slide Content: BlinkGuard Web Client-Side Deployment

## Slide 1: Deployment Goal

**Title:** From Python Prototype to Browser-Based BlinkGuard

- Deploy the eye-state detection pipeline as a real-time web application.
- Run all inference directly in the user's browser.
- Track blink behavior and drowsiness using webcam input.
- Provide alerts when users forget to blink during long focus sessions.
- Keep the system privacy-friendly by avoiding server-side video processing.

**Speaker Note:**  
This deployment stage turns the research pipeline into a usable client-side product called BlinkGuard.

---

## Slide 2: Why Client-Side Deployment?

**Title:** Why Run Everything in the Browser?

- The final TimeSeries EAR model is a lightweight Linear SVM.
- Inference is fast enough to run directly on laptops and phones.
- No backend server is required for camera processing or model inference.
- Webcam frames stay on the user's device.
- The app can be hosted as a static website with very low deployment cost.

**Key Message:**  
Client-side deployment gives us low latency, better privacy, and almost zero server cost.

---

## Slide 3: Deployed Model

**Title:** Model Used for Web Deployment

- Source model: `best_traditional_model.pkl`
- Model type: `StandardScaler + Linear SVC`
- Input: 12 dynamic EAR-based features
- Output classes:
  - `0`: Eye open / no blink
  - `1`: Blink
  - `2`: Sleep / long closure

**Speaker Note:**  
This model was selected because it achieved strong benchmark performance while remaining small and easy to port to JavaScript.

---

## Slide 4: 12-Feature Input Vector

**Title:** Input Features for SVM Inference

- 7 normalized EAR values from a sliding window
- Minimum EAR in the window
- Maximum EAR in the window
- Standard deviation
- Center ratio:
  - `(w[0] + w[6]) / (2 * w[3] + 1e-5)`
- Kurtosis of the window

**Important:**  
The JavaScript feature extraction must match the Python training pipeline exactly.

---

## Slide 5: Converting the Model to JavaScript

**Title:** From `.pkl` Model to JavaScript Function

- Load the trained model with `joblib`.
- Extract `StandardScaler` parameters:
  - `mean_`
  - `scale_`
- Extract Linear SVM parameters:
  - `coef_`
  - `intercept_`
  - `classes_`
- Generate pure JavaScript inference code.

**Output File:**  
`deployment/scripts/svm_model.js`

---

## Slide 6: JavaScript Model Runtime

**Title:** Pure JavaScript SVM Inference

- `scaleFeatures(features)` reproduces `StandardScaler`.
- `scoreSVM(scaledFeatures)` reproduces Linear SVM scoring.
- `predictPipeline(features)` returns the final class.
- No machine learning library is required in the browser.

**Code Concept:**

```javascript
const prediction = predictPipeline(features);
```

---

## Slide 7: Deployment Folder Structure

**Title:** Static Web App Structure

```text
deployment/scripts/
├── index.html
├── app.js
├── svm_model.js
└── favicon.svg
```

- `index.html`: UI layout, camera view, dashboard, controls
- `app.js`: webcam, MediaPipe, EAR, calibration, SVM, alerts
- `svm_model.js`: converted Linear SVM model
- `favicon.svg`: BlinkGuard shield-and-eye app icon

**Key Message:**  
The entire app can be deployed as static files.

---

## Slide 8: Runtime Architecture

**Title:** Real-Time Browser Pipeline

```text
Webcam
  ↓
MediaPipe FaceLandmarker JS
  ↓
Eye landmarks
  ↓
EAR calculation
  ↓
Dynamic calibration
  ↓
7-frame sliding window
  ↓
Hybrid filter / SVM inference
  ↓
Blink and drowsiness logic
  ↓
Dashboard + notifications
```

**Speaker Note:**  
This architecture mirrors the Python realtime pipeline, but it runs fully in the browser.

---

## Slide 9: Eye Tracking with MediaPipe JS

**Title:** Browser-Based Face and Eye Tracking

- Uses `@mediapipe/tasks-vision`.
- Runs `FaceLandmarker` in video mode.
- Extracts facial landmarks directly from the webcam stream.
- No video frame is uploaded to a server.
- MediaPipe assets are loaded from CDN for MVP simplicity.

**Production Note:**  
MediaPipe WASM/model files can be self-hosted later to reduce CDN dependency.

---

## Slide 10: EAR Calculation in the Browser

**Title:** Eye Aspect Ratio on Web Landmarks

- Left eye landmarks:
  - `[33, 160, 158, 133, 153, 144]`
- Right eye landmarks:
  - `[362, 385, 387, 263, 373, 380]`
- EAR formula:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

- Final raw EAR:

```javascript
rawEAR = Math.min(leftEAR, rightEAR);
```

**Reason:**  
Using the smaller EAR improves sensitivity when one eye is partially occluded or estimated incorrectly.

---

## Slide 11: Dynamic Calibration

**Title:** Personalized EAR Normalization

- The app collects the first 15 EAR samples.
- Initial baseline:

```text
currentMax = max(first 15 raw EAR values)
```

- Fallback baseline:

```text
0.23
```

- During runtime, the baseline is updated with expanding max.
- Normalized EAR:

```javascript
earNorm = rawEAR / currentMax;
```

**Key Message:**  
Dynamic calibration adapts the system to different users and eye shapes.

---

## Slide 12: Sampling Rate and Sliding Window

**Title:** Matching the Training-Time Temporal Setup

- The model was trained on EAR sequences sampled around 15 FPS.
- The browser does not run inference on every animation frame.
- Sampling interval:

```javascript
SAMPLE_INTERVAL_MS = 1000 / 15;
```

- Sliding window size:

```javascript
WINDOW_SIZE = 7;
```

- Inference only starts after the 7-frame window is full.

**Speaker Note:**  
This keeps the web runtime consistent with the training pipeline.

---

## Slide 13: Hybrid Heuristic Filter

**Title:** Reducing Unnecessary Model Calls

- If all normalized EAR values in the window are high, the eye is considered open.

```javascript
if (Math.min(slidingWindow) >= 0.5) {
  prediction = 0;
}
```

- SVM inference only runs when the window looks suspicious.
- The dashboard shows:
  - `CPU Saved`
  - `CPU Mode`: Filter or SVM

**Key Message:**  
The hybrid filter reduces CPU usage while preserving real-time responsiveness.

---

## Slide 14: BlinkGuard Web App Features

**Title:** Current BlinkGuard Features

- Start, stop, and reset camera session
- Real-time eye landmark overlay
- Blink count during the session
- Blinks in the last 60 seconds
- 3-minute blink rate trend
- No-blink streak timer
- Drowsy event counter
- SVM prediction display
- CPU saving display
- Notification permission status

**Speaker Note:**  
The web app evolved from a drowsiness demo into a lightweight eye-health assistant.

---

## Slide 15: Blink Rate Monitoring

**Title:** Detecting Low Blink Rate

- The app stores timestamps for detected blink events.
- Blink rate is calculated from the last 60 seconds.
- Default target:

```text
12 blinks per minute
```

- Warning levels:
  - `< 12/min`: Low blink rate
  - `< 7/min`: Very low blink rate

**Use Case:**  
Warn users when they focus too long and blink less than expected.

---

## Slide 16: No-Blink Streak Warning

**Title:** Alerting Users Who Forget to Blink

- The app tracks time since the last detected blink.
- Warning threshold:

```javascript
LONG_NO_BLINK_MS = 15000;
```

- If the user does not blink for over 15 seconds:
  - Web warning is shown.
  - System notification can be sent.

**Key Message:**  
This feature targets users who stare at the screen during long focus sessions.

---

## Slide 17: Drowsiness Warning

**Title:** Detecting Long Eye Closure

- The SVM predicts class `2` for sleep / long closure.
- The app uses temporal smoothing:

```javascript
DROWSY_HISTORY_SIZE = 15;
DROWSY_VOTES_REQUIRED = 10;
```

- If enough recent predictions are class `2`, the app triggers a drowsiness warning.
- Drowsy events are counted in the dashboard.

**Speaker Note:**  
This prevents short blinks from being treated as sleep.

---

## Slide 18: System Notifications

**Title:** Alerts Outside the Web Page

- Uses the Browser Notification API.
- User enables alerts with `Enable Notifications`.
- Notifications can be sent for:
  - Long no-blink streak
  - Low blink rate
  - Drowsiness warning
- Cooldown prevents spam:

```javascript
NOTIFICATION_COOLDOWN_MS = 60000;
```

**Example Notification:**  
“You have not blinked for X seconds. Blink a few times and look away from the screen.”

---

## Slide 19: Deployment on Cloudflare Pages

**Title:** Static Hosting with Cloudflare Pages

- No backend server required.
- No build command required.
- Output directory:

```text
deployment/scripts
```

- Cloudflare provides HTTPS by default.
- HTTPS is required for webcam and notification APIs outside localhost.

**Key Message:**  
BlinkGuard can be deployed as a free or low-cost static web app.

---

## Slide 20: Deployment Summary

**Title:** Final Deployment Outcome

- Python SVM model was converted to pure JavaScript.
- MediaPipe JS handles real-time face and eye tracking.
- EAR-based TimeSeries logic runs fully in the browser.
- BlinkGuard monitors blinking behavior and drowsiness.
- System notifications extend alerts beyond the web UI.
- The app is ready for static hosting on Cloudflare Pages.

**Closing Message:**  
BlinkGuard demonstrates how a lightweight ML model can become a privacy-friendly, real-time web application without server-side inference.
