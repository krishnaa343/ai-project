# Phase 9 — Full System Integration Documentation

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Phase:** 9 (Full System Integration and End-to-End Validation)  
**Status:** Complete  

---

## 1. System Architecture

```
User Browser (http://localhost:5173)
         │
         │  HTTP (same-origin via Vite proxy in dev)
         ▼
┌─────────────────────────────────────────┐
│           Vite Dev Server (:5173)       │
│  /api/*    ──────────────────────────►  │
│  /static/* ──────────────────────────►  │  Proxy
└─────────────────────────────────────────┘
         │
         ▼  HTTP (proxied)
┌─────────────────────────────────────────┐
│           Flask REST API (:5000)        │
│  GET  /api/health                       │
│  POST /api/predict?gradcam=true/false   │
│  GET  /static/gradcam/<uuid>.png        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Prediction Service             │
│  validate upload → save temp file      │
│  preprocess: BGR→RGB → tf.resize 224²  │
│  normalize to [0.0, 1.0]               │
│  batch expand → model.predict()        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          MobileNetV2 Model              │
│  ml/models/best_model.keras             │
│  13 output classes (softmax)           │
│  loaded once at startup (singleton)    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Grad-CAM Service               │
│  GradCAMExplainer (ml/explainability/) │
│  target layer: out_relu                │
│  saves overlay PNG → backend/static/   │
│        gradcam/<uuid>.png              │
└─────────────────────────────────────────┘
         │
         ▼  JSON response
┌─────────────────────────────────────────┐
│          React Results Page             │
│  PredictionCard: disease, confidence   │
│  GradCamViewer: original + heatmap     │
│  "Analyse Another" → reset state       │
└─────────────────────────────────────────┘
```

---

## 2. API Contract (Real — verified from source)

### `GET /api/health`

**Response 200:**
```json
{
  "status":       "ok",
  "model_loaded": true,
  "num_classes":  13
}
```

---

### `POST /api/predict?gradcam=true|false`

**Request:**
- Content-Type: `multipart/form-data`
- Field name: **`file`** (image file)
- Accepted types: `jpg`, `jpeg`, `png`, `webp`
- Max size: 10 MB

**Response 200 (success):**
```json
{
  "status":            "success",
  "predicted_class":   "Tomato___Early_blight",
  "class_index":       10,
  "confidence":        0.9421,
  "confidence_pct":    94.21,
  "all_probabilities": {
    "Tomato___Early_blight": 0.9421,
    "Tomato___Late_blight": 0.0421,
    "...": "..."
  },
  "inference_ms":      105.3,
  "gradcam": {
    "image_url":    "/static/gradcam/gradcam_<uuid>.png",
    "target_layer": "out_relu"
  }
}
```
> `gradcam` is `null` when `?gradcam=false`.  
> `inference_ms` is `null` when Grad-CAM path is used (full pipeline timing not separately measured).

**Response 400 (validation error):**
```json
{
  "status":  "error",
  "message": "Human-readable validation error"
}
```

**Response 500 (server error):**
```json
{
  "status":  "error",
  "message": "Prediction error: ..."
}
```

---

## 3. Known Class Mapping (13 classes)

Source: `ml/data/class_indices.json`

| Index | Class Name |
|---|---|
| 0 | Apple___Apple_scab |
| 1 | Apple___Black_rot |
| 2 | Apple___healthy |
| 3 | Corn_(maize)___Common_rust_ |
| 4 | Corn_(maize)___Northern_Leaf_Blight |
| 5 | Corn_(maize)___healthy |
| 6 | Potato___Early_blight |
| 7 | Potato___Late_blight |
| 8 | Potato___healthy |
| 9 | Tomato___Bacterial_spot |
| 10 | Tomato___Early_blight |
| 11 | Tomato___Late_blight |
| 12 | Tomato___healthy |

---

## 4. Preprocessing Pipeline (Must Match Training)

All preprocessing in **prediction_service.py** and **ml/explainability/image_utils.py**:
1. Load image from disk via OpenCV (`cv2.imread`)
2. Convert BGR → RGB
3. `tf.image.resize` to `(224, 224)`
4. Normalize to `[0.0, 1.0]` (divide by 255.0)
5. Expand batch dimension: `(1, 224, 224, 3)`

This is **identical** to Phase 3/4/5/6 training preprocessing.

---

## 5. Grad-CAM Flow

1. `GradCAMExplainer.explain(image_path, class_index=None)` called
2. Predicted class index used as target if none provided
3. Gradient tape applied to `out_relu` layer
4. Heatmap computed, JET colormap applied
5. Blended overlay saved to `backend/static/gradcam/gradcam_<uuid>.png`
6. URL returned as `/static/gradcam/gradcam_<uuid>.png` (relative)
7. Vite proxy serves the image in development

---

## 6. Environment Configuration

### Backend
- No `.env` needed for local development
- Uses defaults: port 5000, debug mode on
- CORS origin: `http://localhost:5173`

### Frontend (`frontend/.env`)
```ini
VITE_API_BASE_URL=http://localhost:5000
VITE_MAX_FILE_SIZE_BYTES=10485760
```
> `VITE_API_BASE_URL` is used only as a fallback for full URL construction.  
> In development, the Vite proxy handles `/api/*` and `/static/*` transparently.

### `frontend/.env.example`
```ini
VITE_API_BASE_URL=http://localhost:5000
VITE_MAX_FILE_SIZE_BYTES=10485760
```

---

## 7. CORS Configuration

Configured in `backend/app/__init__.py`:
```python
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173"]}})
```

- CORS is applied **only** to `/api/*` routes
- `/static/*` is not CORS-controlled — the Vite proxy handles it in development
- Do NOT broad-allow `*` origins in production

---

## 8. How to Start the System

### Step 1 — Start Flask Backend
```bash
cd backend
venv\Scripts\python run.py
```
- Runs on `http://127.0.0.1:5000`
- Model loads at startup (MobileNetV2, ~3–5 seconds)
- Verify: `GET http://localhost:5000/api/health`

### Step 2 — Start React Frontend
```bash
cd frontend
npm install        # first time only
npm run dev
```
- Runs on `http://localhost:5173`
- Vite proxy forwards `/api/*` and `/static/*` to Flask

### Step 3 — Open Browser
Navigate to: **http://localhost:5173**

---

## 9. End-to-End Testing Procedure

1. Open browser → `http://localhost:5173`
2. Click **Diagnose Leaf Now** → navigate to `/detect`
3. Upload a crop leaf image (JPG/PNG/WebP ≤ 10 MB)
4. Verify image preview appears with filename and file size
5. Ensure **Generate Grad-CAM Visual Heatmap** checkbox is checked
6. Click **Run AI Diagnosis**
7. Loading indicator shows pipeline status messages
8. Results page (`/results`) shows:
   - Predicted disease name (parsed from class format)
   - **"Model confidence: XX.XX%"** badge (green/amber/red)
   - Inference latency in ms
   - Probability distribution bars for all 13 classes
   - Side-by-side Grad-CAM viewer with original + heatmap
   - Scientific disclaimer for Grad-CAM
9. Click **Diagnose Another Leaf** → returns to `/detect` with state reset
10. Select a second image, repeat

---

## 10. Integration Bugs Found and Fixed in Phase 9

### Fix 1 — Wrong repo root in `backend/app/__init__.py`
**Problem:** `_REPO_ROOT = Path(__file__).resolve().parents[3]` resolved to `Desktop` instead of the project root.  
**Impact:** `ml.*` sys.path injection was pointing to the wrong directory.  
**Fix:** Changed `parents[3]` → `parents[2]` (correct for `backend/app/__init__.py`).

### Fix 2 — `predictDisease` API function signature mismatch
**Problem:** `Detection.jsx` called `predictDisease(file, { includeGradcam, targetLayer })` but `api.js` expected `predictDisease(file, signal)` where second argument is an AbortSignal. The `includeGradcam` state was silently ignored.  
**Fix:** Updated `api.js` `predictDisease` to accept `options = { includeGradcam, signal }`. The `gradcam` query parameter is now correctly controlled by the caller.

### Fix 3 — Grad-CAM URL double-prefixing in `resolveGradCamUrl`
**Problem:** `resolveGradCamUrl` was prepending `API_BASE` (`http://localhost:5000`) to root-relative paths like `/static/gradcam/...`. This bypassed the Vite proxy and caused CORS issues since the image was fetched directly from port 5000 which does not have CORS headers on `/static/*`.  
**Fix:** Updated `resolveGradCamUrl` to return root-relative paths as-is. Vite proxy handles them correctly in development.

---

## 11. Error Handling

| Scenario | Frontend Behavior |
|---|---|
| No file selected | Client-side validation before submit |
| Unsupported file type | Immediate validation error in uploader |
| File too large | Immediate size validation error |
| Backend offline | `checkHealth()` shows warning banner on `/detect`; `predictDisease` shows network error |
| Backend 400 | User-friendly error message from `data.message` |
| Backend 500 | Generic "server error" message, no stack trace exposed |
| Grad-CAM image missing | `<img onError>` handler shows fallback with direct-link button |
| Direct `/results` without state | Shows "No Diagnosis Found" with link back to `/detect` |

---

## 12. Known Limitations

1. **No upload progress bar** — large images show no progress during upload
2. **No prediction cancellation** — once analysis starts, it cannot be cancelled mid-request
3. **Grad-CAM overlay only** — no separate heatmap-only file is returned; the overlay is the single PNG
4. **In-memory state loss on browser refresh** — `/results` page state is in React Router `location.state` and is lost on hard refresh (handled gracefully with redirect)
5. **No Grad-CAM for non-gradcam requests** — when `?gradcam=false` the results page shows "No Grad-CAM explanation was generated"
6. **Static Grad-CAM files accumulate** — no automatic cleanup of old Grad-CAM PNGs from `backend/static/gradcam/`

---

## 13. Files Modified in Phase 9

| File | Change |
|---|---|
| `backend/app/__init__.py` | Fixed `_REPO_ROOT` from `parents[3]` → `parents[2]` |
| `frontend/src/services/api.js` | Fixed `predictDisease` signature to accept options object; fixed `resolveGradCamUrl` to return relative paths |
| `docs/integration.md` | Created (this file) |

No model files, training code, or dataset files were modified.
