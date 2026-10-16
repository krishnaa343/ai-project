# System Architecture Specification
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Document Version:** 1.0.0  
**Phase:** 16 — Final Documentation  
**Status:** Complete & Validated  

---

## 1. High-Level Architecture Overview

The system implements a decoupled, multi-tiered client-server architecture designed for high availability, low-latency inference, clear separation of concerns, and robust security isolation.

```
                                [ End User / Farmer / Agronomist ]
                                                │
                                                │ Browser Request (HTTPS)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TIER 1: FRONTEND LAYER (Vercel)                             │
│                                                                                             │
│   React 19 + Vite 8 SPA                                                                     │
│   ├── Navigation & Routing (React Router DOM: Home, Detect, Results, Dashboard)             │
│   ├── Image Upload & Client-Side Validation (MIME type, file size <= 10MB)                  │
│   ├── Responsive Diagnostics UI (Confidence meters, Disease info accordion, Action guides)  │
│   ├── Interactive 3D Anatomy Visualizer (Three.js procedural leaf mesh + OrbitControls)     │
│   └── Analytics Dashboard (Surveillance charts, KPIs, paginated history table)              │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               │ REST API Calls (JSON / Multipart Form-Data)
                                               │ Strict CORS Whitelist
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  TIER 2: BACKEND LAYER (Render)                             │
│                                                                                             │
│   Python 3.11 + Flask 3.0 + Gunicorn WSGI Server                                            │
│   ├── API Routing Blueprints (/api/health, /api/predict, /api/predictions)                  │
│   ├── Multipart File Validation (Magic byte headers, safe filenames, dimensions)            │
│   ├── Model Inference Engine:                                                               │
│   │   ├── MobileNetV2 (13-class fine-tuned model loaded as thread-safe singleton)           │
│   │   └── Preprocessing (RGB decode, resize to 224x224, float32 scaling [0, 1])             │
│   ├── Explainable AI (XAI) Engine:                                                          │
│   │   └── Grad-CAM Explainer (GradientTape backpropagation to 'out_relu' layer)             │
│   ├── Disease Knowledge Base Service (Symptoms, cause, management, prevention)              │
│   └── Ephemeral Static File Server (/static/gradcam/*.png)                                  │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               │ SQL Connections (PyMySQL / SQLAlchemy Engine)
                                               │ Parameterized Queries via Port 3306
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TIER 3: DATABASE LAYER (Railway)                            │
│                                                                                             │
│   MySQL 8.x Managed Relational Database                                                    │
│   ├── Database: `crop_disease_detection`                                                    │
│   └── Table: `prediction_history`                                                           │
│       ├── Primary Key: id (BIGINT AUTO_INCREMENT)                                           │
│       ├── Image metadata: image_name                                                        │
│       ├── Classification: crop, predicted_class, display_name, confidence, class_index      │
│       ├── Health status: is_healthy (BOOLEAN)                                               │
│       ├── Timestamp: created_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)                       │
│       └── B-Tree Indexes: idx_created_at, idx_crop, idx_predicted_class                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Data & Execution Flow

```
User Action: Select & Submit Leaf Image
   │
   ▼
[1] Frontend Client Validation
   ├── File existence & type check (JPEG, PNG, WEBP)
   ├── Size ceiling verification (<= 10 MB)
   └── Immediate canvas thumbnail generation
   │
   ▼
[2] HTTP POST /api/predict?gradcam=true (Multipart/form-data)
   │
   ▼
[3] Backend Upload Intake & Security Scrubbing
   ├── Filename sanitization via Werkzeug secure_filename
   ├── Magic bytes inspection via Pillow (verifies real image payload, blocks scripts)
   └── Temporary file write to ephemeral scratch directory
   │
   ▼
[4] Model Inference Pipeline
   ├── Image decoding to RGB array
   ├── Bilinear interpolation resizing to (224, 224, 3)
   ├── Tensor conversion & normalization (pixel / 255.0)
   ├── Singleton model feed-forward pass (MobileNetV2)
   ├── Softmax probability vector calculation (13 classes)
   └── Argmax extraction -> (class_index, predicted_class, confidence)
   │
   ▼
[5] Explainable AI (Grad-CAM) Generation
   ├── tf.GradientTape records gradients of predicted class score w.r.t. 'out_relu'
   ├── Global average pooling of gradients -> neuron importance weights (alpha_k)
   ├── Weighted linear combination of feature maps -> raw 2D heatmap
   ├── ReLU activation applied (isolate positive contributors)
   ├── Min-max normalization [0, 1] & bilateral smoothing
   ├── Jet colormap colorization & alpha-blending over original image
   └── Output saved as ephemeral static image: /static/gradcam/gradcam_<uuid>.png
   │
   ▼
[6] Disease Information Enrichment
   ├── Lookup in curated disease_information.json
   └── Extraction of display_name, symptoms, chemical/organic management, and prevention
   │
   ▼
[7] MySQL Persistence (Fault-Tolerant)
   ├── Parameterized INSERT into `prediction_history`
   ├── Try-catch wrapper: If MySQL is offline, prediction still returns to user with warning
   └── Clean up temporary input upload file from disk
   │
   ▼
[8] HTTP 200 Response Payload sent to Frontend
   │
   ▼
[9] Results Page Rendering
   ├── Prediction badge & confidence gauge
   ├── High-resolution Grad-CAM visualizer with side-by-side comparison
   ├── Interactive 3D conceptual leaf visualizer (Three.js)
   └── Full disease management & prevention accordion
   │
   ▼
[10] Real-Time Dashboard Updates
   ├── User navigates to /dashboard
   ├── Backend queries MySQL aggregates (COUNT, AVG, GROUP BY)
   └── Renders interactive surveillance graphs and searchable history table
```

---

## 3. Component Details & Boundaries

### 3.1 Frontend Subsystem (React 19 + Vite 8)
- **Deployment Platform:** Vercel Edge Network.
- **State Management:** Local React state, custom asynchronous hooks, and browser navigation state (`useNavigate`, `useLocation`).
- **Styling Architecture:** Responsive Vanilla CSS tokens, glassmorphic dark mode, fluid typography, CSS grid and flex layouts.
- **Security Posture:**
  - Zero database credentials embedded in code or bundle.
  - Zero private API keys in client-side code.
  - All communication uses HTTPS with strict path routing via `vercel.json`.

### 3.2 Backend Subsystem (Python 3.11 + Flask 3.0)
- **Deployment Platform:** Render Web Service with Gunicorn WSGI.
- **Process Model:** Single worker process (`--workers 1 --timeout 120`) to conserve container memory while serving TensorFlow graphs safely.
- **Model Lifecycle:** Singleton eager loading during Flask application factory initialization. Eager warmup ensures zero first-request inference penalty.
- **CORS Architecture:** Whitelisted explicit origins matching the production Vercel domain and local development ports. Wildcard origins (`*`) are strictly prohibited.

### 3.3 Database Subsystem (MySQL 8.x)
- **Deployment Platform:** Railway Managed MySQL Service.
- **Schema Management:** Idempotent, safe migration script (`init_db.py`) executing `CREATE TABLE IF NOT EXISTS`.
- **Query Architecture:** 100% parameterized queries using PyMySQL with DB-API compliant parameter escaping to prevent SQL injection vulnerabilities.

---

## 4. 3D Visualization Architecture: Conceptual vs Reconstruction

> ### IMPORTANT TECHNICAL CLARIFICATION
> The 3D leaf visualizer component (`Leaf3DViewer.jsx`) is a **frontend-only procedural 3D model** built using **Three.js**.
> 
> - **What it is:** A conceptual, interactive educational model representing the biological structure of the identified plant (e.g., standard ovate leaf for Apple, lobed compound leaf for Tomato, elongated blade for Corn, pinnate leaf for Potato). The surface coloration and procedural lesion patches adjust dynamically to match the disease status (healthy green vs diseased chlorotic/necrotic spots).
> - **What it is NOT:** It is **NOT** a 3D reconstruction of the user's uploaded photograph. It does **NOT** perform photogrammetry, neural radiance field (NeRF) synthesis, depth estimation, or actual physical lesion mapping.
> - **Visual Explainability Separation:** Actual pathological evidence and attention heatmaps are exclusively provided by the **Grad-CAM** module, never by the 3D mesh.

---

## 5. Security & Isolation Matrix

| Layer | Threat Vector | Mitigation Strategy Implemented |
|---|---|---|
| **Frontend** | Credential Leakage | Backend credentials completely absent from frontend codebase and Vite bundle; scan verified zero matches for `MYSQL_` or `SECRET_KEY`. |
| **Frontend** | Cross-Site Scripting (XSS) | React DOM automatic string escaping; no `dangerouslySetInnerHTML` on user input. |
| **API Boundary** | Unauthorized Domain Access | Flask-CORS strictly restricts origins to authorized frontend domains. |
| **Upload Pipeline** | Malicious File Upload | Magic header byte verification via Pillow rejects executables, scripts, or non-image payloads masquerading with `.jpg` extensions. |
| **Upload Pipeline** | Disk Exhaustion / DoS | 10 MB strict payload limit enforced at web server and Flask layer; temporary files purged in `finally:` block. |
| **Backend** | Stack Trace Disclosure | When `FLASK_DEBUG=0`, exceptions trigger generic client error messages while writing detailed traces to server-side logs only. |
| **Database** | SQL Injection | 100% parameterized queries (`%s` bindings); no raw SQL string formatting. |
| **Database** | Connection Failure Cascades | Fault-tolerant persistence wrapper ensures inference and diagnostic results are returned even if the database is temporarily unreachable. |
