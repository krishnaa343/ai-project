# AI-Based Intelligent Crop Disease Detection Framework for Agriculture

<div align="center">
  <h3>🌿 CropScan AI — Intelligent Crop Disease Detection & Agricultural Decision Support</h3>
  <p>An end-to-end, production-ready diagnostic platform combining deep transfer learning, Explainable AI (Grad-CAM), relational surveillance analytics, and interactive 3D conceptual leaf modeling.</p>
</div>

---

## Overview

**CropScan AI** is an intelligent web-based decision-support system designed to assist farmers, agricultural extension workers, and agronomists in identifying foliar crop diseases from leaf photographs. The system classifies foliar conditions across **13 categories** covering four vital crops: **Apple**, **Corn (Maize)**, **Potato**, and **Tomato**. 

Beyond basic class prediction, the platform addresses the deep learning "black box" challenge by computing **Grad-CAM visual attention heatmaps**, provides curated agronomic disease management strategies, logs diagnoses to a managed **MySQL database**, renders epidemiological trends in an **Analytics Dashboard**, and includes a procedural **Three.js 3D conceptual leaf visualizer**.

*Formal Benchmark Accuracy:* **96.40%** across 1,499 held-out test images from the PlantVillage dataset ($F_1\text{-score} = 0.9547$).

---

## Features

- **Automated Foliar Classification:** High-precision diagnosis across 13 crop-disease categories.
- **Explainable AI (Grad-CAM):** Gradient-based visual heatmaps highlighting the exact foliar regions that influenced the model's prediction.
- **Disease Information Panel:** Actionable organic remedies, chemical treatments, conducive climatic factors, and preventative guidelines.
- **Interactive 3D Conceptual Leaf Viewer:** WebGL-accelerated 3D leaf models reflecting crop morphology and disease status with OrbitControls.
- **Relational Surveillance Logging:** Automated persistence of diagnostic transactions in MySQL (`prediction_history`).
- **Analytics Dashboard:** Real-time KPIs, healthy-to-diseased ratios, crop surveillance distributions, and searchable history tables.
- **Production-Hardened Security:** Rigorous magic-byte file validation, parameterized SQL queries, CORS origin isolation, and zero-leakage client bundles.

---

## Supported Crops

The system supports four staple agricultural crops:
1. **Apple** (*Malus domestica*)
2. **Corn / Maize** (*Zea mays*)
3. **Potato** (*Solanum tuberosum*)
4. **Tomato** (*Solanum lycopersicum*)

---

## Supported Diseases (13-Class Taxonomy)

| Index | Formal Trained Identifier | Crop | Condition | Status |
|:---:|---|---|---|:---:|
| `0` | `Apple___Apple_scab` | Apple | Apple Scab (*Venturia inaequalis*) | 🔴 Diseased |
| `1` | `Apple___Black_rot` | Apple | Black Rot (*Botryosphaeria obtusa*) | 🔴 Diseased |
| `2` | `Apple___healthy` | Apple | Healthy Foliage | 🟢 Healthy |
| `3` | `Corn_(maize)___Common_rust_` | Corn (Maize) | Common Rust (*Puccinia sorghi*) | 🔴 Diseased |
| `4` | `Corn_(maize)___Northern_Leaf_Blight` | Corn (Maize) | Northern Leaf Blight (*Exserohilum turcicum*) | 🔴 Diseased |
| `5` | `Corn_(maize)___healthy` | Corn (Maize) | Healthy Foliage | 🟢 Healthy |
| `6` | `Potato___Early_blight` | Potato | Early Blight (*Alternaria solani*) | 🔴 Diseased |
| `7` | `Potato___Late_blight` | Potato | Late Blight (*Phytophthora infestans*) | 🔴 Diseased |
| `8` | `Potato___healthy` | Potato | Healthy Foliage | 🟢 Healthy |
| `9` | `Tomato___Bacterial_spot` | Tomato | Bacterial Spot (*Xanthomonas campestris*) | 🔴 Diseased |
| `10` | `Tomato___Early_blight` | Tomato | Early Blight (*Alternaria solani*) | 🔴 Diseased |
| `11` | `Tomato___Late_blight` | Tomato | Late Blight (*Phytophthora infestans*) | 🔴 Diseased |
| `12` | `Tomato___healthy` | Tomato | Healthy Foliage | 🟢 Healthy |

---

## System Architecture

The application adopts a decoupled three-tier architecture:

```
[ User Browser ]
       │
       │ HTTPS / REST
       ▼
┌──────────────────────────────────────┐
│  Tier 1: Frontend (Vercel)           │
│  React 19 + Vite 8 Single-Page App   │
│  Three.js 3D Conceptual Visualizer   │
└──────────────────┬───────────────────┘
                   │ HTTPS API Requests (CORS Whitelisted)
                   ▼
┌──────────────────────────────────────┐
│  Tier 2: Backend (Render)            │
│  Python 3.11 + Flask 3.0 + Gunicorn  │
│  MobileNetV2 Inference Engine        │
│  Grad-CAM Explainer (OpenCV)         │
└──────────────────┬───────────────────┘
                   │ TCP 3306 (SSL Parameterized SQL)
                   ▼
┌──────────────────────────────────────┐
│  Tier 3: Database (Railway)          │
│  MySQL 8.x Managed Instance          │
│  Table: `prediction_history`         │
└──────────────────────────────────────┘
```

---

## Technology Stack

- **Frontend:** React 19, Vite 8, JavaScript (ES2022), Vanilla CSS.
- **3D Graphics:** Three.js (r186), WebGL, OrbitControls.
- **Backend Framework:** Python 3.11, Flask 3.0, Flask-CORS, Gunicorn 22.0, Werkzeug.
- **Machine Learning & Vision:** TensorFlow 2.16 (Keras v3), OpenCV, NumPy, Pillow.
- **Database Layer:** MySQL 8.x, PyMySQL, SQLAlchemy.
- **Hosting Targets:** Vercel (Frontend), Render (Backend), Railway (Database).

---

## Dataset

- **Source:** PlantVillage Agricultural Benchmark Repository.
- **Total Curated Samples:** 7,495 leaf photographs across 13 classes.
- **Partitioning:** Stratified 70% Train (5,246), 10% Validation (750), 20% Test (1,499) with fixed seed (`42`).
- **Data Augmentation:** Random horizontal flips, brightness variations ($\pm 10\%$), and contrast scaling ($\pm 10\%$).

---

## Machine Learning Model

- **Base Backbone:** MobileNetV2 (ImageNet-1k pre-trained weights).
- **Head Structure:** `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(13, activation='softmax')`.
- **Two-Stage Training:**
  - *Stage 1:* Frozen backbone, trained classification head (5 epochs, Adam $\eta = 10^{-3}$).
  - *Stage 2:* Unfrozen layers 100–154, fine-tuned (4 epochs, Adam $\eta = 10^{-5}$, EarlyStopping).
- **Inference Runtime:** Preloaded thread-safe singleton with mean CPU inference latency of **108.56 ms**.

---

## Explainable AI / Grad-CAM

To ensure diagnostic transparency:
- Computes gradients of the top predicted class score with respect to feature maps in the final convolutional layer (`out_relu`).
- Generates a coarse 2D activation grid highlighting regions that positively drove classification.
- Bilinearly upsampled to $224 \times 224$, colorized with OpenCV's Jet colormap, and alpha-blended over the input leaf image.

---

## Disease Information

Integrated knowledge base providing:
- Common and scientific binomial pathology names.
- Detailed foliar, stem, and fruit symptom descriptions.
- Conducive environmental triggers (temperature, humidity, rainfall).
- Two-tiered management advice: cultural organic practices and approved chemical remedies.
- Preventative agricultural measures and crop rotation guidance.

---

## MySQL Prediction History

All diagnoses are logged to the `prediction_history` table:
- Preserves filename, crop, predicted class, friendly name, confidence, health status, and timestamp.
- Strict parameterization prevents SQL injection attacks.
- Fault-tolerant wrapper ensures database network hiccups never block user diagnosis results.

---

## Dashboard

Located at `/dashboard`, providing real-time surveillance insights:
- **Key Metrics:** Total diagnoses, healthy/diseased counts, average confidence, health ratio progress bar.
- **Visual Analytics:** Crop surveillance distribution and disease prevalence bar charts.
- **Interactive History:** Reverse-chronological table with instant text filtering and pagination.

---

## Interactive 3D Visualization

Located on the diagnosis results view:
- Procedural crop-specific 3D leaf morphology (Apple, Corn, Potato, Tomato).
- Dynamic shader textures reflecting healthy chlorophyll versus diseased chlorosis and necrotic spot patterns.
- Interactive OrbitControls (360° rotation, dolly zoom, pan, auto-spin toggle, camera reset).
- *Technical Note:* Conceptual anatomical visualization, not an image-to-3D photogrammetric reconstruction. Includes automatic 2D fallback for devices lacking WebGL.

---

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- MySQL 8.x (local or cloud)

### Clone Repository
```bash
git clone https://github.com/your-username/crop-disease-detection.git
cd crop-disease-detection
```

---

## Environment Variables

### Backend (`backend/.env`)
```ini
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secure-random-secret-key
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=crop_disease_detection
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
```

### Frontend (`frontend/.env`)
```ini
VITE_API_BASE_URL=http://localhost:5000
VITE_MAX_FILE_SIZE_BYTES=10485760
```

---

## Running Locally

### 1. Database Initialization
```bash
python backend/scripts/init_db.py
```

### 2. Backend Startup
```bash
cd backend
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
python run.py
# Backend runs at http://localhost:5000
```

### 3. Frontend Startup
```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/api/health` | Readiness probe and model status |
| `POST` | `/api/predict?gradcam=true` | Multipart upload for diagnosis & Grad-CAM |
| `GET` | `/api/predictions?page=1&limit=20` | Paginated prediction history |
| `GET` | `/api/predictions/<id>` | Single prediction details by primary key |
| `GET` | `/static/gradcam/<file>.png` | Direct serving of generated Grad-CAM heatmaps |

---

## Testing

Run the automated backend test suite (32 unit and integration tests):
```bash
pytest backend/tests -v
```
All 32 tests pass covering API contracts, file validation, Grad-CAM generation, and MySQL persistence.

---

## Deployment

The system is configured for cloud deployment:
- **Frontend:** Vercel (`frontend/vercel.json` for SPA rewrites).
- **Backend:** Render (`render.yaml` using Gunicorn WSGI: `wsgi:app`).
- **Database:** Railway Managed MySQL instance.
- *Detailed Deployment Guide:* Refer to [docs/deployment.md](docs/deployment.md).

---

## Limitations

1. **Closed-Set Assumption:** Softmax activation forces non-leaf or unsupported images into one of the 13 supported classes.
2. **Pathological Confusion:** Visual symptom overlap between Tomato Early Blight and Late Blight causes reciprocal misclassifications.
3. **Class Skew:** Limited sample support for `Potato___healthy` (15 test samples) resulted in 68.18% precision despite 100% recall.
4. **Field Distribution Shift:** Training on laboratory images results in confidence variance when tested against complex outdoor field backgrounds.
5. **Procedural 3D Visualizer:** The 3D leaf is a conceptual model, not a 3D reconstruction of the uploaded photo.

---

## Future Improvements

- Open-Set / Out-of-Distribution (OOD) rejection via Mahalanobis distance scoring.
- Fine-tuning on in-the-wild agricultural datasets (e.g., PlantDoc).
- Lesion severity quantification using semantic segmentation (U-Net).
- Expanding class coverage to staple grains (Rice, Wheat) and abiotic stressors.
- Offline edge inference using TensorFlow Lite on mobile devices.

---

## Academic Disclaimer
*CropScan AI is an automated decision-support framework intended for educational, research, and advisory purposes. It is not an absolute substitute for certified agricultural laboratory phytopathological diagnosis.*
