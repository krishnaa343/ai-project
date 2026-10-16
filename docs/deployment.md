# Deployment Guide
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Phase:** 15 — Production Deployment
**Architecture:** Vercel (Frontend) + Render (Backend) + Railway (MySQL)

---

## 1. Architecture Overview

```
User (Browser)
      │
      ▼
┌─────────────────────────┐
│   Vercel (Frontend)     │  React + Vite SPA
│   https://your-app.vercel.app │
└────────────┬────────────┘
             │ API requests (HTTPS)
             ▼
┌─────────────────────────┐
│   Render (Backend)      │  Python 3.11 + Flask + Gunicorn
│   https://your-api.onrender.com │
│   ├── /api/health       │
│   ├── /api/predict      │  TensorFlow/Keras MobileNetV2
│   ├── /api/predictions  │  Grad-CAM via OpenCV
│   └── /static/gradcam/  │  Ephemeral generated images
└────────────┬────────────┘
             │ TCP/3306
             ▼
┌─────────────────────────┐
│   Railway (MySQL)       │  crop_disease_detection DB
│   prediction_history    │  table
└─────────────────────────┘
```

**Security principle:**
- Frontend ↔ Backend: HTTPS only
- Backend ↔ Database: Internal network (Railway private networking preferred)
- Database credentials: Backend environment variables only, never in frontend

---

## 2. Local Development Setup

### Prerequisites
- Python 3.11
- Node.js 18+
- MySQL 8.x running locally

### Backend (Flask)
```bash
cd backend
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
cp .env.example .env               # Edit .env with your local MySQL credentials
python backend/scripts/init_db.py  # Create DB and table
python run.py                      # Starts at http://localhost:5000
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
# .env already exists with VITE_API_BASE_URL=http://localhost:5000
npm run dev                        # Starts at http://localhost:5173
```

### ML Model
The model (`ml/models/best_model.keras`) is required by the backend.
It is excluded from Git due to its size. Ensure it is present before starting.

---

## 3. Production Deployment

### Step 1: MySQL Database — Railway

1. Go to [railway.app](https://railway.app) → New Project → MySQL
2. Copy these values from the Railway dashboard:
   - `MYSQLHOST` → use as `MYSQL_HOST`
   - `MYSQLPORT` → use as `MYSQL_PORT`
   - `MYSQLDATABASE` → use as `MYSQL_DATABASE`
   - `MYSQLUSER` → use as `MYSQL_USER`
   - `MYSQLPASSWORD` → use as `MYSQL_PASSWORD`
3. Run `init_db.py` pointed at Railway MySQL to create the `prediction_history` table:
   ```bash
   MYSQL_HOST=<railway-host> MYSQL_USER=<user> MYSQL_PASSWORD=<pass> \
   python backend/scripts/init_db.py
   ```

### Step 2: Backend — Render

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repository
3. Configure:
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 wsgi:app`
4. Set these **Environment Variables** in the Render dashboard:

| Variable | Value |
|----------|-------|
| `FLASK_ENV` | `production` |
| `FLASK_DEBUG` | `0` |
| `SECRET_KEY` | *Strong random string* |
| `ALLOWED_ORIGINS` | *Your Vercel frontend URL* (set after Step 3) |
| `MYSQL_HOST` | *Railway MySQL host* |
| `MYSQL_PORT` | `3306` |
| `MYSQL_DATABASE` | `crop_disease_detection` |
| `MYSQL_USER` | *Railway MySQL user* |
| `MYSQL_PASSWORD` | *Railway MySQL password* |

> **Important:** The ML model file (`best_model.keras`) must be uploaded to Render.
> Options:
> - Option A: Commit the model to Git (large file — use Git LFS)
> - Option B: Upload via Render disk (Render → Service → Disk → Mount to `/opt/model`)
>   Then set env var `MODEL_PATH=/opt/model/best_model.keras`
> - Option C: Download from cloud storage (S3/GCS) at startup

5. Set **Health Check Path:** `/api/health`
6. Deploy → wait for health check to pass

### Step 3: Frontend — Vercel

1. Go to [vercel.com](https://vercel.com) → New Project → Import from GitHub
2. Set **Root Directory:** `frontend`
3. Framework: **Vite** (auto-detected)
4. Set these **Environment Variables** in the Vercel dashboard:

| Variable | Value |
|----------|-------|
| `VITE_API_BASE_URL` | *Your Render backend URL* (e.g., `https://cropscan-ai-backend.onrender.com`) |
| `VITE_MAX_FILE_SIZE_BYTES` | `10485760` |

5. Deploy → get your Vercel URL

### Step 4: Update Backend CORS

After getting your Vercel URL:
1. Go to Render dashboard → Environment Variables
2. Update `ALLOWED_ORIGINS` to your Vercel URL:
   ```
   ALLOWED_ORIGINS=https://your-app.vercel.app
   ```
3. Redeploy the backend

---

## 4. Environment Variables Reference

### Backend (.env / Render Dashboard)
| Variable | Required | Description |
|----------|---------|-------------|
| `FLASK_ENV` | Yes | `development` or `production` |
| `FLASK_DEBUG` | Yes | `1` (dev) or `0` (prod) |
| `FLASK_PORT` | No | Default: 5000 |
| `SECRET_KEY` | Yes | Flask session secret — strong random value |
| `ALLOWED_ORIGINS` | Yes | Comma-separated frontend origins |
| `MYSQL_HOST` | Yes | MySQL server hostname |
| `MYSQL_PORT` | No | Default: 3306 |
| `MYSQL_DATABASE` | Yes | `crop_disease_detection` |
| `MYSQL_USER` | Yes | MySQL username |
| `MYSQL_PASSWORD` | Yes | MySQL password |

### Frontend (Vercel Dashboard / frontend/.env)
| Variable | Required | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | Yes | Backend base URL (no trailing slash) |
| `VITE_MAX_FILE_SIZE_BYTES` | No | Default: 10485760 (10 MB) |

> ⚠️ Never put `MYSQL_*`, `SECRET_KEY`, or any backend secret in `VITE_*` variables.
> Anything prefixed with `VITE_` is embedded in the public JavaScript bundle.

---

## 5. Model Deployment

The trained model is:
- **File:** `ml/models/best_model.keras` (24.75 MB)
- **Architecture:** MobileNetV2 (ImageNet weights fine-tuned on PlantVillage)
- **Classes:** 13 crop disease classes
- **Input:** 224 × 224 × 3 RGB, normalized [0, 1]

The model is **not committed to Git** (excluded by `.gitignore`).

For Render deployment, you must make the model available. The recommended approach is a **Render Disk**:
1. Create a Render Disk for the service
2. Mount at `/opt/render/project/src/ml/models/`
3. Upload `best_model.keras` and `class_indices.json` to the disk
4. The path matches what `model_service.py` expects automatically

---

## 6. Grad-CAM Static Files

Grad-CAM images are generated at runtime and stored in `backend/static/gradcam/`.

- **On Render:** The `/static/gradcam/` directory is **ephemeral** (cleared on redeploy)
- Generated images are accessible immediately after prediction via `/static/gradcam/gradcam_*.png`
- This is acceptable since Grad-CAM images are session-specific results, not permanent data
- The `prediction_history` table stores prediction metadata but not Grad-CAM images

**Known Limitation:** Grad-CAM images are lost on Render redeploys. This does not affect prediction accuracy or database records.

---

## 7. Database Schema

The `prediction_history` table is created by `backend/scripts/init_db.py`.
Run this once against the production database before starting the backend.

```sql
CREATE TABLE IF NOT EXISTS prediction_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NULL,
    crop VARCHAR(100) NOT NULL,
    predicted_class VARCHAR(100) NOT NULL,
    display_name VARCHAR(150) NOT NULL,
    confidence FLOAT NOT NULL,
    class_index INT NOT NULL,
    is_healthy BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at DESC),
    INDEX idx_crop (crop),
    INDEX idx_predicted_class (predicted_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**Production starts with zero prediction records.** Development records (103 test records) are NOT migrated to production.

---

## 8. Build Commands Reference

### Frontend
```bash
cd frontend
npm install          # Install dependencies
npm run dev          # Development server (localhost:5173)
npm run build        # Production build → dist/
npm run preview      # Preview production build locally
```

### Backend
```bash
cd backend
pip install -r requirements.txt    # Install dependencies
python run.py                      # Development server (localhost:5000)
# Production (Render uses this automatically):
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 wsgi:app
```

---

## 9. Health Check

After deployment, verify the backend is healthy:
```bash
curl https://your-api.onrender.com/api/health
```
Expected response:
```json
{"model_loaded": true, "num_classes": 13, "status": "ok"}
```

---

## 10. Post-Deployment Test Checklist

- [ ] `GET /api/health` returns `{"status": "ok", "model_loaded": true, "num_classes": 13}`
- [ ] `POST /api/predict` (with a leaf image) returns prediction + disease_info + gradcam
- [ ] `GET /api/predictions` returns records from production MySQL
- [ ] Frontend home page loads
- [ ] Frontend detect page loads with uploader
- [ ] Upload → prediction → results page works end-to-end
- [ ] Grad-CAM image loads (no 403/404 errors)
- [ ] 3D leaf viewer renders on results page
- [ ] Dashboard loads with real data
- [ ] CORS: no cross-origin errors in browser console

---

## 11. CORS Configuration

Development origins (local `.env`):
```
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Production (Render environment variable):
```
ALLOWED_ORIGINS=https://your-app.vercel.app
```

Do NOT use `*` for `ALLOWED_ORIGINS` in production.

---

## 12. Security Checklist

- [ ] `backend/.env` is in `.gitignore` and not committed
- [ ] `frontend/.env` is in `.gitignore` and not committed
- [ ] `SECRET_KEY` is a strong random value in production
- [ ] `FLASK_DEBUG=0` in production
- [ ] Database password not in any frontend file
- [ ] CORS restricted to specific Vercel domain
- [ ] SQL queries use `%s` parameterized placeholders (no concatenation)
- [ ] Uploaded files validated by extension + size + content-type
- [ ] Uploaded files deleted after inference (no permanent storage)
- [ ] API returns generic error messages (no stack traces)

---

## 13. Rollback / Redeployment

### Frontend (Vercel)
- Go to Vercel dashboard → Deployments → Select any previous deployment → Redeploy

### Backend (Render)
- Go to Render dashboard → Service → Deploys → Rollback to previous deploy

### Database
- No rollback needed for schema (CREATE IF NOT EXISTS is idempotent)
- Prediction records are append-only — no data is deleted on redeploy

---

## 14. Known Deployment Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Grad-CAM images lost on Render redeploy | Session images disappear | Regenerate on next prediction |
| Render free tier spins down after inactivity | Cold start ~30s | Use paid tier or keep-alive ping |
| TensorFlow model ~24.75 MB | Must be manually uploaded to Render disk | Use Render disk or Git LFS |
| Model runs on CPU only | ~2–3s per inference | Use Render GPU instance for production |
| Render free tier: 512 MB RAM | TF model loads ~400–500 MB | Upgrade to Starter plan ($7/mo) |

---

## 15. Support & Troubleshooting

**Backend not starting on Render:**
- Check build logs for missing dependencies
- Verify `FLASK_ENV=production` is set
- Verify `wsgi.py` exists in `/backend/`

**MySQL connection refused:**
- Verify Railway MySQL is running
- Verify `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD` are correct in Render env vars
- Run `init_db.py` against production MySQL first

**CORS error in browser:**
- Verify `ALLOWED_ORIGINS` in Render matches the exact Vercel domain
- No trailing slash in the origin URL

**Model not found:**
- Verify `best_model.keras` is on the Render disk at the expected path
- Check `ml/models/model_metadata.json` for the stored path
