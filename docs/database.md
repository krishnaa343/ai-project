# Phase 11 — MySQL Database and Prediction History

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Phase:** 11 (MySQL Database and Prediction History)  
**Status:** Complete & Verified  

---

## 1. Overview & Architecture

Phase 11 introduces persistent relational database storage using **MySQL** for crop disease prediction history.

### Decoupled Data Flow
The AI prediction pipeline and database persistence are cleanly decoupled:
1. User uploads an image via the React frontend.
2. Flask backend routes the image to the MobileNetV2 prediction model.
3. Model outputs class index, predicted class, and confidence score.
4. Grad-CAM visual explainability heatmap is computed if requested.
5. Structured disease information and agricultural recommendations are attached.
6. The prediction is persisted to MySQL in the `prediction_history` table.
7. If MySQL is unavailable, the AI prediction response still succeeds with a clear persistence warning (`database.saved: false`).

```
React Frontend
      │
      │ POST /api/predict?gradcam=true
      ▼
Flask Backend (app/routes/prediction.py)
      │
      ├──► AI Prediction Service (MobileNetV2 inference)
      │         │
      ├──► Grad-CAM Service (Feature activation heatmap)
      │         │
      ├──► Disease Info Service (Structured recommendations)
      │         │
      └──► Prediction History Service (app/services/prediction_history_service.py)
                │
                ▼
      MySQL Database (crop_disease_detection)
      Table: prediction_history
                │
                ▲
      GET /api/predictions (History API)
```

---

## 2. MySQL Requirements & Installation

- **Database Engine:** MySQL Server 8.0+
- **Default Port:** `3306`
- **Default Host:** `localhost` / `127.0.0.1`
- **Python Connector:** `PyMySQL` (with `cryptography` for `caching_sha2_password` authentication)
- **GUI Tool (Optional):** MySQL Workbench 8.0+

### Starting MySQL on Windows
MySQL runs as a Windows Service named `MySQL80` (or `MySQL`).

To verify or start MySQL from PowerShell (Run as Administrator):
```powershell
# Check service status
Get-Service -Name MySQL80

# Start service if stopped
Start-Service -Name MySQL80
```

---

## 3. Environment Configuration

Database connection settings are loaded through environment variables managed in `backend/.env`.

> [!IMPORTANT]
> The `backend/.env` file contains sensitive local credentials and is strictly excluded from version control via `.gitignore`. A safe template is provided in `backend/.env.example`.

### Variables
| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `MYSQL_HOST` | Hostname of the MySQL server | `localhost` |
| `MYSQL_PORT` | Port of the MySQL server | `3306` |
| `MYSQL_DATABASE` | Target database name | `crop_disease_detection` |
| `MYSQL_USER` | MySQL user account | `root` |
| `MYSQL_PASSWORD` | MySQL user password | `YourLocalPassword` |

---

## 4. Database Initialization

Database and table creation is completely automated, safe, and idempotent.

Run the initialization script from the `backend/` directory:
```powershell
cd backend
.\venv\Scripts\python scripts/init_db.py
```

### Idempotency Guarantee
- Uses `CREATE DATABASE IF NOT EXISTS crop_disease_detection`
- Uses `CREATE TABLE IF NOT EXISTS prediction_history`
- Re-running the script will **never** drop existing databases, delete records, or recreate tables destructively.

---

## 5. Table Schema: `prediction_history`

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `BIGINT` | `PRIMARY KEY, AUTO_INCREMENT` | Unique identifier for each prediction record |
| `image_name` | `VARCHAR(255)` | `NULLABLE` | Original uploaded file name (raw binary is NOT stored) |
| `crop` | `VARCHAR(100)` | `NOT NULL, INDEX` | Crop name (e.g. `Tomato`, `Apple`, `Corn`) |
| `predicted_class` | `VARCHAR(100)` | `NOT NULL, INDEX` | Full class name (e.g. `Tomato___Late_blight`) |
| `display_name` | `VARCHAR(150)` | `NOT NULL` | Human-readable disease name |
| `confidence` | `FLOAT` | `NOT NULL` | Model confidence probability (0.0 to 1.0) |
| `class_index` | `INT` | `NOT NULL` | Output neuron index (0 to 12) |
| `is_healthy` | `BOOLEAN` (`TINYINT(1)`) | `NOT NULL, DEFAULT FALSE` | Flag indicating healthy vs diseased plant |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP, INDEX` | UTC timestamp when prediction was recorded |

### Storage Principles
1. **No Raw Images:** Raw image binaries are **never** stored in MySQL. They are processed in temporary storage and cleaned up.
2. **Parameterized SQL:** All queries use PyMySQL parameterized queries (`%s`) preventing SQL injection vulnerabilities.
3. **Optimized Indexes:** Indexes on `created_at DESC`, `crop`, and `predicted_class` guarantee fast paginated queries.

---

## 6. Prediction History API Endpoints

### 1. List Predictions (Paginated, Newest First)
```http
GET /api/predictions?page=1&limit=20
```

**Query Parameters:**
- `page` *(optional, default: 1)*: Page number (must be >= 1).
- `limit` *(optional, default: 20)*: Page size (1 to 100).

**Response (200 OK):**
```json
{
  "status": "success",
  "predictions": [
    {
      "id": 7,
      "image_name": "tomato_leaf.jpg",
      "crop": "Tomato",
      "predicted_class": "Tomato___Late_blight",
      "display_name": "Tomato Late Blight",
      "confidence": 0.9986,
      "class_index": 11,
      "is_healthy": false,
      "created_at": "2026-10-05T14:58:24"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1,
    "total_pages": 1
  }
}
```

### 2. Get Single Prediction by ID
```http
GET /api/predictions/<id>
```

**Status Codes:**
- `200 OK`: Single prediction object returned.
- `400 Bad Request`: Invalid or non-positive ID.
- `404 Not Found`: ID does not exist in the database.
- `500 Internal Server Error`: Database connectivity failure.

---

## 7. Prediction Response Compatibility

The existing `POST /api/predict` response structure remains 100% backward compatible:
- Fields preserved: `status`, `predicted_class`, `class_index`, `confidence`, `confidence_pct`, `all_probabilities`, `inference_ms`, `gradcam`, `disease_info`.
- Added persistence fields:
  ```json
  "prediction_id": 7,
  "database": {
    "saved": true,
    "prediction_id": 7
  }
  ```

### Controlled Failure Behavior
If MySQL is stopped or unreachable during inference:
- The API **still returns HTTP 200 OK**.
- The full AI prediction, Grad-CAM visualization, and disease information are preserved.
- The `database` block clearly states:
  ```json
  "prediction_id": null,
  "database": {
    "saved": false,
    "warning": "Prediction succeeded, but database persistence failed.",
    "prediction_id": null
  }
  ```
- No artificial or fabricated prediction ID is ever generated.

---

## 8. Verifying Records in MySQL

### Using MySQL CLI
```powershell
& "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p -e "USE crop_disease_detection; SELECT id, crop, predicted_class, confidence, is_healthy, created_at FROM prediction_history ORDER BY created_at DESC;"
```

### Using MySQL Workbench
1. Open MySQL Workbench.
2. Connect to **ai project** or **Local instance MySQL80**.
3. In a query tab, execute:
   ```sql
   USE crop_disease_detection;
   SELECT * FROM prediction_history ORDER BY created_at DESC;
   ```

---

## 9. Common Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Access denied for user 'root'@'localhost'` | Incorrect password in `backend/.env` | Verify password in `backend/.env` matching Workbench connection |
| `Can't connect to MySQL server on 'localhost'` | MySQL service not running | Start the service with `Start-Service MySQL80` |
| `Unknown database 'crop_disease_detection'` | Database not initialized | Run `python backend/scripts/init_db.py` |
| Changes in `.env` not taking effect | Flask process caching old env | Restart Flask backend (`.\venv\Scripts\python run.py`) |
