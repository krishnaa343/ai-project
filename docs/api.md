# REST API Specification & Reference
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Document Version:** 1.0.0  
**Phase:** 16 — Final Documentation  
**Status:** Complete & Standardized  

---

## 1. Overview & General Conventions

The CropScan AI backend provides a RESTful API communicating via HTTP/HTTPS.

- **Base URL (Local):** `http://localhost:5000`
- **Base URL (Production):** `https://cropscan-ai-backend.onrender.com`
- **Content Type:** `application/json` (except image upload which uses `multipart/form-data`)
- **CORS Support:** Enabled for authorized origins configured in `ALLOWED_ORIGINS`.
- **Character Encoding:** UTF-8

---

## 2. API Endpoints

### 2.1 Service Health & Readiness
- **Route:** `GET /api/health`
- **Description:** Verifies service readiness, model loading state in memory, and the number of active classification categories. Polled by frontend before initiating diagnosis.
- **Request Parameters:** None.
- **Response `200 OK`:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "num_classes": 13
}
```

---

### 2.2 Image Prediction & Explainability
- **Route:** `POST /api/predict`
- **Description:** Accepts an uploaded leaf image, validates format/integrity, performs deep learning classification, optionally generates a Grad-CAM attention heatmap, attaches disease guidance, and persists diagnosis metadata to MySQL.
- **Query Parameters:**
  - `gradcam` *(optional, boolean)*: Set to `true` to generate and attach Grad-CAM heatmap. Defaults to `false`.
- **Request Headers:**
  - `Content-Type: multipart/form-data`
- **Request Body:**
  - `file`: Binary image file (JPEG, PNG, or WEBP; maximum size 10 MB).

- **Response `200 OK`:**
```json
{
  "status": "success",
  "predicted_class": "Tomato___Late_blight",
  "class_index": 11,
  "confidence": 0.8884,
  "confidence_pct": 88.84,
  "inference_ms": 31.8,
  "all_probabilities": [
    0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0003, 0.0452, 0.0001, 0.0012, 0.0642, 0.8884, 0.0001
  ],
  "gradcam": {
    "image_url": "/static/gradcam/gradcam_aa332c480edf42daa998559829723c3c.png",
    "target_layer": "out_relu"
  },
  "disease_info": {
    "crop": "Tomato",
    "display_name": "Tomato Late Blight",
    "scientific_name": "Phytophthora infestans",
    "is_healthy": false,
    "symptoms": [
      "Large, irregular water-soaked lesions on leaves and stems",
      "White fungal-like growth on the undersides of leaves in humid conditions",
      "Rapid browning and collapse of entire leaf clusters"
    ],
    "causes": "Oomycete pathogen Phytophthora infestans favored by cool, wet weather.",
    "management": {
      "cultural": ["Ensure wide plant spacing", "Avoid overhead irrigation", "Destroy infected crop debris"],
      "chemical": ["Apply copper-based fungicides or chlorothalonil preventively"]
    },
    "prevention": ["Plant certified resistant cultivars", "Practice crop rotation with non-solanaceous crops"]
  },
  "database": {
    "saved": true,
    "prediction_id": 104
  },
  "prediction_id": 104
}
```

- **Error Responses:**
  - `400 Bad Request` (Invalid or missing file):
    ```json
    {
      "status": "error",
      "message": "File content does not match a supported image format (detected: unknown)."
    }
    ```
  - `400 Bad Request` (File exceeds 10 MB ceiling):
    ```json
    {
      "status": "error",
      "message": "File size exceeds the 10 MB maximum limit."
    }
    ```
  - `500 Internal Server Error` (Server-side exception; sanitized in production):
    ```json
    {
      "status": "error",
      "message": "An error occurred while processing the prediction. Please try again."
    }
    ```

---

### 2.3 Paginated Prediction History
- **Route:** `GET /api/predictions`
- **Description:** Retrieves paginated historical prediction records from MySQL, ordered in reverse-chronological order (newest first). Used by the Analytics Dashboard.
- **Query Parameters:**
  - `page` *(optional, integer)*: 1-indexed page number (default: `1`).
  - `limit` *(optional, integer)*: Number of items per page between 1 and 500 (default: `20`).

- **Response `200 OK`:**
```json
{
  "status": "success",
  "predictions": [
    {
      "id": 104,
      "image_name": "tomato_field_leaf.jpg",
      "crop": "Tomato",
      "predicted_class": "Tomato___Late_blight",
      "display_name": "Tomato Late Blight",
      "confidence": 0.8884,
      "class_index": 11,
      "is_healthy": false,
      "created_at": "2026-10-05T23:15:53"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 104,
    "total_pages": 6
  }
}
```

- **Error Responses:**
  - `400 Bad Request` (Invalid query parameter):
    ```json
    {
      "status": "error",
      "message": "Query parameter 'page' must be a positive integer."
    }
    ```
  - `500 Internal Server Error` (Database unavailable):
    ```json
    {
      "status": "error",
      "message": "Could not retrieve prediction history from the database."
    }
    ```

---

### 2.4 Single Prediction Record by ID
- **Route:** `GET /api/predictions/<id>`
- **Description:** Retrieves full historical diagnosis details for a specific record ID.
- **URL Parameters:**
  - `id` *(integer, required)*: The unique primary key identifier of the record.

- **Response `200 OK`:**
```json
{
  "status": "success",
  "prediction": {
    "id": 104,
    "image_name": "tomato_field_leaf.jpg",
    "crop": "Tomato",
    "predicted_class": "Tomato___Late_blight",
    "display_name": "Tomato Late Blight",
    "confidence": 0.8884,
    "class_index": 11,
    "is_healthy": false,
    "created_at": "2026-10-05T23:15:53"
  }
}
```

- **Error Responses:**
  - `404 Not Found` (Record does not exist):
    ```json
    {
      "status": "error",
      "message": "Prediction with ID 9999 was not found."
    }
    ```

---

### 2.5 Static Grad-CAM Asset Route
- **Route:** `GET /static/gradcam/<filename>.png`
- **Description:** Direct static file serving of generated visual explanation heatmaps.
- **Response `200 OK`:** Raw binary PNG image with `Content-Type: image/png` and CORS headers allowing client-side rendering.
