# Phase 12 — Dashboard & Analytics

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Phase:** 12 (Dashboard & Analytics)  
**Status:** Complete & Verified  

---

## 1. Dashboard Purpose

Phase 12 delivers an interactive, real-time analytics dashboard for agricultural surveillance. It provides stakeholders, agronomists, and farmers with an aggregated visual overview of crop pathology history, health ratios, crop-specific disease patterns, and individual diagnostic logs.

All metrics, ratios, charts, and table rows are derived **strictly from real persistent MySQL records** stored during inference in Phase 11. No mock, synthetic, or hardcoded analytics numbers are used.

---

## 2. Dedicated Route & Navigation

- **Route:** `/dashboard`
- **Navigation:** Accessible from the top navbar via the **"Dashboard"** link on desktop, tablet, and mobile drawer.
- **Workflow Interoperability:** Farmers can navigate between **"Home"**, **"Detect Disease"** (`/detect`), **"Diagnostic Results"** (`/results`), and **"Dashboard"** (`/dashboard`) with complete state preservation.

---

## 3. Data Architecture & Security

The React dashboard strictly communicates through the Flask REST API layer and **never connects directly to MySQL**.

```
React Dashboard (/dashboard)
          │
          │ HTTP GET /api/predictions?page=1&limit=500
          ▼
Flask REST API Backend (app/routes/history.py)
          │
          │ Parameterized SQL queries
          ▼
MySQL Database (crop_disease_detection)
Table: prediction_history
```

### Security & Credential Isolation:
- No database credentials (`MYSQL_PASSWORD`, `MYSQL_USER`, `MYSQL_HOST`) exist anywhere in client-side code or bundles.
- Verified with automated security tests scanning all frontend source files.
- No raw image binaries are stored or retrieved over the wire.

---

## 4. API Endpoints Used

| Endpoint | Method | Query Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/predictions` | `GET` | `page=1&limit=500` | Retrieves prediction records ordered newest first (`created_at DESC, id DESC`). Supports up to 500 records per fetch for accurate client-side metric derivation. |
| `/api/predictions/<id>`| `GET` | — | Retrieves single prediction details by primary key ID. |

---

## 5. Summary Metrics (KPI Cards)

Five key performance indicator cards are rendered at the top of the dashboard:

1. **Total Predictions:** Total number of actual diagnosis records stored in the database.
2. **Diseased Predictions:** Total samples where `is_healthy == false`, with percentage of total diagnoses.
3. **Healthy Predictions:** Total samples where `is_healthy == true`, with percentage of total diagnoses.
4. **Average Confidence:** Arithmetic mean of model confidence probabilities across all stored records, formatted as a percentage (e.g. `94.62%`).
5. **Most Prevalent Disease:** Highest frequency disease diagnosis (excluding healthy leaf samples). Displays user-friendly `display_name` and total occurrence count. If only healthy records exist, displays `N/A`.

---

## 6. Real-Time Chart Visualizations

All charts dynamically adapt to whatever crops and disease classes exist in MySQL.

### 1. Plant Health Ratio (Donut Chart)
- **Visualization:** Clean, responsive SVG circle with computed stroke-dasharray and stroke-dashoffset.
- **Metrics:** Displays Healthy vs. Diseased proportions with a central sample counter and interactive color-coded legend.

### 2. Surveillance by Crop
- **Visualization:** Proportional horizontal bar tracks dynamically generated for each unique crop found in the database (e.g., Tomato, Corn, Apple, Potato).
- **Metrics:** Sample count and percentage share per crop type.

### 3. Condition & Disease Class Distribution
- **Visualization:** Ranked horizontal bar chart displaying all diagnosed conditions.
- **Features:** Color-coded status dots (green for healthy, red/amber for pathogen), display names, crop tags, exact sample counts, and percentages.

---

## 7. Recent Predictions History Table

A data table displaying chronological diagnostic records:
- **Columns:**
  - `ID`: Primary key record identifier (`#ID`).
  - `Crop`: Monitored agricultural crop.
  - `Diagnosis`: Human-readable `display_name` with underlying model class code.
  - `Health Status`: Visual badges (`🌿 Healthy` vs. `🍂 Disease Detected`).
  - `Confidence`: Formatted percentage with color-coded certainty meter (`high` >= 80%, `med` >= 50%, `low` < 50%).
  - `Recorded Date & Time`: Formatted local timestamp (`MMM D, YYYY, h:mm A`).
  - `Sample File`: Uploaded sample filename.

---

## 8. Search & Filtering Functionality

- **Live Text Search:** Instant, multi-field filtering matching across `crop`, `display_name`, `predicted_class`, and record `id`. Includes a quick clear button (`×`).
- **Health Status Filter:** Toggle buttons for `All`, `Healthy`, and `Diseased`.
- **Client-Side Pagination:** 10 records per page with Previous / Next navigation controls, current page indicator, and record range info.

---

## 9. Loading, Error & Empty States

1. **Loading State:** Shimmering animated skeleton placeholders for summary cards, charts, and table rows while data is being fetched.
2. **Error State:** Informative callout with alert icon and a prominent `🔄 Retry Connection` button if the backend or database is unreachable.
3. **Empty Database State:** Clean empty state illustration explaining that the database is initialized but empty, with a direct CTA button leading to `/detect` to analyze the first leaf.

---

## 10. Verification & Automated Test Results

- **Automated Backend Pytest Suite:** **32/32 tests passed** (10.87s).
- **Vite Production Build:** Transformed 54 modules cleanly with **0 errors**.
- **Real MySQL Telemetry Verified:**
  - 55 records evaluated in browser test:
    - Total: 55
    - Diseased: 20 (36.4%)
    - Healthy: 35 (63.6%)
    - Average Confidence: 94.62%
    - Top Outbreak: Tomato Late Blight (11 occurrences)
  - Search and filter controls verified via browser subagent.
  - Navigation between `/detect` and `/dashboard` verified.
