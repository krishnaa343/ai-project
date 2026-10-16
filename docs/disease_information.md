# Phase 10 — Disease Information and Recommendations

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Phase:** 10 (Disease Information and Recommendations)  
**Status:** Complete

---

## 1. Purpose

Phase 10 adds a structured, locally-served disease-information layer to the existing AI prediction pipeline.

When the model predicts a class, the backend now looks up and attaches factual, conservative general information about that disease (or healthy state) to the API response. The frontend displays this information below the prediction and Grad-CAM panels.

**The prediction algorithm, model, preprocessing, and Grad-CAM are unchanged.**

---

## 2. Supported Classes (13 total)

| Index | Class Name | Crop | Type |
|---|---|---|---|
| 0 | `Apple___Apple_scab` | Apple | disease |
| 1 | `Apple___Black_rot` | Apple | disease |
| 2 | `Apple___healthy` | Apple | healthy |
| 3 | `Corn_(maize)___Common_rust_` | Corn (Maize) | disease |
| 4 | `Corn_(maize)___Northern_Leaf_Blight` | Corn (Maize) | disease |
| 5 | `Corn_(maize)___healthy` | Corn (Maize) | healthy |
| 6 | `Potato___Early_blight` | Potato | disease |
| 7 | `Potato___Late_blight` | Potato | disease |
| 8 | `Potato___healthy` | Potato | healthy |
| 9 | `Tomato___Bacterial_spot` | Tomato | disease |
| 10 | `Tomato___Early_blight` | Tomato | disease |
| 11 | `Tomato___Late_blight` | Tomato | disease |
| 12 | `Tomato___healthy` | Tomato | healthy |

**Healthy classes (4):** Apple___healthy, Corn_(maize)___healthy, Potato___healthy, Tomato___healthy  
**Disease classes (9):** all others

---

## 3. Information Schema

File: `backend/data/disease_information.json`

```json
{
  "_metadata": { "version": "1.0", "supported_classes": 13, ... },
  "entries": {
    "<class_name>": {
      "class_name":    "Tomato___Early_blight",
      "crop":          "Tomato",
      "display_name":  "Tomato Early Blight",
      "type":          "disease",
      "is_healthy":    false,
      "description":   "...",
      "symptoms":      ["...", "..."],
      "management":    ["...", "..."],
      "prevention":    ["...", "..."],
      "severity_note": "...",
      "disclaimer":    "..."
    }
  }
}
```

**Healthy classes** have:
- `type: "healthy"`, `is_healthy: true`
- `symptoms: []` (empty — no disease symptoms to show)
- `severity_note: null`
- Description worded as model assessment, not confirmed diagnosis

---

## 4. Backend Service

File: `backend/app/services/disease_info_service.py`

- Loads `disease_information.json` **once** at first call (module-level cache)
- Never reloads per request
- `get_disease_info(class_name)` → returns dict or `None` (never raises)
- `list_supported_classes()` → sorted list of all keys
- `is_healthy_class(class_name)` → bool with string-heuristic fallback

---

## 5. API Response Changes

`POST /api/predict` now includes a `disease_info` field:

```json
{
  "status":            "success",
  "predicted_class":   "Tomato___Late_blight",
  "class_index":       11,
  "confidence":        0.9986,
  "confidence_pct":    99.86,
  "all_probabilities": { ... },
  "inference_ms":      null,
  "gradcam":           { "image_url": "/static/gradcam/...", "target_layer": "out_relu" },
  "disease_info": {
    "class_name":    "Tomato___Late_blight",
    "crop":          "Tomato",
    "display_name":  "Tomato Late Blight",
    "type":          "disease",
    "is_healthy":    false,
    "description":   "...",
    "symptoms":      ["...", "..."],
    "management":    ["...", "..."],
    "prevention":    ["...", "..."],
    "severity_note": "...",
    "disclaimer":    "..."
  }
}
```

If no entry exists: `"disease_info": null`  
If lookup raises: `"disease_info": null` (prediction still succeeds)

**All Phase 9 fields are unchanged.**

---

## 6. Frontend Changes

### New component: `DiseaseInfoPanel.jsx` + `DiseaseInfoPanel.css`

Displays disease information below prediction and Grad-CAM panels.

**States handled:**
- `disease_info` = valid object → full information with collapsible sections
- `disease_info` = null → "Information Unavailable" message, prediction still shown
- `disease_info` = undefined (old response format) → treated as null, graceful fallback

**Sections displayed:**
1. Header (display_name, crop, healthy/disease tag, icon)
2. Severity note callout (diseases only, when present)
3. Description / Model Assessment
4. Common Visible Symptoms (diseases only, non-empty)
5. General Management (collapsible, expanded by default)
6. Prevention (collapsible, collapsed by default)
7. Disclaimer (always shown)

### Updated: `Results.jsx`
- Imports `DiseaseInfoPanel`
- Renders below the prediction grid as a full-width row
- `disease_info` extracted safely: `prediction.disease_info !== undefined ? prediction.disease_info : null`

---

## 7. Healthy Class Handling

For healthy classes:
- `is_healthy: true`, `type: "healthy"`
- Tag shows **"Healthy Leaf"** (green)
- Symptoms section is hidden (empty array)
- Severity note is hidden (null)
- Description uses assessment language: "The model did not identify disease patterns..."
- Management shows general maintenance guidance only
- Disclaimer: "A model prediction of healthy does not replace a qualified field inspection."

---

## 8. Missing Information Handling

- Prediction API: `disease_info: null` returned — prediction succeeds
- Frontend: "Disease information is currently unavailable for this predicted class. The prediction result above is still valid."
- Disclaimer shown even in unavailable state

---

## 9. Disclaimer Wording

All disease entries include a disclaimer field. The global disclaimer shown in the panel:
> "This information is general guidance only. Always consult a qualified agricultural expert for field-level diagnosis and treatment decisions."

Confidence is labeled **"Model confidence"** — never "diagnosis certainty" or "confirmed disease".

---

## 10. Files Created / Modified in Phase 10

| File | Action |
|---|---|
| `backend/data/disease_information.json` | **Created** — 13-class structured info dataset |
| `backend/app/services/disease_info_service.py` | **Created** — singleton info lookup service |
| `backend/app/routes/prediction.py` | **Modified** — attach `disease_info` to response |
| `frontend/src/components/DiseaseInfoPanel.jsx` | **Created** — info display component |
| `frontend/src/components/DiseaseInfoPanel.css` | **Created** — component styles |
| `frontend/src/pages/Results.jsx` | **Modified** — render DiseaseInfoPanel |
| `frontend/src/pages/Results.css` | **Modified** — add `.results-info-row` |
| `docs/disease_information.md` | **Created** — this file |

**Not modified:** model files, preprocessing, Grad-CAM, class indices, training/validation/test datasets.

---

## 11. Known Limitations

1. Disease information is general guidance only — not region-specific or variety-specific
2. Information does not account for disease severity grading
3. No dynamic information — all content is static JSON
4. No multilingual support
5. Corn common rust class name has a trailing underscore (`Corn_(maize)___Common_rust_`) — this is the actual trained class name and is matched exactly
