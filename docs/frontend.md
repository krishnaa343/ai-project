# Phase 8 — React Frontend and User Interface Documentation

**Project:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Phase:** 8 (React Frontend and User Interface)  
**Status:** Completed  

---

## 1. Overview & Architecture

Phase 8 introduces a modern, responsive, light-themed web interface for the Crop Disease Detection Framework. Built with **React 19** and bundled with **Vite 8**, it communicates seamlessly with the Phase 7 Flask REST API running at `http://localhost:5000`.

### Technology Stack
- **Library:** React 19 (`react`, `react-dom`)
- **Routing:** React Router v7 (`react-router-dom`)
- **Build Tool:** Vite 8 (Hot Module Replacement, fast ES module bundling)
- **Styling:** Vanilla CSS design system (`index.css` + scoped component CSS), zero heavy CSS framework overhead
- **Typography:** Inter (via Google Fonts) with fallback to system UI sans-serif
- **API Client:** Native `fetch` with typed error wrapping (`services/api.js`)

---

## 2. Directory & Component Structure

```
frontend/
├── index.html                   # HTML entry point with Inter font & SEO meta tags
├── vite.config.js               # Dev server configuration with /api & /static proxy
├── package.json                 # Dependencies & build scripts
├── .env.example                 # Environment variable templates
├── .env                         # Local dev environment configuration
└── src/
    ├── index.css                # Global design system, tokens, resets, utility classes
    ├── main.jsx                 # React root renderer
    ├── App.jsx                  # Main router shell with Navbar, Footer & page routes
    ├── assets/                  # Static assets & icons
    ├── services/
    │   └── api.js               # API service: checkHealth, predictDisease, resolveGradCamUrl, ApiError
    ├── utils/
    │   └── validation.js        # File size/type validators, parseClassName, formatters
    ├── components/
    │   ├── Navbar.jsx           # Responsive navigation bar with mobile drawer
    │   ├── Navbar.css
    │   ├── Footer.jsx           # Clean footer with disclaimers and metadata
    │   ├── Footer.css
    │   ├── ImageUploader.jsx    # Drag-and-drop & file picker with validation
    │   ├── ImageUploader.css
    │   ├── ImagePreview.jsx     # Selected image card with dimensions & remove button
    │   ├── ImagePreview.css
    │   ├── LoadingState.jsx     # Accessible multi-stage progress indicator
    │   ├── LoadingState.css
    │   ├── ConfidenceBadge.jsx  # Color-coded confidence level badge
    │   ├── ConfidenceBadge.css
    │   ├── PredictionCard.jsx   # Diagnosis card, latency, & probability distribution
    │   ├── PredictionCard.css
    │   ├── GradCamViewer.jsx    # Side-by-side / tabbed viewer with scientific disclaimer
    │   ├── GradCamViewer.css
    │   ├── ErrorMessage.jsx     # Accessible alerts (error, warning, info)
    │   └── ErrorMessage.css
    └── pages/
        ├── Home.jsx             # Hero, 4-step workflow, supported crops, CTA
        ├── Home.css
        ├── Detection.jsx        # Upload stage, options toggle, and analysis trigger
        ├── Detection.css
        ├── Results.jsx          # Diagnosis display, Grad-CAM viewer, print report
        └── Results.css
```

---

## 3. Page Routes & User Flows

| Route | Page | Purpose |
|---|---|---|
| `/` | `Home` | Introduces the system, explains the 4-step workflow, showcases supported crop species, and guides users to start detection. |
| `/detect` | `Detection` | Primary diagnostic workspace. Allows selecting/dropping leaf photos, performs client-side validation, previews the image, and initiates inference with optional Grad-CAM. |
| `/results` | `Results` | Displays AI diagnostic output: disease name, confidence badge, latency, probability breakdown bars, side-by-side Grad-CAM heatmap overlay, and print report action. |

---

## 4. Design System & Aesthetics

- **Color Palette:**
  - Background: Warm natural off-white (`#f7f9f5`)
  - Primary Green: `#2d7d46` (600), `#226136` (700), `#184727` (800)
  - Card Surfaces: Pure white `#ffffff` with subtle borders (`#e2e8f0`) and soft shadows (`0 4px 6px -1px rgba(0,0,0,0.05)`)
  - Accent / Warnings: Amber `#f59e0b` and Rose `#ef4444`
- **Aesthetic Principles:** Clean, professional, agriculture-focused light mode. No dark neon backgrounds.
- **Accessibility:** High-contrast text ratios, `aria-live` loading announcements, role descriptors, and keyboard navigation.

---

## 5. API Integration & Error Handling

### Endpoints Consumed
1. `GET /api/health` — Backend health check: verified upon opening `/detect`.
2. `POST /api/predict?gradcam=true` — Multipart form upload sending leaf image under field name `file`.
3. `/static/gradcam/<filename>.png` — Serves the generated Grad-CAM heatmap overlay, proxied through Vite in development.

### Validation & Defensive Architecture
- **Client-Side Validation:** Validates file type (`image/jpeg`, `image/png`, `image/webp`) and enforces 10 MB maximum size before network transmission.
- **Error Handling:** Custom `ApiError` class parses HTTP status codes, structured JSON error bodies, and network failures. Friendly alerts guide the user without exposing raw system paths or stack traces.
- **Scientific Honesty:** Model confidence is explicitly labeled as **"Model confidence: XX.XX%"** (never labeled "Accuracy").
- **Grad-CAM Disclaimer:** Every visual explanation explicitly states: *"Highlighted regions indicate visual areas that contributed most strongly to the model's prediction. This reflects AI activation patterns and is not a definitive biological test."*

---

## 6. How to Run

### Development Mode
Ensure the Flask backend is running on `http://localhost:5000`, then run:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### Production Build
```bash
cd frontend
npm run build
npm run preview
```
The optimized bundle is generated into `frontend/dist/`.
