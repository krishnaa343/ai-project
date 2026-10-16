# Final Project Presentation Outline (15 Slides)
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Document Version:** 1.0.0  
**Phase:** 16 — Final Documentation  
**Audience:** Academic Evaluation Committee / Technical Evaluators  

---

### Slide 1: Title & Overview
- **Title:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture
- **Subtitle:** An End-to-End Decision-Support Platform with Transfer Learning, Grad-CAM Explainability, Relational Analytics, and Interactive 3D Visualization
- **Key Details:**
  - Candidate Name / Registration Number
  - Department of Computer Science & Engineering
  - Academic Year 2026
- **Suggested Visual:** Hero banner featuring the CropScan AI application logo and screenshots of the Web UI, Grad-CAM heatmap, and 3D leaf model.

---

### Slide 2: Problem Statement & Motivation
- **Title:** The Agricultural Challenge
- **Key Points:**
  - 20% to 40% of global agricultural yield is lost annually to foliar pathogens and pests (FAO).
  - Severe shortage of expert phytopathologists in rural farming communities causes delayed diagnosis.
  - Inaccurate visual diagnosis leads to excessive or inappropriate chemical fungicide application.
  - Conventional AI models act as untrustworthy "black boxes" lacking visual justification.
- **Suggested Visual:** Infographic illustrating crop yield loss statistics alongside contrasting photos of diseased crops and healthy crops.

---

### Slide 3: Project Objectives
- **Title:** Project Objectives & Scope
- **Key Points:**
  - Develop a lightweight, high-accuracy deep CNN for 13 foliar disease classes across 4 staple crops.
  - Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to provide transparent visual explanations.
  - Provide actionable, curated agricultural management strategies (cultural and chemical).
  - Build a secure, relational MySQL persistence logging system.
  - Engineer a real-time surveillance dashboard with epidemiological charts and metrics.
  - Create an interactive 3D conceptual leaf visualizer using Three.js.
- **Suggested Visual:** High-level schematic connecting Input Leaf $\rightarrow$ CNN $\rightarrow$ Grad-CAM $\rightarrow$ Web App $\rightarrow$ Dashboard.

---

### Slide 4: Literature & Evolutionary Context
- **Title:** Literature & Technological Evolution
- **Key Points:**
  - *Phase 1: Handcrafted CV (Pre-2012)* — SIFT, GLCM, color histograms; fragile under lighting changes.
  - *Phase 2: Deep Convolutional Learning (2012–Present)* — Automated hierarchical feature extraction; high benchmark accuracy (Mohanty et al., PlantVillage).
  - *Phase 3: Explainable AI & Deployment (Current Focus)* — Transitioning from isolated notebooks to explainable, production-ready, cloud-hosted decision support.
- **Suggested Visual:** Timeline diagram illustrating the progression from Handcrafted Features to Deep CNNs to Explainable AI (XAI).

---

### Slide 5: Proposed System Architecture
- **Title:** Decoupled Three-Tier System Architecture
- **Key Points:**
  - **Tier 1 (Presentation):** React 19 + Vite 8 SPA hosted on Vercel Edge.
  - **Tier 2 (Application & ML):** Python 3.11 + Flask 3.0 REST API with Gunicorn on Render.
  - **Tier 3 (Data Persistence):** MySQL 8.x managed service on Railway.
  - Strict security boundary: Zero database credentials exposed to client-side frontend.
- **Suggested Visual:** Architecture block diagram showing data flow from Browser $\rightarrow$ Flask $\rightarrow$ MobileNetV2 / Grad-CAM $\rightarrow$ MySQL.

---

### Slide 6: Dataset & Preprocessing Pipeline
- **Title:** Dataset & Preprocessing Methodology
- **Key Points:**
  - PlantVillage benchmark dataset: 7,495 curated images across 13 classes.
  - 4 crops covered: Apple (3 classes), Corn (3 classes), Potato (3 classes), Tomato (4 classes).
  - Stratified partitioning: 70% Train (5,246), 10% Validation (750), 20% Test (1,499).
  - Preprocessing: Decoded to RGB, bilaterally resized to $224 \times 224$, normalized to $[0.0, 1.0]$.
  - Data Augmentation: Horizontal flips, brightness ($\pm 10\%$), contrast ($\pm 10\%$).
- **Suggested Visual:** Grid showing sample images from each of the 4 crops alongside data augmentation examples.

---

### Slide 7: Neural Network Architecture & Transfer Learning
- **Title:** Model Architecture & Two-Stage Training
- **Key Points:**
  - **Backbone:** MobileNetV2 (inverted residual blocks, depthwise separable convolutions).
  - **Classification Head:** GlobalAveragePooling2D $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(13, Softmax).
  - **Stage 1 (Feature Extraction):** Backbone frozen; trained head for 5 epochs ($\eta = 10^{-3}$).
  - **Stage 2 (Fine-Tuning):** Layers 100–154 unfrozen; trained for 4 epochs ($\eta = 10^{-5}$).
  - **Loss & Optimizer:** Sparse Categorical Cross-Entropy with inverse class weighting + Adam.
- **Suggested Visual:** Layer-by-layer diagram of MobileNetV2 highlighting frozen vs. fine-tuned layers and bottleneck blocks.

---

### Slide 8: Model Evaluation & Performance Metrics
- **Title:** Experimental Evaluation & Results
- **Key Points:**
  - **Test Accuracy:** **96.40%** (1,445 correct out of 1,499 held-out test samples).
  - **Macro Precision:** 0.9455 \| **Macro Recall:** 0.9692 \| **Macro F1-Score:** 0.9547.
  - **Macro ROC-AUC:** 0.9991.
  - **Inference Latency:** 108.56 ms mean CPU latency (9.2 FPS throughput).
  - Perfect 1.000 F1-score achieved on Corn Common Rust and Potato Early Blight.
- **Suggested Visual:** High-resolution confusion matrix heatmap and ROC curve charts from evaluation phase.

---

### Slide 9: Explainable AI: Grad-CAM
- **Title:** Explainable AI with Grad-CAM
- **Key Points:**
  - Resolves the "black-box" dilemma by computing gradients flowing into the `out_relu` convolutional layer.
  - Generates coarse 2D localization maps showing regions of positive attribution.
  - Colorized with OpenCV Jet colormap and alpha-blended ($60\%$ image, $40\%$ heatmap).
  - Builds trust by demonstrating model focus on actual foliar lesions rather than background noise.
- **Suggested Visual:** Side-by-side comparison showing raw input leaf, intermediate gradient heatmap, and final blended Grad-CAM overlay.

---

### Slide 10: Responsive Web Application UI
- **Title:** Modern Client Interface (React 19 + Vite 8)
- **Key Points:**
  - Accessible, glassmorphic dark-mode design system with responsive layouts across mobile, tablet, and desktop.
  - Intuitive drag-and-drop upload zone with client-side image preview and validation.
  - Real-time confidence gauge, disease category badge, and severity indicators.
  - Curated Disease Information Panel with symptoms, causes, cultural remedies, and chemical treatments.
- **Suggested Visual:** Screenshots of the Home landing page, Upload interface, and Diagnosis Results page.

---

### Slide 11: Relational Persistence & Analytics Dashboard
- **Title:** MySQL Persistence & Analytics Dashboard
- **Key Points:**
  - All diagnoses are logged to MySQL table `prediction_history` with full metadata.
  - Fault-tolerant persistence: Database failure never interrupts diagnosis delivery to the user.
  - Real-time Dashboard KPIs: Total Diagnoses, Diseased/Healthy Breakdown, Health Ratio Gauge.
  - Disease prevalence bar charts and crop surveillance distribution.
  - Searchable, filterable, and paginated historical diagnosis log table.
- **Suggested Visual:** Screenshot of the full Analytics Dashboard showing KPI cards, distribution graphs, and the history table.

---

### Slide 12: Interactive 3D Conceptual Leaf Visualization
- **Title:** 3D Interactive Leaf Visualizer (Three.js)
- **Key Points:**
  - WebGL-powered client-side 3D model embedded directly in the diagnosis results view.
  - Procedural crop-aware morphology (distinct geometries for Apple, Corn, Potato, Tomato).
  - Dynamic shader materials: smooth green for healthy leaves; necrotic/chlorotic spot shaders for diseased leaves.
  - Interactive OrbitControls (360° rotation, zoom, pan, auto-spin toggle, reset).
  - *Honesty Clarification:* Conceptual educational model, not an image-to-3D photogrammetric reconstruction.
- **Suggested Visual:** Screenshots of the 3D leaf model in both healthy and diseased states with UI rotation controls.

---

### Slide 13: System Validation & Cloud Deployment
- **Title:** Full-System Testing & Cloud Deployment
- **Key Points:**
  - **Testing:** 32/32 automated Pytest test suites passing (API, DB, validation, security).
  - **Security:** Zero credentials exposed in frontend bundle; 100% parameterized SQL queries; MIME/header upload inspection.
  - **Frontend Deployment:** Vercel Edge SPA hosting with `vercel.json` rewrites.
  - **Backend Deployment:** Render Python Web Service with Gunicorn WSGI.
  - **Database Deployment:** Railway Managed MySQL 8.x instance.
- **Suggested Visual:** Terminal output showing 32 passed tests alongside cloud architecture logos (Vercel, Render, Railway).

---

### Slide 14: System Limitations & Future Research
- **Title:** Scientific Limitations & Future Roadmap
- **Key Points:**
  - *Closed-Set Limitation:* Softmax forces non-leaf or out-of-distribution images into 1 of 13 classes.
  - *Pathological Confusion:* Visual convergence between Tomato Early Blight and Late Blight.
  - *Distribution Shift:* Performance delta between clean laboratory images and complex field backgrounds.
  - *Future Scope:* Out-of-Distribution (OOD) rejection algorithms, expanding to field datasets (PlantDoc), lesion severity segmentation (U-Net), and mobile offline inference (TFLite).
- **Suggested Visual:** Example showing the OOD test case (ant image) and diagram outlining the future open-set rejection filter.

---

### Slide 15: Conclusion & Summary
- **Title:** Conclusion & Project Summary
- **Key Points:**
  - Successfully built a complete, production-grade agricultural decision-support framework.
  - Achieved **96.40%** benchmark accuracy with MobileNetV2 on 13 classes.
  - Bridged the interpretability gap using Grad-CAM visual heatmaps.
  - Integrated persistent relational surveillance, an analytics dashboard, and interactive 3D visualization.
  - Fully tested, secured, and deployed on cloud infrastructure.
  - Ready for live demonstration.
- **Suggested Visual:** Summary infographic with thank you note, repository link, and Q&A prompt.
