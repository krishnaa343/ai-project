# System Limitations & Boundary Constraints
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Document Version:** 1.0.0  
**Phase:** 16 — Final Documentation  
**Status:** Documented & Audited (Unmodified)  

---

## 1. Overview & Transparency Statement

Scientific rigor and agricultural engineering demand complete transparency regarding model and architectural limitations. While the framework achieves **96.40%** classification accuracy on the formal PlantVillage test benchmark, real-world deployment reveals technical boundaries and domain shifts. 

These limitations were verified during the Phase 14 system validation and are intentionally documented without masking or premature suppression. They form the foundational requirements for the subsequent dedicated improvement phase.

---

## 2. Core Machine Learning Limitations

### 2.1 Closed-Set Classification Assumption
- **Description:** The deep learning model operates under a closed-world assumption where the output space is strictly constrained to the 13 supported classes ($k \in \{0, \dots, 12\}$).
- **Behavior:** The Softmax normalization function forces the sum of class probabilities to equal 1.0 ($\sum_{i=0}^{12} P_i = 1.0$). Consequently, when presented with an out-of-distribution (OOD) image—such as a non-leaf object, an insect (e.g., ant), a human hand, or an unsupported plant species (e.g., Grape, Mango)—the network will assign 100% of the probability mass across the 13 known classes, often producing a high-confidence, erroneous classification.
- **Example Tested in Phase 14:** An image of an ant against a neutral background was classified as `Apple___Black_rot` with 84.7% confidence because dark pixels matched the convolutional feature filters of necrotic fungal lesions.
- **Mitigation Status:** Open-set / OOD rejection, Mahalanobis distance scoring, and maximum softmax threshold gating are scheduled for the dedicated post-deployment improvement phase.

### 2.2 Interspecies & Symptomatic Confusion (Tomato Early Blight vs Late Blight)
- **Description:** Early Blight (*Alternaria solani*) and Late Blight (*Phytophthora infestans*) produce visually convergent necrotic lesions during early-to-intermediate infection stages on tomato leaves.
- **Evaluation Evidence:** Confusion matrix analysis revealed that 15 out of 190 actual Tomato Late Blight test samples were misclassified as Tomato Early Blight, and 4 Early Blight samples were misclassified as Late Blight. Additional cross-crop confusion was observed where 6 Tomato Late Blight samples were classified as Potato Late Blight due to shared oomycete pathogen pathology.
- **Root Cause:** Both diseases cause dark brown, water-soaked concentric lesions that exhibit identical spectral and textural features at low resolutions.

### 2.3 Class Imbalance & Potato Healthy Precision
- **Description:** The PlantVillage dataset possesses significant class distribution skew. In our stratified test split, `Potato___healthy` contained only **15** test images, compared to 213 images for `Tomato___Bacterial_spot` and 190 images for `Tomato___Late_blight`.
- **Impact:** While `Potato___healthy` achieved 100% recall (all 15 healthy potato leaves were correctly found), false positive assignments from diseased classes lowered its precision to **68.18%** ($F_1 = 0.8108$).

### 2.4 Domain Distribution Shift (Lab vs Field Conditions)
- **Description:** The PlantVillage training set consists predominantly of individual leaves detached from plants and photographed against uniform gray or white laboratory backgrounds under controlled artificial illumination.
- **Field Impact:** When testing on raw agricultural field images captured directly from mobile phones (complex backgrounds with soil, shadows, sunlight glare, overlapping leaves, and varying focal depths), the model's effective confidence and accuracy experience observable degradation.

---

## 3. Explainability (Grad-CAM) Limitations

### 3.1 Neural Attribution vs Pathological Ground Truth
- **Description:** Grad-CAM computes gradients at the terminal convolutional feature map (`out_relu`) to highlight spatial regions that strongly activated the predicted class output.
- **Limitation:** A highlighted area indicates **correlative neural activation**, NOT an anatomically or biologically certified lesion. For instance, if background leaf venation or specular glare contributes to a high activation score, Grad-CAM will highlight those pixels. It must be interpreted as a diagnostic aid, never as absolute laboratory confirmation.

### 3.2 Coarse Spatial Resolution
- **Description:** The spatial resolution of the `out_relu` layer in MobileNetV2 for an input of $(224, 224)$ is $(7, 7)$.
- **Limitation:** The resulting $7 \times 7$ activation grid must be upsampled by a factor of 32 using bilinear interpolation to match the original image size. As a result, the heatmap provides coarse regional guidance rather than fine-grained pixel-level lesion boundary segmentation.

---

## 4. Frontend & 3D Architectural Limitations

### 4.1 3D Visualization is Procedural, Not Reconstructive
- **Description:** The 3D leaf viewer is built with Three.js to provide an interactive, educational representation of plant anatomy and typical lesion manifestation.
- **Limitation:** It is **NOT** a 3D reconstruction of the uploaded photo. It does not use photogrammetry, depth estimation, or NeRF techniques. It does not map the specific spots from the photo onto the 3D surface.

### 4.2 Three.js Client Bundle Size
- **Description:** Incorporating the Three.js library and OrbitControls increases the minified frontend JavaScript bundle size to **901.37 kB** (243.51 kB gzipped).
- **Impact:** While modern broadband connections load this bundle in under 300 ms, users on constrained 2G/3G mobile cellular networks in remote rural areas may experience an initial page load delay.

---

## 5. Deployment & Runtime Limitations

### 5.1 Ephemeral Storage for Grad-CAM Heatmaps
- **Description:** Render's free tier uses an ephemeral filesystem. Generated Grad-CAM heatmap PNGs stored under `backend/static/gradcam/` are accessible during the active session but are purged whenever the backend container restarts or redeploys.
- **Mitigation:** Historical prediction metadata (crop, disease, confidence, timestamp) remains permanently safe in Railway MySQL.

### 5.2 Cloud CPU Free-Tier Spin-Down
- **Description:** On Render's free tier, inactive services enter sleep mode after 15 minutes of inactivity. The initial incoming request incurs a "cold start" latency of approximately 30–50 seconds while the Python environment and TensorFlow model initialize. Once warm, subsequent requests execute in ~100 ms.
