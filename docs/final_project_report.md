# Academic Project Report
# AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Project Title:** AI-Based Intelligent Crop Disease Detection Framework for Agriculture  
**Academic Phase:** Phase 16 — Final Documentation & Viva Preparation  
**System Status:** Complete, Validated, and Production-Ready  

---

## 1. Title
**AI-Based Intelligent Crop Disease Detection Framework for Agriculture**  
*An End-to-End Decision Support System Incorporating Deep Transfer Learning, Explainable AI (Grad-CAM), Relational Surveillance Persistence, and Interactive 3D Conceptual Anatomical Modeling.*

---

## 2. Abstract
Agricultural productivity faces severe annual losses exceeding 20–40% globally due to plant pathogens, fungal infections, and bacterial blights. Traditional manual foliar inspection requires expert phytopathologists, whose scarcity in rural farming communities causes delayed diagnosis and inappropriate chemical treatment. This project presents an end-to-end, full-stack intelligent crop disease detection framework designed to bridge this accessibility gap. 

The framework employs a fine-tuned MobileNetV2 deep convolutional neural network trained on a standardized 13-class subset of the PlantVillage benchmark dataset, covering four commercially critical crops: Apple, Corn (Maize), Potato, and Tomato. The system achieves a formal test classification accuracy of **96.40%** across 1,499 held-out test images ($F_1\text{-score} = 0.9547$, Macro Precision = 0.9455, Macro Recall = 0.9692). To overcome the "black-box" dilemma of deep neural networks, Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to generate high-resolution visual attention heatmaps that visually corroborate which leaf regions influenced model predictions. 

The inference engine is wrapped in a production-hardened Python Flask REST API connected to a MySQL relational database for persistent surveillance logging and temporal tracking. A modern React 19 single-page application provides intuitive image submission, confidence estimation gauges, curative and preventative agricultural management guidelines, an analytics dashboard displaying regional crop health distributions, and an interactive WebGL-based 3D conceptual leaf visualizer built with Three.js. Comprehensive system testing encompassing 32 automated test suites validated functional integrity, database persistence, and security isolation. Known limitations—including closed-set classification assumptions, inter-species symptom confusion, and distribution shifts under uncontrolled field lighting—are documented transparently to guide future open-set research.

---

## 3. Introduction
Agriculture is the foundation of global food security and economic stability. According to the Food and Agriculture Organization (FAO), transboundary plant pests and diseases destroy between 20% and 40% of global agricultural production annually. In smallholder farming systems across developing economies, early detection is critical. Farmers frequently rely on visual symptom inspection, which is prone to human error, cognitive bias, and misdiagnosis between pathogen-induced necrosis and environmental physiological stresses.

Advances in Computer Vision (CV) and Deep Learning (DL) have demonstrated transformative potential in automated image classification. By leveraging deep Convolutional Neural Networks (CNNs) pre-trained on massive image corpora, complex visual hierarchical representations can be adapted to agricultural pathology via transfer learning. However, practical adoption by farmers and agronomists requires more than high benchmark accuracy: it demands **visual interpretability**, **actionable treatment recommendations**, **longitudinal disease tracking**, and **responsive, accessible user interfaces**. This project develops a unified framework fulfilling these requirements.

---

## 4. Problem Statement
Manual identification of crop diseases suffers from three fundamental bottlenecks:
1. **Scarcity of Pathological Expertise:** Expert agronomists cannot physically inspect millions of scattered smallholder plots in real time.
2. **The Black-Box Mistrust Problem:** High-accuracy deep learning models output raw class labels and probabilities without visual justification. Without understanding *why* an AI model made a specific prediction, agricultural stakeholders are hesitant to invest in costly chemical or cultural remedies.
3. **Absence of Unified Digital Workflows:** Existing academic implementations often conclude at the Jupyter Notebook stage, lacking production-grade REST APIs, secure relational logging, surveillance analytics, and multi-platform client accessibility.

---

## 5. Objectives
The explicit objectives successfully realized in this project are:
1. **Automated Image Classification:** Develop and evaluate a deep CNN capable of classifying foliar diseases across 13 distinct crop-health classes with high discriminative accuracy.
2. **Visual Explainability (XAI):** Integrate Grad-CAM into the inference pipeline to extract spatial attention heatmaps directly from the terminal convolutional layers.
3. **Actionable Knowledge Delivery:** Provide an integrated repository of verified symptoms, organic remedies, chemical treatments, and preventative practices.
4. **Relational Data Persistence:** Implement a scalable MySQL logging system to record diagnosis transactions, crop types, confidence scores, and timestamps.
5. **Surveillance Dashboard:** Engineer an interactive analytics dashboard computing real-time KPIs, disease distribution charts, and filterable prediction logs.
6. **Interactive 3D Visualization:** Implement a client-side Three.js 3D conceptual leaf model to provide an engaging educational exploration of leaf anatomy and disease manifestation.
7. **Cloud Production Deployment:** Architect a decoupled deployment using Vercel (Frontend), Render (Flask API), and Railway (MySQL) with strict security isolation.

---

## 6. Motivation
The primary motivation of this research is democratizing access to phytopathological diagnostics. A smallholder farmer armed with a basic smartphone camera should be able to upload a leaf photograph and receive an instant, interpretable diagnosis within seconds, coupled with immediate treatment protocols. By preserving prediction logs in a central relational database, regional agricultural extension services gain a real-time surveillance dashboard to track emerging disease hotspots, prevent catastrophic contagion, and optimize resource allocation.

---

## 7. Literature & Reference Context
Automated leaf disease detection has evolved through three distinct algorithmic eras:
- **Classical Computer Vision (Pre-2012):** Early methodologies utilized handcrafted morphological, textural, and color features (e.g., Gray-Level Co-occurrence Matrices [GLCM], Scale-Invariant Feature Transforms [SIFT], color histograms) paired with Support Vector Machines (SVMs) or Random Forests. These methods were fragile, highly sensitive to illumination variations, and required manual segmentation.
- **Deep Convolutional Networks (2012–Present):** Following the AlexNet breakthrough, deep CNNs automated feature representation learning. Mohanty et al. (2016) demonstrated over 99% accuracy on the PlantVillage dataset using deep architectures. Subsequent works demonstrated that transfer learning with lightweight architectures (e.g., MobileNet, EfficientNet) offers high classification capacity with low computational overhead.
- **Explainable AI in Agriculture (2018–Present):** Selvaraju et al. (2017) introduced Grad-CAM, enabling visual attribution of CNN decisions without architectural modification. Integrating Grad-CAM into agricultural vision systems provides agronomists with interpretable evidence verifying that the model focuses on actual pathological lesions rather than irrelevant background artifacts.

---

## 8. Proposed System
The proposed system, **CropScan AI**, integrates an end-to-end processing pipeline:
1. **Client Interface:** Responsive web application allowing image selection via drag-and-drop or device camera.
2. **Security & Intake Guard:** Multi-tiered file validation verifying MIME types, magic byte headers, and file size constraints ($< 10\text{ MB}$).
3. **Inference Pipeline:** Thread-safe singleton MobileNetV2 model executing feed-forward inference on normalized RGB arrays ($224 \times 224 \times 3$).
4. **Attribution Engine:** Dynamic backpropagation extracting gradients from layer `out_relu` to synthesize alpha-blended Grad-CAM heatmaps.
5. **Guidance Engine:** Localized JSON service attaching structured management strategies.
6. **Relational Persistence:** Fault-tolerant MySQL logging service recording diagnosis history.
7. **Analytics Dashboard:** SQL aggregation pipeline rendering surveillance charts, health ratios, and search mechanisms.
8. **Conceptual 3D Mesh:** Three.js parametric geometry illustrating plant morphology.

---

## 9. System Architecture
The framework follows a strict three-tier decoupled model:
- **Tier 1 (Presentation):** React 19 SPA running on Vercel Edge.
- **Tier 2 (Application & ML):** Flask 3.0 REST API running on Render under Gunicorn WSGI.
- **Tier 3 (Data Persistence):** MySQL 8.x running on Railway.

Communication between Tier 1 and Tier 2 occurs via HTTPS REST requests restricted by explicit CORS origins. Tier 1 has zero visibility or access to Tier 3 credentials. Database transactions are managed entirely by Tier 2 using parameterized SQL statements over port 3306.

---

## 10. Dataset Specification
The model was trained on a curated subset of the public **PlantVillage** benchmark repository.
- **Total Selected Dataset:** 7,495 standardized leaf images across 13 classes.
- **Selected Crops:** Apple, Corn (Maize), Potato, Tomato.
- **Stratified Partitioning:**
  - **Training Split (70%):** 5,246 images (used for parameter optimization).
  - **Validation Split (10%):** 750 images (used for checkpointing and early stopping).
  - **Testing Split (20%):** 1,499 images (held out strictly for final unbiased evaluation).
- **Leakage Prevention:** Manifests were partitioned deterministically using a fixed random seed (`seed = 42`) with stratified class distribution. No image was shared across splits.

---

## 11. Data Preprocessing
All images undergo an identical deterministic preprocessing pipeline:
1. **Color Decoding:** Image files decoded into standard 8-bit unsigned integer RGB matrices.
2. **Spatial Resizing:** Resampled to $(224, 224, 3)$ using bilinear interpolation.
3. **Intensity Scaling:** Pixel intensities scaled to floating-point values in $[0.0, 1.0]$ via $x_{norm} = \frac{x}{255.0}$.
4. **Training Augmentation:** Applied exclusively during training to enhance generalization without distorting lesion structure:
   - Random horizontal flip ($p = 0.5$)
   - Random brightness variation ($\pm 10\%$)
   - Random contrast variation ($\pm 10\%$)

---

## 12. Model Development
The neural network architecture utilizes **MobileNetV2** pre-trained on ImageNet-1k as the foundational feature extractor. The network leverages depthwise separable convolutions consisting of a $3 \times 3$ depthwise convolution followed by a $1 \times 1$ pointwise convolution, substantially reducing parameters and computational operations while preserving spatial representation fidelity.

The custom classification head appended to the terminal feature map (`out_relu`, spatial dimension $7 \times 7 \times 1280$) consists of:
- `GlobalAveragePooling2D`: Collapses spatial dimensions to a 1,280-dimensional feature vector.
- `Dropout(rate = 0.3)`: Injects stochastic regularization to mitigate overfitting.
- `Dense(13, activation='softmax')`: Produces the posterior probability distribution across all 13 classes.

---

## 13. Model Training
Training was executed in two sequential stages:
- **Stage 1 (Feature Extraction):** The MobileNetV2 backbone was frozen. The classification head was trained for 5 epochs using the Adam optimizer ($\eta = 10^{-3}$, batch size 32). Loss converged rapidly with validation accuracy reaching 96.40%.
- **Stage 2 (Fine-Tuning):** Layers 100 through 154 were unfrozen while layers 0–99 remained frozen. Training continued for 4 epochs using Adam with a reduced learning rate ($\eta = 10^{-5}$) under `EarlyStopping` monitoring `val_loss`.
- **Loss Function:** `sparse_categorical_crossentropy` with inverse class weighting applied to balance minority classes.
- **Training Time:** 7.4 minutes on modern workstation GPU.

---

## 14. Model Evaluation & Benchmark Results
The model was evaluated on the independent 1,499-image test dataset.

### Summary Metrics:
- **Test Accuracy:** **96.40%** (1,445 correct, 54 errors)
- **Macro Precision:** 0.9455
- **Macro Recall:** 0.9692
- **Macro F1-Score:** 0.9547
- **Weighted Precision:** 0.9657
- **Weighted Recall:** 0.9640
- **Weighted F1-Score:** 0.9642
- **ROC-AUC (Macro):** 0.9991

### Class-Wise Performance Table:
| Class Index | Class Identifier | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| 0 | Apple___Apple_scab | 0.9692 | 1.0000 | 0.9844 | 63 |
| 1 | Apple___Black_rot | 0.9841 | 1.0000 | 0.9920 | 62 |
| 2 | Apple___healthy | 0.9938 | 0.9817 | 0.9877 | 164 |
| 3 | Corn_(maize)___Common_rust_ | 1.0000 | 1.0000 | 1.0000 | 119 |
| 4 | Corn_(maize)___Northern_Leaf_Blight | 1.0000 | 0.9796 | 0.9897 | 98 |
| 5 | Corn_(maize)___healthy | 0.9831 | 1.0000 | 0.9915 | 116 |
| 6 | Potato___Early_blight | 1.0000 | 1.0000 | 1.0000 | 100 |
| 7 | Potato___Late_blight | 0.9381 | 0.9100 | 0.9239 | 100 |
| 8 | Potato___healthy | 0.6818 | 1.0000 | 0.8108 | 15 |
| 9 | Tomato___Bacterial_spot | 0.9591 | 0.9906 | 0.9746 | 213 |
| 10 | Tomato___Early_blight | 0.8286 | 0.8700 | 0.8488 | 100 |
| 11 | Tomato___Late_blight | 0.9595 | 0.8737 | 0.9146 | 190 |
| 12 | Tomato___healthy | 0.9937 | 0.9937 | 0.9937 | 159 |

---

## 15. Explainable AI: Grad-CAM Implementation
To generate interpretability heatmaps without retraining, Grad-CAM utilizes the gradient of the predicted class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer `out_relu`:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

The weights $\alpha_k^c$ capture the importance of feature map $k$ for class $c$. A weighted linear combination followed by a rectified linear unit (ReLU) produces the 2D coarse heatmap:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

The resulting activation grid is normalized to $[0, 1]$, upsampled to $224 \times 224$, mapped through OpenCV's Jet colormap, and alpha-blended ($60\%$ original, $40\%$ heatmap) to highlight the critical foliar lesion boundaries.

---

## 16. Backend REST API Development
The backend is structured using Flask application factories and modular blueprints:
- `routes/health.py`: Readiness probe returning model loading state and class count.
- `routes/prediction.py`: Multipart ingestion, validation, inference execution, Grad-CAM generation, and MySQL logging.
- `routes/history.py`: Reverse-chronological paginated history retrieval.
- `services/model_service.py`: Thread-safe double-checked singleton loader.
- `services/prediction_history_service.py`: Parameterized database persistence manager.
- `utils/file_validation.py`: Binary header inspector rejecting non-image payloads.

---

## 17. Frontend Web Application Development
Built with React 19 and Vite 8, the client interface delivers high aesthetic and usability standards:
- **Responsive Navigation:** Fluid top navigation bar with collapsible mobile slide-out drawer.
- **Drag-and-Drop Uploader:** Client-side image preview with instant MIME and file-size validation.
- **Results View:** Visual hierarchy featuring large confidence badges, expandable disease information tabs, and side-by-side original/Grad-CAM visual inspection.
- **Analytics Dashboard:** Real-time KPI summaries, health status ratios, disease frequency bar charts, and a search-enabled historical inspection table.

---

## 18. Disease Information & Guidance Repository
A comprehensive knowledge base (`backend/data/disease_information.json`) maps every supported class to curated agricultural information:
- Standard user-friendly display name and binomial scientific name.
- Primary visual symptoms across leaf, stem, and fruit tissue.
- Pathological causative factors and conducive weather triggers.
- Two-tiered management strategies: organic cultural practices and authorized chemical interventions.
- Long-term prevention protocols including crop rotation intervals and resistant seed cultivars.

---

## 19. MySQL Database Integration
The persistence tier provides permanent tracking of all diagnostic transactions:
- **Schema:** Managed relational table `prediction_history` with indexed fields for `created_at`, `crop`, and `predicted_class`.
- **Fault-Tolerant Persistence:** If the database engine experiences an outage, prediction inference and disease guidance still return successfully to the user with a non-blocking persistence warning.
- **Security:** Complete parameterization (`%s`) prevents SQL injection attacks; frontend has zero direct connection to database credentials.

---

## 20. Dashboard & Surveillance Analytics
The Analytics Dashboard transforms raw database records into actionable agricultural intelligence:
- **Summary Metrics:** Total Diagnoses, Diseased Count, Healthy Count, and Average Diagnostic Confidence.
- **Health Ratio Indicator:** Dynamic visual progress gauge reflecting the healthy-to-diseased ratio across all recorded crops.
- **Crop Surveillance Distribution:** Aggregated breakdown of diagnoses grouped by crop type (Apple, Corn, Potato, Tomato).
- **Searchable Log Table:** Paginated tabular view with instant text search by crop or disease name, and health-state filter chips.

---

## 21. 3D Conceptual Leaf Visualization
An interactive procedural 3D leaf model developed with Three.js provides an engaging anatomical visualizer:
- **Procedural Geometry:** Parametric curves generate species-specific leaf silhouettes (ovate for Apple, ribbon blade for Corn, compound pinnate for Potato, lobed for Tomato).
- **Dynamic Shading:** Material parameters adjust dynamically—healthy classifications display smooth, bright green chloroplast textures, while diseased classifications superimpose irregular chlorotic and necrotic spot shaders.
- **Interactivity:** Full OrbitControls supporting 3D rotation, zooming, panning, auto-spin toggle, and camera reset.
- **Resilience:** Automatic WebGL detection displays a clean 2D fallback message if hardware acceleration is unavailable.

---

## 22. System Integration
All subsystems are linked through unified API contracts:
- Development environment utilizes Vite dev server proxying `/api/*` and `/static/*` to Flask at `http://localhost:5000`.
- Production environment routes API calls cross-origin via HTTPS with explicit CORS whitelisting between Vercel and Render.
- Grad-CAM image paths dynamically resolve to full URLs in production to ensure cross-domain asset delivery.

---

## 23. Testing & System Validation
Validation conducted during Phase 14 established 100% test success:
- **Backend Test Suite:** 32 automated Pytest unit and integration tests passing in 9.19s.
- **File Validation Tests:** Successfully intercepted and rejected oversized files, empty payloads, spoofed extensions, and executable files.
- **Security Tests:** Verified zero credential leakage in frontend distribution assets; verified proper parameterization across all SQL queries.
- **Parity Test:** Successfully executed end-to-end data parity verification confirming that frontend input, backend inference, MySQL database row, and dashboard KPI matched exactly.

---

## 24. Production Deployment Architecture
The system is deployed across industry-standard cloud providers:
- **Frontend:** Hosted on **Vercel** as an edge-optimized SPA with route rewrites defined in `vercel.json`.
- **Backend:** Hosted on **Render** as a Python 3.11 Web Service running Gunicorn WSGI (`--workers 1 --timeout 120`).
- **Database:** Hosted on **Railway** as a managed MySQL 8.x instance.
- **Model Storage:** Frozen `best_model.keras` (24.75 MB) tracked directly in version control for automated container loading.

---

## 25. Known Limitations
1. **Closed-Set Assumption:** Non-leaf images or unsupported plant species are forced into one of the 13 classes by standard Softmax normalization.
2. **Pathological Confusion:** Visual similarity between Early Blight and Late Blight on Tomato foliage causes reciprocal misclassifications.
3. **Class Skew:** Restricted sample support for `Potato___healthy` (15 test samples) resulted in reduced precision (68.18%).
4. **Lighting Shift:** Model was trained on laboratory PlantVillage images; complex field backgrounds with shadows and direct sunlight cause observable confidence variance.
5. **Procedural 3D Model:** The 3D leaf is a conceptual visualization and not an image-to-3D photogrammetric reconstruction.

---

## 26. Future Improvements
1. **Open-Set / Out-of-Distribution Rejection:** Integrate Mahalanobis distance scoring or energy-based OOD detection to reject non-leaf or unsupported images.
2. **Model Expansion:** Expand taxonomy to cover additional crops (e.g., Rice, Wheat, Cassava, Cotton) and abiotic stress factors (nutrient deficiencies).
3. **In-the-Wild Adaptation:** Fine-tune on unconstrained in-field datasets (e.g., PlantDoc) to improve robustness under complex sunlight and cluttered backgrounds.
4. **Severity Estimation:** Implement semantic segmentation (U-Net / DeepLabv3+) to calculate exact percentage lesion coverage.
5. **Mobile Offline Edge Inference:** Convert model to TensorFlow Lite (TFLite) / ONNX for offline on-device inference without internet connectivity.

---

## 27. Conclusion
The **AI-Based Intelligent Crop Disease Detection Framework for Agriculture** successfully demonstrates the integration of deep convolutional neural networks, explainable artificial intelligence, relational surveillance tracking, and responsive web technologies. Achieving 96.40% classification accuracy on the 13-class PlantVillage benchmark, providing visual transparency through Grad-CAM, delivering curated disease management guidelines, and offering an engaging 3D visualization, the system stands as a robust, production-ready decision-support tool. It establishes a strong technological blueprint for the future of digital phytopathology and smart farming.

---

## 28. References
1. Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to show the possibility of disease detection.* arXiv preprint arXiv:1511.08060. (PlantVillage Dataset).
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520.
3. Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618-626.
4. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
5. Singh, A. K., Ganapathysubramanian, B., Sarkar, S., & Singh, A. (2018). *Deep learning for plant stress phenotyping: trends and future perspectives.* Trends in Plant Science, 23(10), 883-898.
6. Food and Agriculture Organization of the United Nations (FAO). (2021). *The State of Food and Agriculture: Making agrifood systems more resilient to shocks and stresses.* Rome.
