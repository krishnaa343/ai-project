# Live Demonstration Script (5–10 Minutes)
## AI-Based Intelligent Crop Disease Detection Framework for Agriculture

**Document Version:** 1.0.0  
**Phase:** 16 — Final Documentation  
**Target Audience:** Evaluators, Professors, and Technical Assessors  
**Total Duration:** 7–9 minutes  

---

## Pre-Demonstration Checklist
- Ensure local or production backend is running (`http://localhost:5000` or deployed Render URL).
- Ensure frontend is running (`http://localhost:5173` or deployed Vercel URL).
- Prepare sample test leaf images on desktop:
  - `test_leaf.jpg` (Tomato Late Blight)
  - `healthy_apple.jpg` or another supported class
  - `test_ant_non_leaf.jpg` (for demonstrating closed-set limitation if requested)
- Open browser in full-screen mode at `http://localhost:5173`.

---

## Minute-by-Minute Demonstration Flow

### Minute 0:00 – 1:00: Introduction & Home Page
- **Action:** Display the Home landing page (`/`).
- **What to Say:**
  > *"Good morning respected committee members. Today I am demonstrating **CropScan AI**, an AI-based intelligent crop disease detection framework for agriculture. The motivation behind this project is to address the 20% to 40% annual crop yield losses caused by foliar pathogens by providing farmers and agronomists with an accessible, interpretable diagnostic tool. 
  > 
  > The platform currently supports 13 categories across four major staple crops: Apple, Corn, Potato, and Tomato. The system does not merely classify images as a black box; it combines deep transfer learning via MobileNetV2 with visual explainability via Grad-CAM, integrated management guidance, an interactive 3D conceptual leaf visualizer, and a real-time MySQL analytics dashboard."*

---

### Minute 1:00 – 2:30: Image Upload & Real-Time Diagnosis
- **Action:** Click **"Start Diagnosis"** or navigate to the **Detect** page (`/detect`). Drag and drop `test_leaf.jpg` into the uploader zone.
- **What to Say:**
  > *"Here on the Detect page, the user can upload a leaf photograph. Notice that before sending the file to the server, the client immediately performs pre-upload validation, verifying supported image formats (JPEG, PNG, WEBP) and enforcing a 10 MB file size limit.
  > 
  > I will now submit this tomato leaf image. When I click 'Analyze Leaf', the image is transmitted to our Flask REST API. The backend verifies the magic bytes to prevent script injection, normalizes the RGB image to 224 by 224 pixels, and executes inference through our thread-safe singleton MobileNetV2 model."*

---

### Minute 2:30 – 4:00: Diagnostic Results & Explainable AI (Grad-CAM)
- **Action:** View the Results page (`/results`). Highlight the Prediction Badge, Confidence Gauge, and Grad-CAM section.
- **What to Say:**
  > *"The diagnosis is returned in under 100 milliseconds. The model classifies this leaf as **Tomato Late Blight** with **88.8% confidence**. 
  > 
  > Now, rather than forcing the user to trust an arbitrary percentage, let us look at the **Grad-CAM Explainable AI panel**. Grad-CAM calculates the gradients of the Late Blight prediction with respect to the final convolutional feature maps in our MobileNetV2 backbone. 
  > 
  > As you can see in the side-by-side view, the warm red and yellow activation zones directly overlay the dark, water-soaked necrotic lesions on the leaf tissue. This visual proof confirms that the neural network is focusing on actual biological disease pathology rather than background artifacts or lighting reflections."*

---

### Minute 4:00 – 5:00: Disease Management Guidance
- **Action:** Scroll down to the **Disease Information Panel**. Expand the Symptoms, Management, and Prevention tabs.
- **What to Say:**
  > *"A diagnosis is only useful if accompanied by actionable guidance. Here, our system automatically retrieves tailored agronomic recommendations from our disease knowledge base. 
  > 
  > We see the scientific binomial classification—Phytophthora infestans—the primary foliar symptoms, and dual-track management advice: organic cultural practices such as avoiding overhead irrigation and removing infected foliage, alongside approved chemical fungicide interventions. We also emphasize long-term preventative measures such as crop rotation."*

---

### Minute 5:00 – 6:30: Interactive 3D Conceptual Leaf Visualization
- **Action:** Scroll to the **3D Interactive Leaf Viewer**. Click and drag to rotate the 3D leaf; use mouse wheel to zoom; click **Auto-Spin**; click **Reset View**.
- **What to Say:**
  > *"Next, we provide an interactive educational tool: our **3D Conceptual Leaf Visualizer**, built natively with Three.js and WebGL. 
  > 
  > I would like to clarify an important technical distinction: this 3D model is a **procedural conceptual visualization**, not an image-to-3D photogrammetric reconstruction of the photograph. The geometry adapts to the crop type—in this case, displaying the lobed compound morphology of a Solanum lycopersicum leaf—and the surface shaders dynamically reflect the disease status, rendering characteristic chlorotic yellow halos and necrotic brown lesions. 
  > 
  > The user can rotate 360 degrees, zoom in on the leaf venation, enable ambient auto-spin, and reset the perspective. If a user accesses this on a low-end mobile browser without WebGL, our code detects this and falls back to a clean 2D anatomical card without crashing."*

---

### Minute 6:30 – 8:00: Relational Persistence & Analytics Dashboard
- **Action:** Click **Dashboard** in the top navigation bar (`/dashboard`).
- **What to Say:**
  > *"Every diagnosis performed by the system is automatically persisted to our managed MySQL database table, `prediction_history`. When we open the Analytics Dashboard, the frontend queries our API to calculate real-time surveillance metrics.
  > 
  > Here we observe our aggregate KPIs: Total Diagnoses recorded, the count of Diseased versus Healthy plants, the Plant Health Ratio gauge, and the overall Average Confidence. Below, we see the distribution of diseases and our crop surveillance breakdown.
  > 
  > In the Recent Predictions table, you can see the diagnosis we just performed recorded at the very top with its exact timestamp, crop, confidence score, and status. The table includes instant live search—for instance, typing 'Potato' or 'Tomato' immediately filters the records—along with pagination. This provides agricultural extension officers with a regional disease surveillance tool."*

---

### Minute 8:00 – 9:00: Limitations, Future Work & Conclusion
- **Action:** Navigate back to the home page or open the presentation summary.
- **What to Say:**
  > *"Finally, from a scientific standpoint, we document our system's boundaries honestly. The model achieves **96.40% accuracy** on the 1,499 PlantVillage test split, but operates under a **closed-set assumption**—meaning that an image of a non-leaf or unsupported crop will be assigned to one of the 13 classes. Furthermore, there is known visual confusion between Tomato Early Blight and Late Blight due to convergent necrotic lesion patterns.
  > 
  > In our upcoming improvement phase, we plan to implement out-of-distribution (OOD) energy-based rejection, fine-tune on in-field datasets like PlantDoc, and deploy mobile offline edge inference with TensorFlow Lite.
  > 
  > In conclusion, CropScan AI integrates deep transfer learning, visual explainability, relational surveillance, and modern web engineering into a cohesive, production-ready agricultural decision-support framework.
  > 
  > Thank you, and I now welcome any questions."*
