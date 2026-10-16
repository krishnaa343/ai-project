import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const SUPPORTED_CROPS = [
  { name: 'Apple', icon: '🍎', diseases: 'Apple Scab, Black Rot, Healthy' },
  { name: 'Corn (Maize)', icon: '🌽', diseases: 'Common Rust, Northern Leaf Blight, Healthy' },
  { name: 'Potato', icon: '🥔', diseases: 'Early Blight, Late Blight, Healthy' },
  { name: 'Tomato', icon: '🍅', diseases: 'Bacterial Spot, Early Blight, Late Blight, Healthy' },
];

export default function Home() {
  return (
    <div className="home-page fade-in">
      {/* Hero Section */}
      <section className="hero">
        <div className="container hero__container">
          <div className="hero__badge">
            <span className="badge-pulse" />
            AI-Powered Crop Pathology Diagnostic System
          </div>
          <h1 className="hero__title">
            Instant Crop Disease Diagnosis with{' '}
            <span className="text-gradient">Explainable AI</span>
          </h1>
          <p className="hero__subtitle">
            Upload a clear photo of an affected leaf to receive real-time deep neural network identification and Grad-CAM visual attention heatmaps in milliseconds.
          </p>

          <div className="hero__actions">
            <Link to="/detect" className="btn btn--primary btn--lg">
              <span>Diagnose Leaf Now</span>
              <span className="btn-arrow" aria-hidden="true">→</span>
            </Link>
            <a href="#how-it-works" className="btn btn--secondary btn--lg">
              How It Works
            </a>
          </div>

          <div className="hero__stats">
            <div className="stat-card">
              <span className="stat-card__val">13</span>
              <span className="stat-card__label">Crop & Pathology Classes</span>
            </div>
            <div className="stat-card">
              <span className="stat-card__val">&lt; 100ms</span>
              <span className="stat-card__label">Average Inference Latency</span>
            </div>
            <div className="stat-card">
              <span className="stat-card__val">Grad-CAM</span>
              <span className="stat-card__label">Visual AI Explainability</span>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="section how-it-works">
        <div className="container">
          <div className="section-header">
            <span className="section-tag">Diagnostic Workflow</span>
            <h2 className="section-title">How Our AI Diagnoses Plant Health</h2>
            <p className="section-subtitle">
              From field photograph to transparent explainable diagnosis in four automated steps.
            </p>
          </div>

          <div className="steps-grid">
            <div className="step-card">
              <div className="step-card__num">01</div>
              <h3 className="step-card__title">Capture & Upload</h3>
              <p className="step-card__desc">
                Select or drag-and-drop a leaf image from your smartphone or desktop (JPEG, PNG, WebP up to 10MB).
              </p>
            </div>

            <div className="step-card">
              <div className="step-card__num">02</div>
              <h3 className="step-card__title">Neural Inference</h3>
              <p className="step-card__desc">
                Our fine-tuned MobileNetV2 architecture inspects morphological texture, color patterns, and lesion margins.
              </p>
            </div>

            <div className="step-card">
              <div className="step-card__num">03</div>
              <h3 className="step-card__title">Grad-CAM Attention</h3>
              <p className="step-card__desc">
                Class activation mapping tracks the exact convolutional gradients that informed the model's decision.
              </p>
            </div>

            <div className="step-card">
              <div className="step-card__num">04</div>
              <h3 className="step-card__title">Transparent Results</h3>
              <p className="step-card__desc">
                Review the predicted disease condition, model confidence score, and visual heatmap overlay.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Supported Crops */}
      <section className="section supported-crops">
        <div className="container">
          <div className="section-header">
            <span className="section-tag">Agricultural Coverage</span>
            <h2 className="section-title">Supported Crops & Pathologies</h2>
            <p className="section-subtitle">
              Trained on thousands of curated plant pathology samples across major staple and cash crops.
            </p>
          </div>

          <div className="crops-grid">
            {SUPPORTED_CROPS.map((crop) => (
              <div key={crop.name} className="crop-pill">
                <span className="crop-pill__icon" aria-hidden="true">{crop.icon}</span>
                <div className="crop-pill__content">
                  <strong className="crop-pill__name">{crop.name}</strong>
                  <span className="crop-pill__diseases">{crop.diseases}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="cta-banner">
            <div className="cta-banner__content">
              <h3>Have a crop leaf showing spots or discoloration?</h3>
              <p>Test the model now to verify its health status in seconds.</p>
            </div>
            <Link to="/detect" className="btn btn--primary">
              Run Free Diagnosis
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
