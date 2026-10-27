import React from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import PredictionCard from '../components/PredictionCard';
import GradCamViewer from '../components/GradCamViewer';
import DiseaseInfoPanel from '../components/DiseaseInfoPanel';
import { formatFileSize } from '../utils/validation';
import './Results.css';

export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();

  const state = location.state;

  if (!state || !state.prediction) {
    return (
      <div className="results-page container fade-in">
        <div className="results-empty card">
          <span className="results-empty__icon" aria-hidden="true">🍃</span>
          <h2>No Diagnosis Found</h2>
          <p>
            It looks like no leaf image has been analyzed yet, or this page was refreshed.
          </p>
          <Link to="/detect" className="btn btn--primary">
            Upload a Leaf Sample
          </Link>
        </div>
      </div>
    );
  }

  const { prediction, fileName, fileSize, previewUrl } = state;

  // ── Render: Non-Leaf Rejected State ────────────────────────────────────────
  if (prediction.status === 'rejected' || prediction.reason === 'non_leaf_image') {
    const nonLeafConf = prediction.leaf_detection?.confidence
      ? (prediction.leaf_detection.confidence > 1
          ? prediction.leaf_detection.confidence.toFixed(1)
          : (prediction.leaf_detection.confidence * 100).toFixed(1)) + '%'
      : null;

    return (
      <div className="results-page container fade-in">
        <div className="rejection-card card">
          <div className="rejection-icon" aria-hidden="true">🍂 🚫</div>
          <span className="rejection-badge">Input Validation Gate</span>
          <h1 className="rejection-title">Leaf Image Not Detected</h1>
          <p className="rejection-message">
            {prediction.message || 'The uploaded image does not appear to be a crop leaf.'}
          </p>
          <div className="rejection-details">
            <p>
              Our automated input validation gate determined that this image does not match the foliar characteristics of supported agricultural crops.
            </p>
            {nonLeafConf && (
              <div className="rejection-confidence">
                <span>Non-Leaf Confidence:</span>
                <strong>{nonLeafConf}</strong>
              </div>
            )}
          </div>
          <div className="rejection-guidance">
            <h3>Diagnostic Photo Tips:</h3>
            <ul>
              <li>Upload a clear, focused photograph of an individual plant leaf.</li>
              <li>Avoid background clutter, human hands, animals, insects, or unrelated objects.</li>
              <li>Ensure balanced daylight lighting without excessive direct flash or glare.</li>
            </ul>
          </div>
          <div className="rejection-actions">
            <button
              type="button"
              className="btn btn--primary btn--lg"
              onClick={() => navigate('/detect')}
            >
              🔄 Try Another Image
            </button>
            <Link to="/" className="btn btn--secondary btn--lg">
              Return Home
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // disease_info may be present (object), null (no entry), or undefined (old response)
  const diseaseInfo = prediction.disease_info !== undefined
    ? prediction.disease_info
    : null;

  return (
    <div className="results-page container fade-in">
      {/* ── Page header ───────────────────────────────────────────────────── */}
      <div className="results-header">
        <div className="results-header__info">
          <span className="section-tag">Analysis Complete</span>
          <h1 className="results-header__title">Crop Pathology Diagnosis</h1>
          {fileName && (
            <p className="results-header__file">
              Sample: <strong>{fileName}</strong> {fileSize && `(${formatFileSize(fileSize)})`}
            </p>
          )}
        </div>

        <div className="results-header__actions">
          <button
            type="button"
            className="btn btn--secondary"
            onClick={() => window.print()}
            title="Print or save this diagnostic report"
          >
            🖨️ Print Report
          </button>
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => navigate('/detect')}
          >
            Diagnose Another Leaf
          </button>
        </div>
      </div>

      {/* ── Row 1: Prediction card + Grad-CAM viewer ──────────────────────── */}
      <div className="results-grid">
        <div className="results-grid__col">
          <PredictionCard prediction={prediction} />
        </div>

        {prediction.gradcam && (
          <div className="results-grid__col">
            <GradCamViewer
              originalUrl={previewUrl}
              gradcam={prediction.gradcam}
            />
          </div>
        )}
      </div>

      {/* ── Disease Information Panel (full-width) ─────────────────────────── */}
      <div className="results-info-row">
        <DiseaseInfoPanel diseaseInfo={diseaseInfo} />
      </div>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <div className="results-footer">
        <button
          type="button"
          className="btn btn--secondary"
          onClick={() => navigate('/detect')}
        >
          ← Return to Leaf Uploader
        </button>
      </div>
    </div>
  );
}
