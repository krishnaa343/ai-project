import React, { useState } from 'react';
import { resolveGradCamUrl } from '../services/api';
import './GradCamViewer.css';

/**
 * GradCamViewer — side-by-side visualization of the original image and Grad-CAM attention heatmap.
 *
 * Props:
 *   originalUrl {string} - URL or object URL of the uploaded image
 *   gradcam     {Object} - { image_url: string, target_layer?: string }
 */
export default function GradCamViewer({ originalUrl, gradcam }) {
  const [activeTab, setActiveTab] = useState('side-by-side'); // 'side-by-side' | 'gradcam-only' | 'original-only'
  const [imgError, setImgError] = useState(false);

  if (!gradcam || !gradcam.image_url) {
    return (
      <div className="gradcam-card card">
        <div className="gradcam-card__empty">
          <p>No Grad-CAM explanation was generated for this analysis.</p>
        </div>
      </div>
    );
  }

  const resolvedGradCamUrl = resolveGradCamUrl(gradcam.image_url);

  return (
    <div className="gradcam-card card fade-in">
      <div className="gradcam-card__header">
        <div>
          <h3 className="gradcam-card__title">
            <span className="gradcam-card__badge">Explainable AI</span>
            Visual Attention (Grad-CAM)
          </h3>
          <p className="gradcam-card__subtitle">
            See where the convolutional neural network focused its attention to classify this leaf.
          </p>
        </div>

        <div className="gradcam-card__tabs" role="tablist" aria-label="Grad-CAM view options">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'side-by-side'}
            className={`tab-btn ${activeTab === 'side-by-side' ? 'tab-btn--active' : ''}`}
            onClick={() => setActiveTab('side-by-side')}
          >
            Side-by-Side
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'gradcam-only'}
            className={`tab-btn ${activeTab === 'gradcam-only' ? 'tab-btn--active' : ''}`}
            onClick={() => setActiveTab('gradcam-only')}
          >
            Heatmap Only
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'original-only'}
            className={`tab-btn ${activeTab === 'original-only' ? 'tab-btn--active' : ''}`}
            onClick={() => setActiveTab('original-only')}
          >
            Original
          </button>
        </div>
      </div>

      <div className={`gradcam-card__viewer-grid gradcam-card__viewer-grid--${activeTab}`}>
        {(activeTab === 'side-by-side' || activeTab === 'original-only') && (
          <div className="gradcam-view-panel">
            <div className="gradcam-view-panel__badge">Original Leaf</div>
            <div className="gradcam-view-panel__image-box">
              {originalUrl ? (
                <img
                  src={originalUrl}
                  alt="Original uploaded crop leaf"
                  className="gradcam-img"
                />
              ) : (
                <div className="gradcam-placeholder">Original image not available</div>
              )}
            </div>
          </div>
        )}

        {(activeTab === 'side-by-side' || activeTab === 'gradcam-only') && (
          <div className="gradcam-view-panel">
            <div className="gradcam-view-panel__badge gradcam-view-panel__badge--accent">
              Grad-CAM Overlay
            </div>
            <div className="gradcam-view-panel__image-box">
              {imgError ? (
                <div className="gradcam-placeholder gradcam-placeholder--error">
                  <p>⚠️ Unable to load Grad-CAM overlay.</p>
                  <a
                    href={resolvedGradCamUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn--secondary btn--sm"
                  >
                    Open directly in new tab
                  </a>
                </div>
              ) : (
                <img
                  src={resolvedGradCamUrl}
                  alt="Grad-CAM attention heatmap overlay showing model focus"
                  className="gradcam-img"
                  onError={() => setImgError(true)}
                />
              )}
            </div>
          </div>
        )}
      </div>

      {gradcam.target_layer && (
        <div className="gradcam-card__layer-info">
          <span>Target Convolutional Layer: <code>{gradcam.target_layer}</code></span>
        </div>
      )}

      {/* Explanatory color legend */}
      <div className="gradcam-card__legend">
        <span className="legend-label">Heatmap Attention Intensity:</span>
        <div className="legend-bar-wrap">
          <span>Low (0.0)</span>
          <div className="legend-gradient" />
          <span>High (1.0)</span>
        </div>
      </div>

      {/* Mandatory scientific disclaimer */}
      <div className="gradcam-card__disclaimer" role="note">
        <strong>Explainability Note:</strong> Highlighted regions indicate visual areas that contributed most strongly to the model's prediction. This reflects AI activation patterns and is not a definitive biological test.
      </div>
    </div>
  );
}
