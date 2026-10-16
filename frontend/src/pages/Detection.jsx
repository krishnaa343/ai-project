import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUploader from '../components/ImageUploader';
import ImagePreview from '../components/ImagePreview';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { predictDisease, checkHealth } from '../services/api';
import './Detection.css';

export default function Detection() {
  const navigate = useNavigate();

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [includeGradcam, setIncludeGradcam] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ checked: false, online: true, message: '' });

  // Check backend health on mount
  useEffect(() => {
    let isMounted = true;
    checkHealth()
      .then((res) => {
        if (!isMounted) return;
        if (res.model_loaded) {
          setBackendStatus({ checked: true, online: true, message: '' });
        } else {
          setBackendStatus({
            checked: true,
            online: false,
            message: 'AI Model is not loaded on backend server.'
          });
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        setBackendStatus({
          checked: true,
          online: false,
          message: 'Unable to reach AI backend server at localhost:5000. Please ensure the Flask service is running.'
        });
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Cleanup previewUrl on unmount or file change
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = useCallback((file) => {
    setError(null);
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }, [previewUrl]);

  const handleRemove = useCallback(() => {
    if (isAnalyzing) return;
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
  }, [isAnalyzing, previewUrl]);

  const handleDiagnose = async (e) => {
    e.preventDefault();
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await predictDisease(selectedFile, {
        includeGradcam,
        targetLayer: null // use backend default
      });

      // Navigate to results page with prediction and image data
      navigate('/results', {
        state: {
          prediction: response,
          fileName: selectedFile.name,
          fileSize: selectedFile.size,
          previewUrl: previewUrl
        }
      });
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during leaf analysis.');
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="detection-page container fade-in">
      <div className="detection-header">
        <span className="section-tag">Diagnostics</span>
        <h1 className="detection-title">Analyze Crop Leaf Sample</h1>
        <p className="detection-subtitle">
          Submit a high-resolution leaf photograph to run deep-learning pathology inference with explainable visual attention.
        </p>
      </div>

      {backendStatus.checked && !backendStatus.online && (
        <div className="detection-banner">
          <ErrorMessage
            message={backendStatus.message}
            type="warning"
            onDismiss={() => setBackendStatus((s) => ({ ...s, checked: false }))}
          />
        </div>
      )}

      {error && (
        <div className="detection-banner">
          <ErrorMessage
            message={error}
            type="error"
            onDismiss={() => setError(null)}
          />
        </div>
      )}

      <div className="detection-card card">
        {!selectedFile ? (
          <ImageUploader
            onFileSelect={handleFileSelect}
            onError={(msg) => setError(msg)}
            disabled={isAnalyzing}
          />
        ) : (
          <div className="detection-stage">
            <ImagePreview
              file={selectedFile}
              previewUrl={previewUrl}
              onRemove={handleRemove}
              disabled={isAnalyzing}
            />

            {isAnalyzing ? (
              <LoadingState />
            ) : (
              <div className="detection-controls fade-in">
                <label className="checkbox-control">
                  <input
                    type="checkbox"
                    checked={includeGradcam}
                    onChange={(e) => setIncludeGradcam(e.target.checked)}
                    disabled={isAnalyzing}
                  />
                  <span>
                    <strong>Generate Grad-CAM Visual Heatmap</strong> (recommended)
                  </span>
                </label>

                <div className="detection-buttons">
                  <button
                    type="button"
                    className="btn btn--secondary"
                    onClick={handleRemove}
                    disabled={isAnalyzing}
                  >
                    Choose Different Image
                  </button>

                  <button
                    type="button"
                    className="btn btn--primary btn--lg"
                    onClick={handleDiagnose}
                    disabled={isAnalyzing || !selectedFile}
                  >
                    <span>Run AI Diagnosis</span>
                    <span className="btn-arrow" aria-hidden="true">→</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Guidelines Box */}
      <div className="detection-tips card">
        <h3 className="detection-tips__title">
          <span aria-hidden="true">💡</span> Best Practices for High Diagnostic Accuracy
        </h3>
        <ul className="detection-tips__list">
          <li><strong>Focus on Leaf Symptoms:</strong> Frame the affected leaf clearly in the center of the photo.</li>
          <li><strong>Good Lighting:</strong> Take photos in bright, natural daylight avoiding heavy shadows or glare.</li>
          <li><strong>Single Leaf:</strong> Avoid crowded backgrounds with soil, hands, or multiple overlapping plants.</li>
          <li><strong>Supported Formats:</strong> JPEG, PNG, or WebP up to 10 MB.</li>
        </ul>
      </div>
    </div>
  );
}
