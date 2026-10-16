import React, { useState, useEffect } from 'react';
import './LoadingState.css';

const LOADING_MESSAGES = [
  'Uploading and validating image...',
  'Pre-processing leaf tensors...',
  'Running MobileNetV2 neural network...',
  'Computing Grad-CAM attention heatmap...',
  'Finalizing diagnosis & confidence scores...'
];

/**
 * LoadingState — accessible animated indicator showing pipeline stages.
 */
export default function LoadingState({ message }) {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    if (message) return; // If parent provided a static message, don't cycle
    const interval = setInterval(() => {
      setMsgIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [message]);

  const displayMessage = message || LOADING_MESSAGES[msgIndex];

  return (
    <div className="loading-state fade-in" role="status" aria-live="polite">
      <div className="loading-state__spinner-wrap">
        <div className="loading-state__spinner" />
        <div className="loading-state__pulse-ring" />
      </div>
      <p className="loading-state__message">{displayMessage}</p>
      <span className="loading-state__subtext">Please wait while the AI analyzes your crop sample</span>
    </div>
  );
}
