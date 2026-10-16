import React from 'react';
import { formatConfidence } from '../utils/validation';
import './ConfidenceBadge.css';

/**
 * ConfidenceBadge — visual indicator of model confidence score.
 *
 * Props:
 *   confidence {number} - 0.0 to 1.0 (or percentage 0 to 100)
 *   size       {string} - 'sm' | 'md' | 'lg' (default 'md')
 *   showLabel  {boolean} - whether to show "Model confidence: " prefix
 */
export default function ConfidenceBadge({ confidence, size = 'md', showLabel = true }) {
  if (confidence === undefined || confidence === null || isNaN(confidence)) {
    return <span className="confidence-badge confidence-badge--unknown">N/A</span>;
  }

  // Normalize to 0-100 range
  const numValue = confidence > 1 ? confidence : confidence * 100;
  const formatted = formatConfidence(confidence);

  let tone = 'high'; // green >= 80%
  let toneLabel = 'High Confidence';
  if (numValue < 50) {
    tone = 'low';
    toneLabel = 'Low Confidence';
  } else if (numValue < 80) {
    tone = 'moderate';
    toneLabel = 'Moderate Confidence';
  }

  return (
    <span
      className={`confidence-badge confidence-badge--${tone} confidence-badge--${size}`}
      title={`${toneLabel} (${formatted})`}
    >
      <span className="confidence-badge__dot" aria-hidden="true" />
      <span className="confidence-badge__text">
        {showLabel ? `Model confidence: ${formatted}` : formatted}
      </span>
    </span>
  );
}
