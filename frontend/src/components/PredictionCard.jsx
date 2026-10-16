import React, { useState } from 'react';
import ConfidenceBadge from './ConfidenceBadge';
import { parseClassName, formatConfidence } from '../utils/validation';
import './PredictionCard.css';

/**
 * PredictionCard — presents AI prediction results cleanly.
 *
 * Props:
 *   prediction: {
 *     predicted_class: string,
 *     confidence: number,
 *     confidence_pct: number,
 *     inference_ms: number,
 *     all_probabilities: Object
 *   }
 */
export default function PredictionCard({ prediction }) {
  const [showAllProbabilities, setShowAllProbabilities] = useState(false);

  if (!prediction) return null;

  const {
    predicted_class,
    confidence,
    inference_ms,
    all_probabilities = {}
  } = prediction;

  const parsed = parseClassName(predicted_class);
  const confidencePct = confidence > 1 ? confidence : confidence * 100;
  const isLowConfidence = confidencePct < 50;

  // Sort all_probabilities descending
  const sortedProbabilities = Object.entries(all_probabilities)
    .map(([cls, prob]) => ({
      rawClass: cls,
      parsed: parseClassName(cls),
      prob: typeof prob === 'number' ? prob : parseFloat(prob) || 0
    }))
    .sort((a, b) => b.prob - a.prob);

  const displayedProbs = showAllProbabilities
    ? sortedProbabilities
    : sortedProbabilities.slice(0, 3);

  return (
    <div className="prediction-card card fade-in">
      <div className="prediction-card__header">
        <div className="prediction-card__crop-tag">
          <span className="prediction-card__leaf-icon" aria-hidden="true">🌱</span>
          <span className="prediction-card__crop-name">{parsed.crop}</span>
          <span className={`prediction-card__status-tag ${parsed.isHealthy ? 'status--healthy' : 'status--diseased'}`}>
            {parsed.isHealthy ? 'Healthy Leaf' : 'Pathology Detected'}
          </span>
        </div>
        <ConfidenceBadge confidence={confidence} size="md" />
      </div>

      <div className="prediction-card__body">
        <h2 className="prediction-card__diagnosis" title={parsed.disease}>
          {parsed.disease}
        </h2>
        <div className="prediction-card__meta">
          <span className="prediction-card__raw-name" title={predicted_class}>
            Model class: <code>{predicted_class}</code>
          </span>
          {typeof inference_ms === 'number' && (
            <span className="prediction-card__latency">
              ⚡ Inference: <strong>{inference_ms.toFixed(1)} ms</strong>
            </span>
          )}
        </div>
      </div>

      {isLowConfidence && (
        <div className="prediction-card__low-confidence-warning" role="alert">
          <span className="warning-icon" aria-hidden="true">⚠️</span>
          <div>
            <strong>Low confidence prediction ({formatConfidence(confidence)}).</strong>
            <p>The leaf may be out of focus, poorly lit, or belong to a crop type outside the trained classes. Consider capturing a clearer, centered photo in natural light.</p>
          </div>
        </div>
      )}

      {sortedProbabilities.length > 0 && (
        <div className="prediction-card__distribution">
          <div className="prediction-card__distribution-header">
            <h3 className="prediction-card__section-title">
              Probability Distribution
            </h3>
            {sortedProbabilities.length > 3 && (
              <button
                type="button"
                className="btn btn--secondary btn--sm"
                onClick={() => setShowAllProbabilities((prev) => !prev)}
                aria-expanded={showAllProbabilities}
              >
                {showAllProbabilities ? 'Show Top 3' : `View All (${sortedProbabilities.length})`}
              </button>
            )}
          </div>

          <div className="prediction-card__bars" role="list">
            {displayedProbs.map((item, idx) => {
              const probPct = item.prob > 1 ? item.prob : item.prob * 100;
              const isTop = idx === 0;

              return (
                <div key={item.rawClass} className="prob-item" role="listitem">
                  <div className="prob-item__labels">
                    <span className="prob-item__name">
                      {isTop && <span className="prob-item__star" aria-hidden="true">★ </span>}
                      {item.parsed.crop} - {item.parsed.disease}
                    </span>
                    <span className="prob-item__val">
                      {formatConfidence(item.prob)}
                    </span>
                  </div>
                  <div className="prob-item__track">
                    <div
                      className={`prob-item__fill ${isTop ? 'prob-item__fill--top' : ''}`}
                      style={{ width: `${Math.max(probPct, 0.5)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
