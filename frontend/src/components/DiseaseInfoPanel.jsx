import React, { useState } from 'react';
import './DiseaseInfoPanel.css';

/**
 * DiseaseInfoPanel — displays structured disease information returned by the backend.
 *
 * Props:
 *   diseaseInfo {Object|null} - The disease_info field from the API response:
 *     {
 *       display_name, crop, type, is_healthy, description,
 *       symptoms, management, prevention, severity_note, disclaimer
 *     }
 *     OR null if unavailable.
 */
export default function DiseaseInfoPanel({ diseaseInfo }) {
  const [mgmtOpen, setMgmtOpen] = useState(true);
  const [prevOpen, setPrevOpen] = useState(false);

  // ── Missing information state ─────────────────────────────────────────────
  if (diseaseInfo === undefined || diseaseInfo === null) {
    return (
      <div className="disease-panel card fade-in" aria-label="Disease information unavailable">
        <div className="disease-panel__header">
          <span className="disease-panel__tag">About This Condition</span>
          <h2 className="disease-panel__title">Information Unavailable</h2>
        </div>
        <p className="disease-panel__unavailable">
          Disease information is currently unavailable for this predicted class.
          The prediction result above is still valid.
        </p>
        <div className="disease-panel__disclaimer" role="note">
          <strong>Important:</strong> AI predictions should always be verified with a qualified
          agricultural expert before making field-level treatment decisions.
        </div>
      </div>
    );
  }

  const {
    display_name,
    crop,
    is_healthy,
    description,
    symptoms = [],
    management = [],
    prevention = [],
    severity_note,
    disclaimer,
  } = diseaseInfo;

  return (
    <div
      className={`disease-panel card fade-in ${is_healthy ? 'disease-panel--healthy' : 'disease-panel--diseased'}`}
      aria-label={`Disease information: ${display_name}`}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="disease-panel__header">
        <div>
          <span className={`disease-panel__tag ${is_healthy ? 'disease-panel__tag--healthy' : 'disease-panel__tag--disease'}`}>
            {is_healthy ? 'Healthy Leaf' : 'About This Condition'}
          </span>
          <h2 className="disease-panel__title">{display_name}</h2>
          <p className="disease-panel__crop">
            <span className="disease-panel__crop-label">Crop:</span> {crop}
          </p>
        </div>
        {is_healthy ? (
          <span className="disease-panel__healthy-icon" aria-hidden="true">🌿</span>
        ) : (
          <span className="disease-panel__disease-icon" aria-hidden="true">🔬</span>
        )}
      </div>

      {/* ── Severity note (diseases only) ──────────────────────────────────── */}
      {!is_healthy && severity_note && (
        <div className="disease-panel__severity" role="note">
          <span className="disease-panel__severity-icon" aria-hidden="true">⚠️</span>
          <span>{severity_note}</span>
        </div>
      )}

      {/* ── Description ────────────────────────────────────────────────────── */}
      <div className="disease-panel__section">
        <h3 className="disease-panel__section-title">
          {is_healthy ? 'Model Assessment' : 'Description'}
        </h3>
        <p className="disease-panel__description">{description}</p>
      </div>

      {/* ── Symptoms (diseases only) ───────────────────────────────────────── */}
      {!is_healthy && symptoms.length > 0 && (
        <div className="disease-panel__section">
          <h3 className="disease-panel__section-title">Common Visible Symptoms</h3>
          <ul className="disease-panel__list" aria-label="Common symptoms">
            {symptoms.map((s, i) => (
              <li key={i} className="disease-panel__list-item">
                <span className="disease-panel__bullet" aria-hidden="true">•</span>
                {s}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Management (collapsible) ───────────────────────────────────────── */}
      {management.length > 0 && (
        <div className="disease-panel__section">
          <button
            type="button"
            className="disease-panel__accordion"
            onClick={() => setMgmtOpen(o => !o)}
            aria-expanded={mgmtOpen}
          >
            <h3 className="disease-panel__section-title">
              {is_healthy ? 'Maintenance Guidance' : 'General Management'}
            </h3>
            <span className={`disease-panel__chevron ${mgmtOpen ? 'disease-panel__chevron--open' : ''}`} aria-hidden="true">▾</span>
          </button>
          {mgmtOpen && (
            <ul className="disease-panel__list fade-in" aria-label={is_healthy ? 'Maintenance guidance' : 'General management steps'}>
              {management.map((m, i) => (
                <li key={i} className="disease-panel__list-item">
                  <span className="disease-panel__bullet" aria-hidden="true">•</span>
                  {m}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* ── Prevention (collapsible) ──────────────────────────────────────── */}
      {prevention.length > 0 && (
        <div className="disease-panel__section">
          <button
            type="button"
            className="disease-panel__accordion"
            onClick={() => setPrevOpen(o => !o)}
            aria-expanded={prevOpen}
          >
            <h3 className="disease-panel__section-title">Prevention</h3>
            <span className={`disease-panel__chevron ${prevOpen ? 'disease-panel__chevron--open' : ''}`} aria-hidden="true">▾</span>
          </button>
          {prevOpen && (
            <ul className="disease-panel__list fade-in" aria-label="Prevention steps">
              {prevention.map((p, i) => (
                <li key={i} className="disease-panel__list-item">
                  <span className="disease-panel__bullet" aria-hidden="true">•</span>
                  {p}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* ── Disclaimer ─────────────────────────────────────────────────────── */}
      <div className="disease-panel__disclaimer" role="note">
        <strong>Important:</strong>{' '}
        {disclaimer
          ? disclaimer
          : 'This is general guidance information. AI predictions should be verified with a qualified agricultural expert before making field-level treatment decisions.'}
      </div>
    </div>
  );
}
