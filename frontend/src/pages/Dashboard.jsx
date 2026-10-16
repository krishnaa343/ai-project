import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { fetchPredictions, ApiError } from '../services/api';
import './Dashboard.css';

/**
 * Format timestamp into readable local date and time string.
 */
function formatDateTime(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoString;
  }
}

/**
 * Format confidence numeric value (0–1 or 0–100) to percentage string.
 */
function formatConfidencePct(val) {
  if (typeof val !== 'number' || isNaN(val)) return '0.00%';
  const num = val > 1 ? val : val * 100;
  return `${num.toFixed(2)}%`;
}

export default function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Search & Filtering
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL'); // 'ALL' | 'HEALTHY' | 'DISEASED'
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Fetch up to 500 records to calculate accurate client-side metrics
      const data = await fetchPredictions({ page: 1, limit: 500 });
      setPredictions(data.predictions || []);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Unable to load prediction history. Please check your backend connection.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // ── Metrics Calculation (derived strictly from real DB records) ────────────
  const metrics = useMemo(() => {
    const total = predictions.length;
    if (total === 0) {
      return {
        total: 0,
        diseased: 0,
        healthy: 0,
        diseasedPct: '0.0',
        healthyPct: '0.0',
        avgConfidencePct: '0.00%',
        mostPrevalentDisease: 'N/A',
        mostPrevalentCount: 0,
        cropDistribution: [],
        diseaseDistribution: [],
      };
    }

    let healthy = 0;
    let diseased = 0;
    let confidenceSum = 0;

    const cropCounts = {};
    const diseaseCounts = {};

    predictions.forEach((p) => {
      // Health categorization
      if (p.is_healthy) {
        healthy += 1;
      } else {
        diseased += 1;
        // Count prevalent disease (excluding healthy samples)
        const dName = p.display_name || p.predicted_class || 'Unknown Disease';
        diseaseCounts[dName] = (diseaseCounts[dName] || 0) + 1;
      }

      // Numeric confidence sum
      const conf = typeof p.confidence === 'number' ? p.confidence : parseFloat(p.confidence) || 0;
      confidenceSum += conf > 1 ? conf / 100 : conf;

      // Crop distribution
      const cropName = p.crop || 'Unknown';
      cropCounts[cropName] = (cropCounts[cropName] || 0) + 1;
    });

    const avgConf = (confidenceSum / total) * 100;

    // Most prevalent disease
    let topDisease = 'N/A';
    let topCount = 0;
    Object.entries(diseaseCounts).forEach(([name, count]) => {
      if (count > topCount) {
        topCount = count;
        topDisease = name;
      }
    });

    // Sort crops descending
    const cropDistribution = Object.entries(cropCounts)
      .map(([name, count]) => ({ name, count, pct: ((count / total) * 100).toFixed(1) }))
      .sort((a, b) => b.count - a.count);

    // Full class / condition distribution (including healthy conditions with distinct labeling)
    const conditionCounts = {};
    predictions.forEach((p) => {
      const name = p.display_name || p.predicted_class || 'Unknown';
      if (!conditionCounts[name]) {
        conditionCounts[name] = { name, count: 0, isHealthy: p.is_healthy, crop: p.crop };
      }
      conditionCounts[name].count += 1;
    });

    const diseaseDistribution = Object.values(conditionCounts)
      .map((item) => ({ ...item, pct: ((item.count / total) * 100).toFixed(1) }))
      .sort((a, b) => b.count - a.count);

    return {
      total,
      diseased,
      healthy,
      diseasedPct: ((diseased / total) * 100).toFixed(1),
      healthyPct: ((healthy / total) * 100).toFixed(1),
      avgConfidencePct: `${avgConf.toFixed(2)}%`,
      mostPrevalentDisease: topDisease,
      mostPrevalentCount: topCount,
      cropDistribution,
      diseaseDistribution,
    };
  }, [predictions]);

  // ── Filtered & Paginated Records for the Table ─────────────────────────────
  const filteredPredictions = useMemo(() => {
    return predictions.filter((p) => {
      // Status filter
      if (statusFilter === 'HEALTHY' && !p.is_healthy) return false;
      if (statusFilter === 'DISEASED' && p.is_healthy) return false;

      // Search query filter (crop, display_name, predicted_class, id)
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchCrop = (p.crop || '').toLowerCase().includes(q);
        const matchName = (p.display_name || '').toLowerCase().includes(q);
        const matchClass = (p.predicted_class || '').toLowerCase().includes(q);
        const matchId = String(p.id).includes(q);
        return matchCrop || matchName || matchClass || matchId;
      }
      return true;
    });
  }, [predictions, statusFilter, searchQuery]);

  const totalPages = Math.ceil(filteredPredictions.length / pageSize) || 1;
  const paginatedPredictions = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredPredictions.slice(start, start + pageSize);
  }, [filteredPredictions, currentPage, pageSize]);

  // Reset page when filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, statusFilter]);

  // ── Render: Loading Skeleton State ─────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="dashboard-page container fade-in">
        <header className="dashboard-header">
          <div className="dashboard-header__text">
            <h1 className="dashboard-title">Dashboard & Analytics</h1>
            <p className="dashboard-subtitle">Loading live diagnostic telemetry from MySQL database...</p>
          </div>
        </header>

        <div className="summary-grid">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="summary-card card skeleton-card">
              <div className="skeleton skeleton-icon" />
              <div className="skeleton skeleton-title" />
              <div className="skeleton skeleton-number" />
            </div>
          ))}
        </div>

        <div className="analytics-grid">
          <div className="chart-card card skeleton-card" style={{ height: '320px' }}>
            <div className="skeleton skeleton-title" />
            <div className="skeleton" style={{ height: '200px', marginTop: '1rem' }} />
          </div>
          <div className="chart-card card skeleton-card" style={{ height: '320px' }}>
            <div className="skeleton skeleton-title" />
            <div className="skeleton" style={{ height: '200px', marginTop: '1rem' }} />
          </div>
        </div>
      </div>
    );
  }

  // ── Render: Error State ────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="dashboard-page container fade-in">
        <div className="dashboard-error card">
          <span className="dashboard-error__icon" aria-hidden="true">⚠️</span>
          <h2>Unable to Load Analytics</h2>
          <p>{error}</p>
          <button type="button" className="btn btn-primary" onClick={loadData}>
            🔄 Retry Connection
          </button>
        </div>
      </div>
    );
  }

  // ── Render: Empty Database State ───────────────────────────────────────────
  if (predictions.length === 0) {
    return (
      <div className="dashboard-page container fade-in">
        <header className="dashboard-header">
          <div className="dashboard-header__text">
            <h1 className="dashboard-title">Dashboard & Analytics</h1>
            <p className="dashboard-subtitle">Monitor crop disease detection activity and prediction trends.</p>
          </div>
        </header>

        <div className="summary-grid">
          <div className="summary-card card">
            <span className="summary-card__icon">📊</span>
            <span className="summary-card__label">Total Predictions</span>
            <span className="summary-card__value">0</span>
            <span className="summary-card__meta">No activity recorded</span>
          </div>
          <div className="summary-card card">
            <span className="summary-card__icon">🍂</span>
            <span className="summary-card__label">Diseased Crops</span>
            <span className="summary-card__value">0</span>
            <span className="summary-card__meta">0% of total</span>
          </div>
          <div className="summary-card card">
            <span className="summary-card__icon">🌿</span>
            <span className="summary-card__label">Healthy Crops</span>
            <span className="summary-card__value">0</span>
            <span className="summary-card__meta">0% of total</span>
          </div>
          <div className="summary-card card">
            <span className="summary-card__icon">🎯</span>
            <span className="summary-card__label">Avg Confidence</span>
            <span className="summary-card__value">—</span>
            <span className="summary-card__meta">Awaiting records</span>
          </div>
        </div>

        <div className="dashboard-empty card">
          <span className="dashboard-empty__icon" aria-hidden="true">🍃</span>
          <h2>No Prediction History Available Yet</h2>
          <p>
            The MySQL database is connected and initialized, but contains no diagnosis records.
            Upload crop leaf images through the detector to automatically populate this dashboard.
          </p>
          <Link to="/detect" className="btn btn-primary">
            🌿 Diagnose Your First Leaf
          </Link>
        </div>
      </div>
    );
  }

  // ── Render: Full Dashboard with Real Data ──────────────────────────────────
  return (
    <div className="dashboard-page container fade-in">
      {/* ── Page Header ───────────────────────────────────────────────────── */}
      <header className="dashboard-header">
        <div className="dashboard-header__text">
          <h1 className="dashboard-title">Dashboard & Analytics</h1>
          <p className="dashboard-subtitle">
            Telemetry and trend analysis derived from <strong>{metrics.total}</strong> persistent MySQL diagnosis records.
          </p>
        </div>
        <div className="dashboard-header__actions">
          <button
            type="button"
            className="btn btn-secondary btn-sm refresh-btn"
            onClick={loadData}
            title="Refresh database records"
          >
            🔄 Sync Data
          </button>
        </div>
      </header>

      {/* ── Summary Metric Cards ──────────────────────────────────────────── */}
      <section className="summary-grid" aria-label="Key Performance Indicators">
        <div className="summary-card card">
          <div className="summary-card__header">
            <span className="summary-card__icon" aria-hidden="true">📊</span>
            <span className="summary-card__badge badge-neutral">Total Samples</span>
          </div>
          <span className="summary-card__value">{metrics.total}</span>
          <span className="summary-card__label">Total Predictions</span>
          <span className="summary-card__meta">Recorded in MySQL database</span>
        </div>

        <div className="summary-card card summary-card--diseased">
          <div className="summary-card__header">
            <span className="summary-card__icon" aria-hidden="true">🍂</span>
            <span className="summary-card__badge badge-danger">{metrics.diseasedPct}%</span>
          </div>
          <span className="summary-card__value">{metrics.diseased}</span>
          <span className="summary-card__label">Pathologies Detected</span>
          <span className="summary-card__meta">Requires agricultural mitigation</span>
        </div>

        <div className="summary-card card summary-card--healthy">
          <div className="summary-card__header">
            <span className="summary-card__icon" aria-hidden="true">🌿</span>
            <span className="summary-card__badge badge-success">{metrics.healthyPct}%</span>
          </div>
          <span className="summary-card__value">{metrics.healthy}</span>
          <span className="summary-card__label">Healthy Samples</span>
          <span className="summary-card__meta">No active pathogen detected</span>
        </div>

        <div className="summary-card card summary-card--confidence">
          <div className="summary-card__header">
            <span className="summary-card__icon" aria-hidden="true">🎯</span>
            <span className="summary-card__badge badge-info">Certainty</span>
          </div>
          <span className="summary-card__value">{metrics.avgConfidencePct}</span>
          <span className="summary-card__label">Average Confidence</span>
          <span className="summary-card__meta">Overall model certainty</span>
        </div>

        <div className="summary-card card summary-card--prevalent">
          <div className="summary-card__header">
            <span className="summary-card__icon" aria-hidden="true">🔬</span>
            <span className="summary-card__badge badge-warning">Top Outbreak</span>
          </div>
          <span className="summary-card__value summary-card__value--text" title={metrics.mostPrevalentDisease}>
            {metrics.mostPrevalentDisease}
          </span>
          <span className="summary-card__label">Most Predicted Disease</span>
          <span className="summary-card__meta">
            {metrics.mostPrevalentCount > 0
              ? `${metrics.mostPrevalentCount} incidents recorded`
              : 'No diseased samples'}
          </span>
        </div>
      </section>

      {/* ── Analytics Visualizations ──────────────────────────────────────── */}
      <section className="analytics-grid" aria-label="Analytics Visualizations">
        {/* Chart 1: Healthy vs Diseased (Donut Chart) */}
        <div className="chart-card card">
          <div className="chart-card__header">
            <h2 className="chart-card__title">Plant Health Ratio</h2>
            <span className="chart-card__subtitle">Healthy foliage vs confirmed pathogen detections</span>
          </div>

          <div className="donut-chart-container">
            <svg className="donut-chart" viewBox="0 0 160 160" aria-label="Health breakdown donut chart">
              {/* Background ring */}
              <circle
                cx="80"
                cy="80"
                r="60"
                className="donut-ring-bg"
                fill="none"
                strokeWidth="20"
              />
              {/* Healthy Segment (Circumference = 2 * PI * 60 ≈ 376.99) */}
              <circle
                cx="80"
                cy="80"
                r="60"
                className="donut-ring-healthy"
                fill="none"
                strokeWidth="20"
                strokeDasharray={`${(metrics.healthy / metrics.total) * 377} 377`}
                strokeDashoffset="0"
                transform="rotate(-90 80 80)"
              />
              {/* Diseased Segment */}
              <circle
                cx="80"
                cy="80"
                r="60"
                className="donut-ring-diseased"
                fill="none"
                strokeWidth="20"
                strokeDasharray={`${(metrics.diseased / metrics.total) * 377} 377`}
                strokeDashoffset={`-${(metrics.healthy / metrics.total) * 377}`}
                transform="rotate(-90 80 80)"
              />
              {/* Center Counter */}
              <text x="80" y="74" textAnchor="middle" className="donut-center-total">
                {metrics.total}
              </text>
              <text x="80" y="94" textAnchor="middle" className="donut-center-label">
                Samples
              </text>
            </svg>

            <div className="donut-legend">
              <div className="legend-item">
                <span className="legend-indicator legend-healthy" />
                <div className="legend-info">
                  <span className="legend-label">Healthy Leaf</span>
                  <strong className="legend-value">{metrics.healthy} ({metrics.healthyPct}%)</strong>
                </div>
              </div>
              <div className="legend-item">
                <span className="legend-indicator legend-diseased" />
                <div className="legend-info">
                  <span className="legend-label">Diseased Foliage</span>
                  <strong className="legend-value">{metrics.diseased} ({metrics.diseasedPct}%)</strong>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Chart 2: Predictions by Crop */}
        <div className="chart-card card">
          <div className="chart-card__header">
            <h2 className="chart-card__title">Surveillance by Crop</h2>
            <span className="chart-card__subtitle">Diagnostic breakdown across monitored agricultural crops</span>
          </div>

          <div className="crop-bars-container">
            {metrics.cropDistribution.map((crop) => (
              <div key={crop.name} className="crop-bar-row">
                <div className="crop-bar-header">
                  <span className="crop-bar-name">🌱 {crop.name}</span>
                  <span className="crop-bar-count">
                    <strong>{crop.count}</strong> samples ({crop.pct}%)
                  </span>
                </div>
                <div className="crop-bar-track">
                  <div
                    className="crop-bar-fill"
                    style={{ width: `${Math.max(4, parseFloat(crop.pct))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Chart 3: Disease / Class Distribution */}
      <section className="class-distribution-section" aria-label="Disease Frequency Distribution">
        <div className="chart-card card">
          <div className="chart-card__header">
            <h2 className="chart-card__title">Condition & Disease Class Distribution</h2>
            <span className="chart-card__subtitle">
              Ranked frequency of all identified diseases and healthy leaf conditions from database
            </span>
          </div>

          <div className="distribution-bars-container">
            {metrics.diseaseDistribution.map((item) => {
              const maxCount = metrics.diseaseDistribution[0]?.count || 1;
              const barWidth = Math.max(5, (item.count / maxCount) * 100);

              return (
                <div key={item.name} className="distribution-row">
                  <div className="distribution-label">
                    <span className={`status-dot ${item.isHealthy ? 'status-dot--healthy' : 'status-dot--diseased'}`} />
                    <span className="distribution-name" title={item.name}>{item.name}</span>
                    <span className="distribution-crop-tag">{item.crop}</span>
                  </div>
                  <div className="distribution-bar-track">
                    <div
                      className={`distribution-bar-fill ${item.isHealthy ? 'fill-healthy' : 'fill-diseased'}`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <div className="distribution-stat">
                    <strong>{item.count}</strong>
                    <span className="stat-pct">({item.pct}%)</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── Recent Predictions Table Section ──────────────────────────────── */}
      <section className="predictions-table-section" aria-label="Recent Prediction History Table">
        <div className="card table-card">
          <div className="table-card__header">
            <div>
              <h2 className="table-card__title">Recent Prediction History</h2>
              <span className="table-card__subtitle">
                Inspect past diagnoses stored in MySQL with confidence scores and timestamps
              </span>
            </div>

            {/* Search & Filter Controls */}
            <div className="table-filters">
              <div className="search-box">
                <span className="search-icon" aria-hidden="true">🔍</span>
                <input
                  type="text"
                  placeholder="Search by crop, disease, or ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="search-input"
                  aria-label="Search prediction records"
                />
                {searchQuery && (
                  <button
                    type="button"
                    className="clear-search-btn"
                    onClick={() => setSearchQuery('')}
                    aria-label="Clear search"
                  >
                    ×
                  </button>
                )}
              </div>

              <div className="filter-buttons" role="group" aria-label="Filter by health status">
                <button
                  type="button"
                  className={`filter-btn ${statusFilter === 'ALL' ? 'filter-btn--active' : ''}`}
                  onClick={() => setStatusFilter('ALL')}
                >
                  All ({predictions.length})
                </button>
                <button
                  type="button"
                  className={`filter-btn ${statusFilter === 'HEALTHY' ? 'filter-btn--active' : ''}`}
                  onClick={() => setStatusFilter('HEALTHY')}
                >
                  Healthy ({metrics.healthy})
                </button>
                <button
                  type="button"
                  className={`filter-btn ${statusFilter === 'DISEASED' ? 'filter-btn--active' : ''}`}
                  onClick={() => setStatusFilter('DISEASED')}
                >
                  Diseased ({metrics.diseased})
                </button>
              </div>
            </div>
          </div>

          {/* Table Container */}
          <div className="table-responsive">
            <table className="predictions-table">
              <thead>
                <tr>
                  <th scope="col" style={{ width: '70px' }}>ID</th>
                  <th scope="col">Crop</th>
                  <th scope="col">Diagnosis</th>
                  <th scope="col">Health Status</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Recorded Date & Time</th>
                  <th scope="col">Sample File</th>
                </tr>
              </thead>
              <tbody>
                {paginatedPredictions.length === 0 ? (
                  <tr>
                    <td colSpan="7" className="table-empty-row">
                      No matching records found for "{searchQuery}".
                    </td>
                  </tr>
                ) : (
                  paginatedPredictions.map((row) => (
                    <tr key={row.id}>
                      <td className="col-id">#{row.id}</td>
                      <td className="col-crop">
                        <strong>🌱 {row.crop}</strong>
                      </td>
                      <td className="col-diagnosis">
                        <span className="diagnosis-display-name">{row.display_name}</span>
                        <span className="diagnosis-class-code">{row.predicted_class}</span>
                      </td>
                      <td className="col-status">
                        <span className={`status-badge ${row.is_healthy ? 'status-badge--healthy' : 'status-badge--diseased'}`}>
                          {row.is_healthy ? '🌿 Healthy' : '🍂 Disease Detected'}
                        </span>
                      </td>
                      <td className="col-confidence">
                        <div className="confidence-cell">
                          <span className="confidence-text">{formatConfidencePct(row.confidence)}</span>
                          <div className="confidence-bar-track">
                            <div
                              className={`confidence-bar-fill ${row.confidence >= 0.8 ? 'conf-high' : row.confidence >= 0.5 ? 'conf-med' : 'conf-low'}`}
                              style={{ width: `${Math.min(100, (row.confidence > 1 ? row.confidence : row.confidence * 100))}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="col-date">
                        {formatDateTime(row.created_at)}
                      </td>
                      <td className="col-file" title={row.image_name}>
                        <code>{row.image_name ? row.image_name.slice(-25) : '—'}</code>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Table Pagination Footer */}
          {filteredPredictions.length > pageSize && (
            <div className="table-pagination">
              <span className="pagination-info">
                Showing <strong>{((currentPage - 1) * pageSize) + 1}</strong> to{' '}
                <strong>{Math.min(currentPage * pageSize, filteredPredictions.length)}</strong> of{' '}
                <strong>{filteredPredictions.length}</strong> records
              </span>

              <div className="pagination-controls">
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={currentPage <= 1}
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                >
                  ← Previous
                </button>
                <span className="page-indicator">
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={currentPage >= totalPages}
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                >
                  Next →
                </button>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
