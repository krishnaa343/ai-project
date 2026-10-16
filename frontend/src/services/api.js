/**
 * API Service
 * Project: AI-Based Intelligent Crop Disease Detection Framework
 *
 * All network communication with the Flask backend is centralised here.
 * UI components must NOT contain fetch/axios calls directly.
 *
 * Actual backend response shape (POST /api/predict?gradcam=true):
 * {
 *   status:            "success" | "error"
 *   predicted_class:   "Tomato___Early_blight"
 *   class_index:       10
 *   confidence:        0.9421
 *   confidence_pct:    94.21
 *   all_probabilities: { "Tomato___Early_blight": 0.9421, ... }
 *   inference_ms:      105.3 | null
 *   gradcam: {
 *     image_url:   "/static/gradcam/gradcam_<uuid>.png"
 *     target_layer: "out_relu"
 *   } | null
 *   message:           "..." (only on error)
 * }
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

// ── Error class ───────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(message, statusCode = null, isNetwork = false) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.isNetwork = isNetwork;
  }
}

// ── Health check ──────────────────────────────────────────────────────────────

/**
 * Checks whether the Flask backend is reachable and the model is loaded.
 * @returns {Promise<{status: string, model_loaded: boolean, num_classes: number}>}
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) {
      throw new ApiError('Health check failed.', response.status);
    }
    return await response.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Cannot reach the backend server.', null, true);
  }
}

// ── Prediction ────────────────────────────────────────────────────────────────

/**
 * Sends an image file to the backend for disease prediction + Grad-CAM.
 *
 * @param {File} imageFile - The image file selected by the user.
 * @param {Object} [options] - Optional configuration.
 * @param {boolean} [options.includeGradcam=true] - Whether to request Grad-CAM heatmap.
 * @param {AbortSignal} [options.signal] - Optional AbortController signal for cancellation.
 * @returns {Promise<Object>} The validated backend prediction response.
 * @throws {ApiError} On network failure, non-2xx status, or malformed response.
 */
export async function predictDisease(imageFile, options = {}) {
  if (!imageFile) {
    throw new ApiError('No image file provided.');
  }

  const { includeGradcam = true, signal = null } = options;

  const formData = new FormData();
  formData.append('file', imageFile);

  const gradcamParam = includeGradcam ? 'true' : 'false';

  let response;
  try {
    response = await fetch(`${API_BASE}/api/predict?gradcam=${gradcamParam}`, {
      method: 'POST',
      body: formData,
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError('Request was cancelled.', null, false);
    }
    throw new ApiError(
      'Could not connect to the analysis server. Please ensure the backend is running.',
      null,
      true
    );
  }

  // Parse JSON body regardless of status code so we can show backend error messages
  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      `Server returned an unexpected response (HTTP ${response.status}).`,
      response.status
    );
  }

  // Handle known HTTP error codes with user-friendly messages
  if (!response.ok) {
    const backendMessage = data?.message || '';
    if (response.status === 400) {
      throw new ApiError(backendMessage || 'Invalid image or request.', 400);
    }
    if (response.status === 413) {
      throw new ApiError('Image file is too large for the server.', 413);
    }
    if (response.status === 422) {
      throw new ApiError('The image could not be processed. Please try a different file.', 422);
    }
    if (response.status >= 500) {
      throw new ApiError(
        'The analysis server encountered an error. Please try again shortly.',
        response.status
      );
    }
    throw new ApiError(backendMessage || 'An unexpected error occurred.', response.status);
  }

  // Validate required fields in successful response
  _validatePredictionResponse(data);
  return data;
}

// ── Private helpers ───────────────────────────────────────────────────────────

/**
 * Validates that the backend response contains all required prediction fields.
 * Throws ApiError if the response is malformed.
 */
function _validatePredictionResponse(data) {
  // Allow gate rejections (e.g., non-leaf images) to pass through to Results page
  if (data && data.status === 'rejected') {
    return;
  }

  const required = ['status', 'predicted_class', 'confidence'];
  for (const key of required) {
    if (!(key in data)) {
      throw new ApiError(
        'The server response was incomplete. Please try again.',
        null,
        false
      );
    }
  }
  if (data.status !== 'success') {
    throw new ApiError(data.message || 'Prediction was unsuccessful.', null, false);
  }
  if (typeof data.confidence !== 'number' || data.confidence < 0 || data.confidence > 1) {
    throw new ApiError('The server returned an invalid confidence value.', null, false);
  }
}

/**
 * Returns the full URL for a Grad-CAM image path returned by the backend.
 * In development, Vite proxies /static/* → Flask so root-relative paths work as-is.
 * Only prepend API_BASE for externally absolute URLs.
 *
 * @param {string} imagePath - The path returned by the API.
 * @returns {string} Full URL safe to use as an img src.
 */
export function resolveGradCamUrl(imagePath) {
  if (!imagePath) return null;
  // Already a full URL — return as-is
  if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) return imagePath;
  // Root-relative path like /static/gradcam/...
  // In development: Vite proxy forwards /static/* → Flask (no prefix needed).
  // In production: frontend and backend are on different origins, so we must
  // prepend the backend base URL so the browser fetches from the correct server.
  return `${API_BASE}${imagePath}`;
}

// ── Prediction History ────────────────────────────────────────────────────────

/**
 * Fetches paginated prediction history from the backend.
 *
 * @param {Object} [options]
 * @param {number} [options.page=1]
 * @param {number} [options.limit=100]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{status: string, predictions: Array, pagination: Object}>}
 */
export async function fetchPredictions(options = {}) {
  const { page = 1, limit = 100, signal = null } = options;

  let response;
  try {
    response = await fetch(`${API_BASE}/api/predictions?page=${page}&limit=${limit}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError('Request was cancelled.', null, false);
    }
    throw new ApiError(
      'Unable to connect to the prediction history service. Please ensure the backend is running.',
      null,
      true
    );
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      `Server returned an unparseable response (HTTP ${response.status}).`,
      response.status
    );
  }

  if (!response.ok) {
    throw new ApiError(
      data?.message || 'Failed to retrieve prediction history.',
      response.status
    );
  }

  return data;
}

/**
 * Fetches a single prediction by its primary key ID.
 *
 * @param {number|string} id
 * @param {AbortSignal} [signal]
 * @returns {Promise<Object>} The prediction object
 */
export async function fetchPredictionById(id, signal = null) {
  if (!id) {
    throw new ApiError('Prediction ID is required.');
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/api/predictions/${id}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError('Request was cancelled.', null, false);
    }
    throw new ApiError('Could not reach server to fetch prediction details.', null, true);
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      `Server returned an unparseable response (HTTP ${response.status}).`,
      response.status
    );
  }

  if (!response.ok) {
    throw new ApiError(
      data?.message || `Failed to fetch prediction with ID ${id}.`,
      response.status
    );
  }

  return data.prediction;
}

