/**
 * Client-Side Image Validation Utilities
 * Project: AI-Based Intelligent Crop Disease Detection Framework
 *
 * Provides fast, user-friendly pre-upload validation so the user gets
 * immediate feedback without waiting for a network round-trip.
 *
 * NOTE: Server-side validation remains authoritative.
 *       These checks are for UX only.
 */

// Must match backend: jpg, jpeg, png, webp — max 10 MB
const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const ALLOWED_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'webp']);
const MAX_SIZE_BYTES = Number(import.meta.env.VITE_MAX_FILE_SIZE_BYTES) || 10 * 1024 * 1024; // 10 MB
const MAX_SIZE_MB = MAX_SIZE_BYTES / (1024 * 1024);

/**
 * Result shape: { valid: boolean, error: string | null }
 */

/**
 * Validates that a File object is a supported, non-oversized image.
 *
 * Checks:
 *  1. File exists.
 *  2. MIME type is in the allowed set.
 *  3. File extension is in the allowed set.
 *  4. File size is below the configured limit.
 *
 * @param {File|null} file
 * @returns {{ valid: boolean, error: string|null }}
 */
export function validateImageFile(file) {
  if (!file) {
    return { valid: false, error: 'Please select an image to analyse.' };
  }

  // MIME type check
  if (!ALLOWED_TYPES.has(file.type)) {
    return {
      valid: false,
      error: `Unsupported image format (${file.type || 'unknown'}). Please use JPG, PNG, or WebP.`,
    };
  }

  // Extension check (belt-and-suspenders alongside MIME)
  const ext = file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXTENSIONS.has(ext)) {
    return {
      valid: false,
      error: `File extension ".${ext}" is not supported. Please use .jpg, .jpeg, .png, or .webp.`,
    };
  }

  // Size check
  if (file.size > MAX_SIZE_BYTES) {
    const actualMB = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `Image is too large (${actualMB} MB). Maximum allowed size is ${MAX_SIZE_MB} MB.`,
    };
  }

  return { valid: true, error: null };
}

/**
 * Returns a human-readable file size string, e.g. "2.4 MB".
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Formats a raw confidence value (0–1) as a percentage string, e.g. "94.21%"
 * @param {number} confidence - value between 0 and 1
 * @returns {string}
 */
export function formatConfidence(confidence) {
  if (typeof confidence !== 'number') return '—';
  return `${(confidence * 100).toFixed(2)}%`;
}

/**
 * Converts a snake_case or triple-underscore class name to a readable label.
 * Examples:
 *   "Tomato___Early_blight"    → { crop: "Tomato", disease: "Early Blight" }
 *   "Apple___healthy"          → { crop: "Apple",  disease: "Healthy" }
 *   "Corn_(maize)___healthy"   → { crop: "Corn (maize)", disease: "Healthy" }
 *
 * @param {string} className
 * @returns {{ crop: string, disease: string, isHealthy: boolean }}
 */
export function parseClassName(className) {
  if (!className) return { crop: 'Unknown', disease: 'Unknown', isHealthy: false };

  const parts = className.split('___');
  const rawCrop    = parts[0] || 'Unknown';
  const rawDisease = parts[1] || 'Unknown';

  const crop    = rawCrop.replace(/_/g, ' ').trim();
  const disease = rawDisease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).trim();
  const isHealthy = rawDisease.toLowerCase() === 'healthy';

  return { crop, disease, isHealthy };
}
