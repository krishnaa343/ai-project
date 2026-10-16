import React, { useCallback, useRef, useState } from 'react';
import { validateImageFile, formatFileSize } from '../utils/validation';
import ErrorMessage from './ErrorMessage';
import './ImageUploader.css';

/**
 * ImageUploader — drag-and-drop + file-picker image selection component.
 *
 * Props:
 *   onFileSelected   {(file: File) => void}
 *   disabled         {boolean}
 */
export default function ImageUploader({ onFileSelected, onFileSelect, onError, disabled = false }) {
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const inputRef = useRef(null);

  const notifySelected = onFileSelect || onFileSelected;

  const handleFile = useCallback((file) => {
    setValidationError(null);
    const { valid, error } = validateImageFile(file);
    if (!valid) {
      setValidationError(error);
      if (onError) onError(error);
      return;
    }
    if (notifySelected) {
      notifySelected(file);
    }
  }, [notifySelected, onError]);

  // Drag events
  const onDragEnter = (e) => { e.preventDefault(); e.stopPropagation(); if (!disabled) setDragActive(true); };
  const onDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); };
  const onDragOver  = (e) => { e.preventDefault(); e.stopPropagation(); };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  // File input change
  const onInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset so same file can be re-selected after removal
    e.target.value = '';
  };

  const openPicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  return (
    <div className="uploader">
      {/* Hidden native file input */}
      <input
        ref={inputRef}
        id="image-file-input"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        aria-label="Select a crop leaf image"
        className="sr-only"
        onChange={onInputChange}
        disabled={disabled}
      />

      {/* Drop zone */}
      <div
        className={`uploader__zone
          ${dragActive ? 'uploader__zone--active' : ''}
          ${disabled ? 'uploader__zone--disabled' : ''}`}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onClick={openPicker}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Click or drag a crop leaf image here to upload"
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') openPicker(); }}
      >
        <div className="uploader__icon" aria-hidden="true">
          {dragActive ? '📂' : '🍃'}
        </div>
        <p className="uploader__headline">
          {dragActive ? 'Drop your image here' : 'Drop a leaf image here'}
        </p>
        <p className="uploader__sub">
          or <span className="uploader__link">click to browse</span>
        </p>
        <div className="uploader__formats">
          <span className="badge badge-primary">JPG</span>
          <span className="badge badge-primary">PNG</span>
          <span className="badge badge-primary">WebP</span>
          <span className="badge badge-primary">Max 10 MB</span>
        </div>
      </div>

      {/* Validation error */}
      {validationError && (
        <ErrorMessage
          message={validationError}
          onDismiss={() => setValidationError(null)}
        />
      )}
    </div>
  );
}
