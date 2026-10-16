import React from 'react';
import { formatFileSize } from '../utils/validation';
import './ImagePreview.css';

/**
 * ImagePreview — shows the selected image with its filename/size.
 *
 * Props:
 *   file        {File}       - the selected file
 *   previewUrl  {string}     - object URL for the image
 *   onRemove    {function}   - callback to clear the selection
 *   disabled    {boolean}    - disable remove button during analysis
 */
export default function ImagePreview({ file, previewUrl, onRemove, disabled = false }) {
  if (!file || !previewUrl) return null;

  return (
    <div className="preview fade-in">
      <div className="preview__image-wrap">
        <img
          src={previewUrl}
          alt={`Preview of selected file: ${file.name}`}
          className="preview__image"
        />
        {!disabled && (
          <button
            className="preview__remove"
            onClick={onRemove}
            aria-label="Remove selected image"
            title="Remove image"
          >
            ✕
          </button>
        )}
      </div>
      <div className="preview__info">
        <span className="preview__name" title={file.name}>{file.name}</span>
        <span className="preview__size">{formatFileSize(file.size)}</span>
      </div>
    </div>
  );
}
