import React from 'react';
import './ErrorMessage.css';

/**
 * ErrorMessage — displays a user-friendly error alert.
 *
 * Props:
 *   message  {string}     - the error text to display
 *   title    {string}     - optional title (default "Something went wrong")
 *   onDismiss {function}  - optional callback to dismiss
 *   type     {'error'|'warning'|'info'} - defaults to 'error'
 */
export default function ErrorMessage({ message, title, onDismiss, type = 'error' }) {
  if (!message) return null;

  const icons = {
    error:   '⚠️',
    warning: '⚠',
    info:    'ℹ',
  };

  return (
    <div
      className={`error-msg error-msg--${type} fade-in`}
      role="alert"
      aria-live="polite"
    >
      <span className="error-msg__icon" aria-hidden="true">{icons[type]}</span>
      <div className="error-msg__body">
        {title && <strong className="error-msg__title">{title}</strong>}
        <p className="error-msg__text">{message}</p>
      </div>
      {onDismiss && (
        <button
          className="error-msg__dismiss"
          onClick={onDismiss}
          aria-label="Dismiss error"
        >
          ✕
        </button>
      )}
    </div>
  );
}
