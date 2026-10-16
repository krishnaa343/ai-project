"""
File Validation Utilities
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Handles image upload security checks:
  - MIME type validation (extension + magic bytes)
  - File size enforcement
  - Filename sanitisation
"""

import os
from pathlib import Path
from io import BytesIO

from PIL import Image as _PilImage
from werkzeug.utils import secure_filename

# ── Constants ────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"jpeg", "png", "webp"}       # imghdr returns these strings
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024             # 10 MB hard ceiling


class FileValidationError(ValueError):
    """Raised when an uploaded file fails any validation check."""


# ── Public helpers ───────────────────────────────────────────────────────────

def validate_image_upload(file_storage):
    """
    Validates a Werkzeug FileStorage object for safe image upload.

    Checks performed (in order):
      1. File is present and non-empty.
      2. Extension is in the allowed set.
      3. File size is within the configured limit.
      4. Magic-byte check confirms the content is a real image.

    Args:
        file_storage: werkzeug.datastructures.FileStorage

    Returns:
        secure_name (str): sanitised filename safe for disk use.

    Raises:
        FileValidationError: on any failed check.
    """
    if file_storage is None or file_storage.filename == "":
        raise FileValidationError("No file was provided.")

    filename = secure_filename(file_storage.filename)
    if not filename:
        raise FileValidationError("Invalid filename after sanitisation.")

    ext = _get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"File type '.{ext}' is not supported. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Read raw bytes for size + magic-byte validation
    raw_bytes = file_storage.read()
    file_storage.seek(0)           # rewind so the caller can read it again

    size_bytes = len(raw_bytes)
    if size_bytes == 0:
        raise FileValidationError("Uploaded file is empty.")
    if size_bytes > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        raise FileValidationError(
            f"File size {actual_mb:.1f} MB exceeds the {max_mb} MB limit."
        )

    try:
        img = _PilImage.open(BytesIO(raw_bytes))
        detected = (img.format or "").lower()   # e.g. "jpeg", "png", "webp"
    except Exception:
        detected = None

    _PILLOW_OK = {"jpeg", "png", "webp"}
    if detected not in _PILLOW_OK:
        raise FileValidationError(
            f"File content does not match a supported image format "
            f"(detected: {detected or 'unknown'})."
        )

    return filename


def _get_extension(filename: str) -> str:
    """Returns the lowercase file extension without the leading dot."""
    _, ext = os.path.splitext(filename)
    return ext.lstrip(".").lower()
