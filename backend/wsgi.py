"""
WSGI Entry Point — Production Deployment
Project: AI-Based Intelligent Crop Disease Detection Framework

Used by Gunicorn (Render) to serve the Flask application in production.
Development uses run.py directly.

Usage:
    gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 wsgi:app
"""
import os
from app import create_app

# Always use 'production' when running via WSGI
app = create_app(os.getenv('FLASK_ENV', 'production'))
