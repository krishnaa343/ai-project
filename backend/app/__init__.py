import os
import sys
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from .config import config_by_name
from .routes import register_routes

# Ensure the repo root is on sys.path so ml.* imports resolve from the backend
# __init__.py is at: <repo>/backend/app/__init__.py
# parents[0]=app, parents[1]=backend, parents[2]=repo_root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def create_app(config_name=None):
    """
    Application factory for the Flask backend.
    Initializes configuration, CORS, registers API blueprints,
    and eagerly preloads the Keras model to avoid cold-start latency.
    """
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    app = Flask(__name__, static_folder="../static", static_url_path="/static")

    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Enable CORS strictly for specified frontend origins (API and static assets)
    allowed = app.config.get("ALLOWED_ORIGINS", ["http://localhost:5173"])
    CORS(app, resources={
        r"/api/*": {"origins": allowed},
        r"/static/*": {"origins": allowed},
    })

    # Register routes
    register_routes(app)

    # Eagerly preload both models so the first request is not penalised
    with app.app_context():
        try:
            from .services.model_service import load_model
            load_model()
            app.logger.info("Disease classification model preloaded successfully.")
        except Exception as exc:
            app.logger.warning(f"Disease model preload failed: {exc}")

        try:
            from .services.leaf_detector_service import load_leaf_detector
            load_leaf_detector()
            app.logger.info("Leaf / Non-Leaf detector gate preloaded successfully.")
        except Exception as exc:
            app.logger.warning(f"Leaf gate preload failed: {exc}")

    return app
