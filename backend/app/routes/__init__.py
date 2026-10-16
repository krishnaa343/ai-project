from .health import health_bp
from .prediction import prediction_bp
from .history import history_bp


def register_routes(app):
    """Registers API blueprints with the Flask application instance."""
    app.register_blueprint(health_bp,     url_prefix="/api")
    app.register_blueprint(prediction_bp, url_prefix="/api")
    app.register_blueprint(history_bp,    url_prefix="/api")
