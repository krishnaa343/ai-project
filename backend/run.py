import os
from app import create_app

app = create_app(os.getenv("FLASK_ENV", "development"))

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    # In production (FLASK_ENV=production), bind to 0.0.0.0 for cloud hosting.
    # In development, bind to 127.0.0.1 for security.
    env = os.getenv("FLASK_ENV", "development")
    host = "0.0.0.0" if env == "production" else "127.0.0.1"
    app.run(host=host, port=port, debug=debug)

