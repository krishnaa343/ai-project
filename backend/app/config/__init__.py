import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class BaseConfig:
    """Base application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-in-prod")
    DEBUG = False
    TESTING = False
    
    # CORS Configuration - restricted to specified frontend origins
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    
    # Future MySQL connection parameters (prepared for later phases)
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "crop_disease_detection")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")

class DevelopmentConfig(BaseConfig):
    """Development environment specific settings."""
    DEBUG = True

class ProductionConfig(BaseConfig):
    """Production environment specific settings."""
    DEBUG = False

config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}
