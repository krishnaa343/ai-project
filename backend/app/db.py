"""
Database Connection Management
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Provides thread-safe connection handling to MySQL using PyMySQL with proper
cleanup, parameterized queries, and environment-driven configuration.
"""

import os
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from flask import current_app


def get_db_config():
    """Extracts MySQL configuration from Flask current_app or environment variables."""
    if current_app:
        return {
            "host": current_app.config.get("MYSQL_HOST", "localhost"),
            "port": int(current_app.config.get("MYSQL_PORT", 3306)),
            "user": current_app.config.get("MYSQL_USER", "root"),
            "password": current_app.config.get("MYSQL_PASSWORD", ""),
            "database": current_app.config.get("MYSQL_DATABASE", "crop_disease_detection"),
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "connect_timeout": 5
        }
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "crop_disease_detection"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "connect_timeout": 5
    }


@contextmanager
def get_db_connection():
    """
    Context manager yielding a MySQL connection.
    Ensures connection is committed on success, rolled back on error, and closed.
    """
    config = get_db_config()
    conn = pymysql.connect(**config)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_db_connection():
    """Quick connectivity test returning (is_connected: bool, error_message: str | None)."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                return True, None
    except Exception as exc:
        return False, str(exc)
