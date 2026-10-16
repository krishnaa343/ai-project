"""
Database Initialization Script
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Idempotently initializes the MySQL database and the prediction_history table.
Safe to run repeatedly — will never delete, drop, or duplicate existing data.
"""

import os
import sys
from pathlib import Path
import pymysql
from dotenv import load_dotenv

# Ensure backend root is on sys.path and load environment variables
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Load backend/.env first, then fallback to repo root .env
env_path = _BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv(dotenv_path=_REPO_ROOT / ".env")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "crop_disease_detection")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")

TABLE_NAME = "prediction_history"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NULL,
    crop VARCHAR(100) NOT NULL,
    predicted_class VARCHAR(100) NOT NULL,
    display_name VARCHAR(150) NOT NULL,
    confidence FLOAT NOT NULL,
    class_index INT NOT NULL,
    is_healthy BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at DESC),
    INDEX idx_crop (crop),
    INDEX idx_predicted_class (predicted_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


def init_database():
    """Initializes the database and table idempotently."""
    print(f"Connecting to MySQL server at {MYSQL_HOST}:{MYSQL_PORT} as user '{MYSQL_USER}'...")
    try:
        # Step 1: Connect to server without database selected to ensure DB exists
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4",
            autocommit=True
        )
    except Exception as exc:
        print(f"ERROR: Could not connect to MySQL server: {exc}", file=sys.stderr)
        return False

    try:
        with conn.cursor() as cursor:
            # Step 2: Create database if not exists
            print(f"Ensuring database '{MYSQL_DATABASE}' exists...")
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            print(f"Database '{MYSQL_DATABASE}' ready.")

            # Step 3: Switch to database
            cursor.execute(f"USE `{MYSQL_DATABASE}`;")

            # Step 4: Create table if not exists
            print(f"Ensuring table '{TABLE_NAME}' exists...")
            cursor.execute(CREATE_TABLE_SQL)
            print(f"Table '{TABLE_NAME}' ready.")

            # Step 5: Verify table schema
            cursor.execute(f"DESCRIBE `{TABLE_NAME}`;")
            columns = cursor.fetchall()
            print("\nTable schema verification:")
            for col in columns:
                col_name, col_type, nullable, key, default, extra = col[:6]
                print(f" - {col_name:18} {col_type:20} (Key: {key or '-'}, Null: {nullable}, Extra: {extra})")

            # Step 6: Count existing records
            cursor.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`;")
            count = cursor.fetchone()[0]
            print(f"\nExisting records in '{TABLE_NAME}': {count}")

        print("\nDatabase initialization complete and verified.")
        return True
    except Exception as exc:
        print(f"ERROR during schema initialization: {exc}", file=sys.stderr)
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
