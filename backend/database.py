import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace(
                "postgres://",
                "postgresql://",
                1,
            )

        if "sslmode=" not in database_url:
            separator = "&" if "?" in database_url else "?"
            database_url += f"{separator}sslmode=require"

        return psycopg2.connect(
            database_url,
            connect_timeout=10,
        )

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )


def create_table():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255),
                    model VARCHAR(100) NOT NULL,
                    prediction VARCHAR(100) NOT NULL,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_prediction(filename, model_name, prediction, confidence):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO predictions
                    (filename, model, prediction, confidence)
                VALUES (%s, %s, %s, %s)
            """, (filename, model_name, prediction, confidence))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_predictions():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    id,
                    filename,
                    model,
                    prediction,
                    confidence,
                    created_at
                FROM predictions
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "model": row[2],
                "prediction": row[3],
                "confidence": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]
    finally:
        connection.close()
