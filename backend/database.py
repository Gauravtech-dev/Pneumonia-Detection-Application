import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    """
    Create PostgreSQL connection.

    Supports:
    1. DATABASE_URL - Render PostgreSQL
    2. DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD - local setup
    """

    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return psycopg2.connect(database_url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def create_table():
    """
    Create predictions table if it does not already exist.
    """

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            model VARCHAR(100),
            prediction VARCHAR(100),
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """

    cursor.execute(query)

    connection.commit()

    cursor.close()
    connection.close()


def save_prediction(
    filename,
    model_name,
    prediction,
    confidence,
):
    """
    Save one prediction into PostgreSQL.
    """

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO predictions
        (
            filename,
            model,
            prediction,
            confidence
        )
        VALUES (%s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            filename,
            model_name,
            prediction,
            confidence,
        ),
    )

    connection.commit()

    cursor.close()
    connection.close()


def get_predictions():
    """
    Return prediction history.
    """

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        SELECT
            id,
            filename,
            model,
            prediction,
            confidence,
            created_at
        FROM predictions
        ORDER BY created_at DESC
    """

    cursor.execute(query)

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    predictions = []

    for row in rows:
        predictions.append(
            {
                "id": row[0],
                "filename": row[1],
                "model": row[2],
                "prediction": row[3],
                "confidence": row[4],
                "created_at": (
                    row[5].isoformat()
                    if row[5]
                    else None
                ),
            }
        )

    return predictions