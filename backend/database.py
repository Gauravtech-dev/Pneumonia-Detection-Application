import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def get_connection():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is missing."
        )

    connection = psycopg2.connect(
        database_url,
        connect_timeout=10,
    )

    return connection


def create_predictions_table():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            model VARCHAR(100),
            prediction VARCHAR(50),
            confidence DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    cursor.close()
    connection.close()


def save_prediction(
    filename,
    model_name,
    prediction,
    confidence
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (filename, model, prediction, confidence)
        VALUES (%s, %s, %s, %s)
    """, (
        filename,
        model_name,
        prediction,
        confidence,
    ))

    connection.commit()

    cursor.close()
    connection.close()


def get_predictions():

    connection = get_connection()
    cursor = connection.cursor()

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

    cursor.close()
    connection.close()

    return rows