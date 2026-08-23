import os

import psycopg2
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )

    return connection


# =========================================================
# CREATE TABLE
# =========================================================

def create_predictions_table():

    connection = get_connection()
    cursor = connection.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            model VARCHAR(100),
            prediction VARCHAR(50),
            confidence DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

    cursor.execute(query)

    connection.commit()

    cursor.close()
    connection.close()


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(
    filename,
    model_name,
    prediction,
    confidence
):

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
        )
    )

    connection.commit()

    cursor.close()
    connection.close()


# =========================================================
# GET PREDICTION HISTORY
# =========================================================

def get_predictions():

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

    return rows