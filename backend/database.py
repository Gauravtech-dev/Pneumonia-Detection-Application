import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def get_connection():

    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    return connection


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
        (filename, model, prediction, confidence)
        VALUES (%s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            filename,
            model_name,
            prediction,
            confidence
        )
    )

    connection.commit()

    cursor.close()
    connection.close()


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