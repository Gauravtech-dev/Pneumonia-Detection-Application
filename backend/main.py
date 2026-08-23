import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from backend.inference import predict_image
from backend.database import (
    create_predictions_table,
    save_prediction,
    get_predictions,
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Pneumonia Detection Application API",
    description="AI-based Chest X-Ray Pneumonia Detection API",
    version="1.0.0",
)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

@app.on_event("startup")
def startup():

    try:
        create_predictions_table()
        print("PostgreSQL predictions table ready.")

    except Exception as e:
        print(f"Database initialization failed: {e}")


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Pneumonia Detection Application API",
        "status": "running",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": "Pneumonia Detection Application",
    }


# =========================================================
# PREDICT
# RESNET18 + POSTGRESQL
# GRAD-CAM TEMPORARILY DISABLED
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # -------------------------------------------------
        # Validate file
        # -------------------------------------------------

        if not file.content_type:

            raise HTTPException(
                status_code=400,
                detail="File type could not be detected.",
            )

        if not file.content_type.startswith("image/"):

            raise HTTPException(
                status_code=400,
                detail="Please upload a valid image file.",
            )

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except UnidentifiedImageError:

            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file.",
            )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction_result = predict_image(image)

        prediction = None
        confidence = None

        if isinstance(prediction_result, dict):

            prediction = (
                prediction_result.get("prediction")
                or prediction_result.get("label")
                or prediction_result.get("class")
            )

            confidence = (
                prediction_result.get("confidence")
                or prediction_result.get("probability")
            )

        elif isinstance(
            prediction_result,
            (tuple, list)
        ):

            if len(prediction_result) >= 1:

                prediction = prediction_result[0]

            if len(prediction_result) >= 2:

                confidence = prediction_result[1]

        else:

            prediction = prediction_result

        if prediction is None:

            raise RuntimeError(
                "Prediction result was empty."
            )

        # -------------------------------------------------
        # SAVE TO POSTGRESQL
        # -------------------------------------------------

        try:

            save_prediction(
                filename=file.filename or "unknown.jpg",
                model_name="Chest ResNet18",
                prediction=str(prediction),
                confidence=confidence,
            )

            database_status = "saved"

        except Exception as database_error:

            print(
                f"Database save failed: {database_error}"
            )

            database_status = "failed"

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {
            "status": "success",
            "prediction": str(prediction),
            "confidence": confidence,
            "gradcam": None,
            "database": database_status,
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.get("/predictions")
def predictions():

    try:

        rows = get_predictions()

        # Convert PostgreSQL rows into JSON-compatible objects
        formatted_rows = []

        for row in rows:

            formatted_rows.append({
                "id": row[0],
                "filename": row[1],
                "model": row[2],
                "prediction": row[3],
                "confidence": row[4],
                "created_at": str(row[5]),
            })

        return {
            "status": "success",
            "predictions": formatted_rows,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch predictions: {str(e)}",
        )