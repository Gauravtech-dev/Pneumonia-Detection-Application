from pathlib import Path
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from backend.inference import predict_image


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Pneumonia Detection Application API",
    description="AI-based Chest X-Ray Pneumonia Detection API",
    version="1.0.0",
)


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
# TEMPORARILY: PREDICTION ONLY
# Grad-CAM and PostgreSQL disabled for testing
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
        # RESPONSE
        # -------------------------------------------------

        return {
            "status": "success",
            "prediction": str(prediction),
            "confidence": confidence,
            "gradcam": None,
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
# TEMPORARILY DISABLED
# =========================================================

@app.get("/predictions")
def predictions():

    return {
        "status": "success",
        "predictions": [],
        "message": "Prediction history temporarily disabled.",
    }