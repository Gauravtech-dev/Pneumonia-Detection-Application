from pathlib import Path
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import predict_image
from backend.database import save_prediction, get_predictions
from backend.gradcam import generate_gradcam


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRADCAM_DIR = PROJECT_ROOT / "assets" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Pneumonia Detection Application API",
    description="AI-based Chest X-Ray Pneumonia Detection API",
    version="1.0.0",
)


# =========================================================
# STATIC GRAD-CAM FILES
# =========================================================

app.mount(
    "/gradcam",
    StaticFiles(directory=str(GRADCAM_DIR)),
    name="gradcam",
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
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Validate file
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

        # Read image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # Open image
        try:
            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file.",
            )

        # =================================================
        # MODEL PREDICTION
        # =================================================

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

        # =================================================
        # GRAD-CAM
        # ONLY ONE ARGUMENT
        # =================================================

        gradcam_path = generate_gradcam(image)

        gradcam_url = None

        if gradcam_path:
            gradcam_path = Path(
                str(gradcam_path)
            )

            gradcam_url = (
                f"/gradcam/{gradcam_path.name}"
            )

        # =================================================
        # DATABASE
        # =================================================

        save_prediction(
            filename=file.filename or "unknown.jpg",
            model_name="Chest ResNet18",
            prediction=str(prediction),
            confidence=confidence,
        )

        # =================================================
        # RESPONSE
        # =================================================

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "gradcam": gradcam_url,
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

        return {
            "status": "success",
            "predictions": rows,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch predictions: {str(e)}",
        )