import io
import os
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import predict_image
from backend.database import (
    create_table,
    save_prediction,
    get_predictions,
)
from backend.gradcam import generate_gradcam


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
GRADCAM_DIR = ASSETS_DIR / "gradcam"

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# STATIC FILES
# =========================================================

app.mount(
    "/assets",
    StaticFiles(
        directory=str(ASSETS_DIR)
    ),
    name="assets",
)


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():

    try:

        create_table()

        print(
            "Database table ready."
        )

    except Exception as e:

        print(
            f"Database startup warning: {e}"
        )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Pneumonia Detection Application API",
        "status": "running",
        "model": "ResNet18",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": "Pneumonia Detection Application",
        "model": "ResNet18",
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # =====================================================
    # VALIDATE FILE
    # =====================================================

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be detected.",
        )

    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )


    # =====================================================
    # READ IMAGE
    # =====================================================

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )


    # =====================================================
    # OPEN IMAGE
    # =====================================================

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        prediction_result = predict_image(
            image
        )

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}",
        )


    # =====================================================
    # EXTRACT RESULT
    # =====================================================

    prediction = None
    confidence = None


    if isinstance(
        prediction_result,
        dict,
    ):

        prediction = (
            prediction_result.get(
                "prediction"
            )
            or prediction_result.get(
                "label"
            )
            or prediction_result.get(
                "class"
            )
        )

        confidence = (
            prediction_result.get(
                "confidence"
            )
            or prediction_result.get(
                "probability"
            )
        )


    elif isinstance(
        prediction_result,
        (tuple, list),
    ):

        if len(prediction_result) >= 1:

            prediction = (
                prediction_result[0]
            )

        if len(prediction_result) >= 2:

            confidence = (
                prediction_result[1]
            )


    else:

        prediction = prediction_result


    if prediction is None:

        raise HTTPException(
            status_code=500,
            detail="Prediction result was empty.",
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    gradcam_data = None

    try:

        gradcam_path = generate_gradcam(
            image
        )

        if gradcam_path:

            gradcam_path = Path(
                gradcam_path
            )

            gradcam_data = {
                "url": (
                    "/assets/gradcam/"
                    f"{gradcam_path.name}"
                ),
                "filename": (
                    gradcam_path.name
                ),
            }

            print(
                "Grad-CAM generated:"
                f" {gradcam_path.name}"
            )

    except Exception as e:

        print(
            f"Grad-CAM warning: {e}"
        )

        gradcam_data = None


    # =====================================================
    # DATABASE
    # =====================================================

    database_status = "failed"

    try:

        save_prediction(
            filename=(
                file.filename
                or "uploaded_image"
            ),
            model_name="ResNet18",
            prediction=str(
                prediction
            ),
            confidence=(
                float(confidence)
                if confidence is not None
                else None
            ),
        )

        database_status = "saved"

        print(
            "Prediction saved to database."
        )

    except Exception as e:

        # Database failure must NOT
        # break model prediction.

        print(
            f"Database save failed: {e}"
        )

        database_status = "failed"


    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {

        "status": "success",

        "result": {

            "prediction": str(
                prediction
            ),

            "confidence": (
                float(confidence)
                if confidence is not None
                else None
            ),
        },

        "gradcam": gradcam_data,

        "database": {

            "status": database_status
        },

        "model": "ResNet18",
    }


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

        print(
            f"Prediction history error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load prediction "
                f"history: {str(e)}"
            ),
        )


# =========================================================
# RENDER STARTUP
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
    )