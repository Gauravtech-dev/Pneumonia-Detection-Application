from pathlib import Path
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from inference import predict_image
from database import save_prediction, get_predictions
from gradcam import generate_gradcam


# =========================
# PROJECT PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRADCAM_DIR = (
    PROJECT_ROOT
    / "assets"
    / "gradcam"
)

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="CuraVision AI",
    description="Chest X-Ray AI Prediction API",
    version="1.0.0"
)


# =========================
# STATIC FILES
# =========================

app.mount(
    "/gradcam",
    StaticFiles(directory=str(GRADCAM_DIR)),
    name="gradcam"
)


# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {
        "message": "CuraVision AI API is running",
        "status": "online"
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not file.content_type or not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:

        # Read uploaded file
        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Convert to PIL image
        try:

            image = Image.open(
                io.BytesIO(contents)
            )

            image.load()

        except UnidentifiedImageError:

            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image."
            )

        # =========================
        # AI PREDICTION
        # =========================

        result = predict_image(image)

        # =========================
        # SAVE PREDICTION
        # =========================

        save_prediction(
            filename=file.filename,
            model_name="ResNet18",
            prediction=result["prediction"],
            confidence=result["confidence"]
        )

        # =========================
        # GRAD-CAM
        # =========================

        # Temporary image for Grad-CAM
        temp_image_path = (
            GRADCAM_DIR
            / f"input_{file.filename}"
        )

        image.save(
            temp_image_path
        )

        visualization, _, _ = generate_gradcam(
            temp_image_path
        )

        # Save Grad-CAM result
        gradcam_filename = (
            f"gradcam_{Path(file.filename).stem}.jpg"
        )

        gradcam_path = (
            GRADCAM_DIR
            / gradcam_filename
        )

        Image.fromarray(
            visualization
        ).save(
            gradcam_path
        )

        # Remove temporary input image
        if temp_image_path.exists():
            temp_image_path.unlink()

        # =========================
        # RESPONSE
        # =========================

        return {

            "success": True,

            "filename": file.filename,

            "model": "ResNet18",

            "result": result,

            "gradcam": {
                "filename": gradcam_filename,
                "url": f"/gradcam/{gradcam_filename}"
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# =========================
# PREDICTION HISTORY
# =========================

@app.get("/predictions")
def predictions_history():

    try:

        rows = get_predictions()

        return {

            "success": True,

            "count": len(rows),

            "predictions": [

                {
                    "id": row[0],
                    "filename": row[1],
                    "model": row[2],
                    "prediction": row[3],
                    "confidence": float(row[4]),
                    "created_at": row[5]
                }

                for row in rows

            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve prediction history: {str(e)}"
        )