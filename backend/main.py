import io
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.gradcam import generate_gradcam
from backend.inference import predict_image


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
GRADCAM_DIR = ASSETS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ResNet18"
CONFIDENCE_THRESHOLD = 65.0
MIN_WIDTH = 160
MIN_HEIGHT = 160
MIN_FILE_BYTES = 8 * 1024


app = FastAPI(
    title="Pneumonia Detection Application API",
    description="AI-assisted chest X-ray pneumonia detection API",
    version="1.0.0",
)


app.mount(
    "/assets",
    StaticFiles(directory=str(ASSETS_DIR)),
    name="assets",
)


@app.get("/")
def root():
    return {
        "message": "Pneumonia Detection Application API",
        "status": "running",
        "model": MODEL_NAME,
        "classes": ["NORMAL", "PNEUMONIA"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "Pneumonia Detection Application",
        "model": MODEL_NAME,
    }


def basic_image_validation(image: Image.Image, file_size: int):
    if file_size < MIN_FILE_BYTES:
        return False, "The uploaded file is too small."

    width, height = image.size

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, "Image resolution is too small."

    if image.getbbox() is None:
        return False, "The uploaded image appears blank."

    gray = np.asarray(image.convert("L"))

    if gray.size == 0:
        return False, "The image could not be read."

    if float(gray.std()) < 8.0:
        return False, "The image appears blank or unusable."

    return True, ""


def xray_like_check(image: Image.Image):
    """
    Lightweight heuristic gate.
    This is NOT a medical X-ray-vs-non-X-ray ML classifier.
    """

    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if gray.ndim != 2 or gray.size == 0:
        return False, 0.0, "Unsupported image structure."

    height, width = gray.shape

    if min(height, width) < 224:
        return False, 0.0, "Image resolution is too small."

    aspect = width / max(height, 1)

    if not 0.45 <= aspect <= 2.20:
        return False, 0.0, "Image proportions are unsuitable."

    std = float(gray.std())

    if std < 12:
        return False, 0.0, "The image appears blank or unusable."

    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p05)

    if dynamic_range < 35:
        return False, 0.0, "The image does not contain enough visual structure."

    edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
    edge_density = float(np.mean(edges > 0))

    if edge_density < 0.004:
        return False, 0.0, "The image does not contain enough radiographic structure."

    midtone_fraction = float(
        np.mean((gray >= 20) & (gray <= 240))
    )

    if midtone_fraction < 0.40:
        return False, 0.0, "The image does not resemble a usable radiograph."

    score = 0.0

    if 0.55 <= aspect <= 1.90:
        score += 0.20

    if 18 <= std <= 100:
        score += 0.20

    if dynamic_range >= 60:
        score += 0.20

    if 0.008 <= edge_density <= 0.25:
        score += 0.20

    if midtone_fraction >= 0.55:
        score += 0.20

    if score < 0.50:
        return (
            False,
            score,
            "The image does not look sufficiently like a chest X-ray.",
        )

    return True, score, ""


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        )

    valid, message = basic_image_validation(
        image,
        len(image_bytes),
    )

    if not valid:
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    xray_ok, xray_score, xray_reason = xray_like_check(image)

    if not xray_ok:
        raise HTTPException(
            status_code=400,
            detail=f"Please upload a chest X-ray. {xray_reason}",
        )

    try:
        result = predict_image(image)

    except Exception as exc:
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        )

    prediction = result["prediction"]
    confidence = float(result["confidence"])

    uncertain = confidence < CONFIDENCE_THRESHOLD

    gradcam = None

    try:
        gradcam = generate_gradcam(image)

    except Exception as exc:
        print(f"Grad-CAM warning: {exc}")

    return {
        "status": "success",

        "input": {
            "filename": file.filename or "uploaded_image",
            "type": "chest_xray",

            "xray_validation": {
                "accepted": True,
                "score": round(xray_score, 2),
            },
        },

        "result": {
            "prediction": prediction,
            "confidence": confidence,
            "uncertain": uncertain,
            "threshold": CONFIDENCE_THRESHOLD,
        },

        "gradcam": gradcam,

        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
    )
