# backend/main.py

from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.inference import predict_image


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Pneumonia Detection Application API",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Pneumonia Detection Application API",
        "status": "running",
        "model": "ResNet18",
        "classes": [
            "NORMAL",
            "PNEUMONIA",
        ],
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "ResNet18",
    }


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image(image: Image.Image):

    if image.width < 100 or image.height < 100:

        return False

    if image.width > 5000 or image.height > 5000:

        return False

    return True


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # File type
    # -----------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
    }

    if file.content_type not in allowed_types:

        return {
            "prediction": "UNCERTAIN / UNSUPPORTED",
            "confidence": 0,
            "uncertain": True,
            "supported": False,
            "reason": (
                "Please upload a JPG, JPEG "
                "or PNG image."
            ),
            "gradcam": None,
            "gradcam_url": None,
        }


    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    try:

        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        )

        image.load()

        image = image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):

        return {
            "prediction": "UNCERTAIN / UNSUPPORTED",
            "confidence": 0,
            "uncertain": True,
            "supported": False,
            "reason": (
                "The uploaded file is not "
                "a valid image."
            ),
            "gradcam": None,
            "gradcam_url": None,
        }


    # -----------------------------------------------------
    # Basic image validation
    # -----------------------------------------------------

    if not validate_image(image):

        return {
            "prediction": "UNCERTAIN / UNSUPPORTED",
            "confidence": 0,
            "uncertain": True,
            "supported": False,
            "reason": (
                "Image resolution is too small "
                "or too large."
            ),
            "gradcam": None,
            "gradcam_url": None,
        }


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    try:

        result = predict_image(
            image
        )

    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        return {
            "prediction": "UNCERTAIN / UNSUPPORTED",
            "confidence": 0,
            "uncertain": True,
            "supported": False,
            "reason": (
                "The prediction service "
                "encountered an error."
            ),
            "gradcam": None,
            "gradcam_url": None,
        }


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "prediction": result.get(
            "prediction",
            "UNCERTAIN / UNSUPPORTED",
        ),

        "confidence": result.get(
            "confidence",
            0,
        ),

        "normal_probability": result.get(
            "normal_probability",
            0,
        ),

        "pneumonia_probability": result.get(
            "pneumonia_probability",
            0,
        ),

        "uncertain": result.get(
            "uncertain",
            False,
        ),

        "supported": result.get(
            "supported",
            True,
        ),

        "threshold": result.get(
            "threshold",
            {},
        ),

        "reason": result.get(
            "reason",
            "",
        ),

        "consistency": result.get(
            "consistency",
            {},
        ),

        # Grad-CAM disabled on Render
        "gradcam": None,
        "gradcam_url": None,
    }