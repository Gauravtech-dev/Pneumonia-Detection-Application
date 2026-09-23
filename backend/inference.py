from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "chest_xray_resnet18.pth"
)


# =========================================================
# CLASSES
# =========================================================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]


# =========================================================
# IMAGE TRANSFORM
# =========================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# =========================================================
# LOAD MODEL
# =========================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image: Image.Image):

    # -----------------------------------------------------
    # Prepare image
    # -----------------------------------------------------

    image = image.convert("RGB")

    tensor = TRANSFORM(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)


    # -----------------------------------------------------
    # Original prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]


    normal_probability = (
        probabilities[0].item()
    )

    pneumonia_probability = (
        probabilities[1].item()
    )


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = max(
        normal_probability,
        pneumonia_probability,
    ) * 100


    # =====================================================
    # CONSERVATIVE DECISION LOGIC
    # =====================================================

    # A prediction is accepted only when the model has
    # sufficiently strong probability for that class.
    #
    # This avoids automatically calling borderline cases
    # pneumonia.

    NORMAL_THRESHOLD = 0.80

    PNEUMONIA_THRESHOLD = 0.90


    # -----------------------------------------------------
    # NORMAL
    # -----------------------------------------------------

    if (
        normal_probability
        >= NORMAL_THRESHOLD
        and normal_probability
        > pneumonia_probability
    ):

        prediction = "NORMAL"

        supported = True

        uncertain = False

        reason = (
            "The model predicts NORMAL "
            "with sufficient confidence."
        )


    # -----------------------------------------------------
    # PNEUMONIA
    # -----------------------------------------------------

    elif (
        pneumonia_probability
        >= PNEUMONIA_THRESHOLD
        and pneumonia_probability
        > normal_probability
    ):

        prediction = "PNEUMONIA"

        supported = True

        uncertain = False

        reason = (
            "The model predicts PNEUMONIA "
            "with high confidence."
        )


    # -----------------------------------------------------
    # UNCERTAIN
    # -----------------------------------------------------

    else:

        prediction = (
            "UNCERTAIN / UNSUPPORTED"
        )

        supported = False

        uncertain = True

        reason = (
            "The model confidence is not "
            "sufficient for a reliable "
            "NORMAL or PNEUMONIA prediction."
        )


    # =====================================================
    # RETURN
    # =====================================================

    return {
        "prediction": prediction,

        "confidence": round(
            confidence,
            2,
        ),

        "normal_probability": round(
            normal_probability * 100,
            2,
        ),

        "pneumonia_probability": round(
            pneumonia_probability * 100,
            2,
        ),

        "uncertain": uncertain,

        "supported": supported,

        "threshold": {
            "normal": NORMAL_THRESHOLD * 100,
            "pneumonia": PNEUMONIA_THRESHOLD * 100,
        },

        "reason": reason,

        "consistency": {
            "original_prediction": prediction,
            "normal_probability": round(
                normal_probability * 100,
                2,
            ),
            "pneumonia_probability": round(
                pneumonia_probability * 100,
                2,
            ),
        },
    }