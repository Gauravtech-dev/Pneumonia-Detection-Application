# backend/inference.py

from pathlib import Path

import numpy as np
import onnxruntime as ort

from PIL import Image


# =========================================================
# PATH
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "chest_xray_resnet18.onnx"
)


# =========================================================
# CLASSES
# =========================================================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]


# =========================================================
# ONNX RUNTIME
# =========================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

INPUT_NAME = (
    session.get_inputs()[0].name
)


# =========================================================
# PREPROCESS
# =========================================================

def preprocess_image(
    image: Image.Image
):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = (
        image_array / 255.0
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=np.float32,
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=np.float32,
    )

    image_array = (
        image_array - mean
    ) / std

    image_array = np.transpose(
        image_array,
        (2, 0, 1),
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array.astype(
        np.float32
    )


# =========================================================
# SOFTMAX
# =========================================================

def softmax(values):

    values = (
        values
        - np.max(values)
    )

    exp_values = np.exp(values)

    return (
        exp_values
        / np.sum(exp_values)
    )


# =========================================================
# PREDICTION
# =========================================================

def predict_image(
    image: Image.Image
):

    input_tensor = (
        preprocess_image(image)
    )

    outputs = session.run(
        None,
        {
            INPUT_NAME: input_tensor
        },
    )

    logits = outputs[0][0]

    probabilities = softmax(
        logits
    )

    normal_probability = float(
        probabilities[0]
    )

    pneumonia_probability = float(
        probabilities[1]
    )


    # =====================================================
    # DECISION
    # =====================================================

    NORMAL_THRESHOLD = 0.80

    PNEUMONIA_THRESHOLD = 0.90


    if (
        normal_probability
        >= NORMAL_THRESHOLD
        and normal_probability
        > pneumonia_probability
    ):

        prediction = "NORMAL"

        confidence = (
            normal_probability * 100
        )

        supported = True

        uncertain = False

        reason = (
            "The model predicts NORMAL "
            "with sufficient confidence."
        )


    elif (
        pneumonia_probability
        >= PNEUMONIA_THRESHOLD
        and pneumonia_probability
        > normal_probability
    ):

        prediction = "PNEUMONIA"

        confidence = (
            pneumonia_probability * 100
        )

        supported = True

        uncertain = False

        reason = (
            "The model predicts PNEUMONIA "
            "with high confidence."
        )


    else:

        prediction = (
            "UNCERTAIN / UNSUPPORTED"
        )

        confidence = (
            max(
                normal_probability,
                pneumonia_probability,
            )
            * 100
        )

        supported = False

        uncertain = True

        reason = (
            "The model confidence is not "
            "sufficient for a reliable prediction."
        )


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
            "normal": 80.0,
            "pneumonia": 90.0,
        },

        "reason": reason,

        "consistency": {
            "original_prediction": prediction,
        },
    }