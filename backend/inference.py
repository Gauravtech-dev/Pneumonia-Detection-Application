from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# =========================
# PROJECT PATH
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "chest_xray_resnet18.pth"
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("CuraVision AI - Device:", device)


# =========================
# MODEL
# =========================

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# =========================
# LOAD TRAINED MODEL
# =========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Chest ResNet18 loaded successfully")


# =========================
# CLASSES
# =========================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


# =========================
# PREPROCESSING
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# PREDICTION
# =========================

def predict_image(image: Image.Image):

    image = image.convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_index
        ].item()

    prediction = CLASS_NAMES[
        predicted_index
    ]

    return {
        "prediction": prediction,
        "confidence": round(
            confidence * 100,
            2
        )
    }