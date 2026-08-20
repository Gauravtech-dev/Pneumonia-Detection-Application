from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torchvision.models import resnet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================
# PROJECT PATH
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "chest_xray_resnet18.pth"
)

TEST_IMAGE_PATH = (
    PROJECT_ROOT
    / "assets"
    / "test_images"
    / "R.jpg"
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Grad-CAM Device:", device)


# =========================
# MODEL
# =========================

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# =========================
# LOAD MODEL
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

print("Grad-CAM model loaded successfully")


# =========================
# CLASSES
# =========================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


# =========================
# IMAGE TRANSFORM
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
# GRAD-CAM FUNCTION
# =========================

def generate_gradcam(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    resized_image = image.resize(
        (224, 224)
    )

    rgb_image = (
        torch.tensor(
            list(resized_image.getdata()),
            dtype=torch.float32
        )
        .reshape(224, 224, 3)
        .numpy()
        / 255.0
    )

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    target_layers = [
        model.layer4[-1]
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        output = model(
            input_tensor
        )

        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    prediction = CLASS_NAMES[
        predicted_class
    ]

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    return (
        visualization,
        prediction,
        confidence
    )


# =========================
# TEST
# =========================

if __name__ == "__main__":

    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE_PATH}"
        )

    visualization, prediction, confidence = (
        generate_gradcam(TEST_IMAGE_PATH)
    )

    output_path = (
        PROJECT_ROOT
        / "assets"
        / "gradcam_result.jpg"
    )

    Image.fromarray(
        visualization
    ).save(output_path)

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        f"Grad-CAM saved at: {output_path}"
    )