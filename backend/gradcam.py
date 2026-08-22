from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "chest_xray_resnet18.pth"
)

GRADCAM_DIR = (
    BASE_DIR
    / "assets"
    / "gradcam"
)

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Grad-CAM Device: {DEVICE}")


# =========================================================
# RESNET18 MODEL
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    2
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]


clean_checkpoint = {}

for key, value in checkpoint.items():

    new_key = key.replace(
        "module.",
        ""
    )

    clean_checkpoint[new_key] = value


model.load_state_dict(
    clean_checkpoint,
    strict=False
)

model.to(DEVICE)
model.eval()

print("Grad-CAM model loaded successfully")


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(image: Image.Image):

    if not isinstance(
        image,
        Image.Image
    ):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    original = np.array(image)

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )


    # -----------------------------------------------------
    # Target layer
    # -----------------------------------------------------

    target_layer = model.layer4[-1]


    activations = []
    gradients = []


    # -----------------------------------------------------
    # Forward hook
    # -----------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(output)


    # -----------------------------------------------------
    # Backward hook
    # -----------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        output = model(
            input_tensor
        )


        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()


        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        model.zero_grad()

        score = output[
            :,
            predicted_class
        ]

        score.backward()


        # -------------------------------------------------
        # Activations + gradients
        # -------------------------------------------------

        activation = activations[0]

        gradient = gradients[0]


        # -------------------------------------------------
        # Calculate weights
        # -------------------------------------------------

        weights = torch.mean(
            gradient,
            dim=(2, 3),
            keepdim=True
        )


        # -------------------------------------------------
        # CAM
        # -------------------------------------------------

        cam = torch.sum(
            weights * activation,
            dim=1
        )

        cam = F.relu(
            cam
        )


        cam = (
            cam
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )


        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        cam -= cam.min()

        if cam.max() != 0:

            cam /= cam.max()


        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        height, width = (
            original.shape[:2]
        )

        cam = cv2.resize(
            cam,
            (width, height)
        )


        # -------------------------------------------------
        # Heatmap
        # -------------------------------------------------

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        # -------------------------------------------------
        # Original → BGR
        # -------------------------------------------------

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )


        # -------------------------------------------------
        # Overlay
        # -------------------------------------------------

        overlay = cv2.addWeighted(
            original_bgr,
            0.55,
            heatmap,
            0.45,
            0
        )


        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        filename = (
            f"gradcam_"
            f"{uuid.uuid4().hex}.jpg"
        )

        output_path = (
            GRADCAM_DIR
            / filename
        )


        cv2.imwrite(
            str(output_path),
            overlay
        )


        return output_path


    finally:

        forward_handle.remove()

        backward_handle.remove()
