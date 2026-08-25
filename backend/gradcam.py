from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =========================================================
# PROJECT PATHS
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
# LOAD RESNET18
# =========================================================

gradcam_model = models.resnet18(
    weights=None
)

gradcam_model.fc = torch.nn.Linear(
    gradcam_model.fc.in_features,
    2
)


# =========================================================
# LOAD TRAINED CHECKPOINT
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


gradcam_model.load_state_dict(
    clean_checkpoint,
    strict=False
)

gradcam_model.to(DEVICE)

gradcam_model.eval()

print(
    "Grad-CAM model loaded successfully"
)


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

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
# GENERATE GRAD-CAM
# =========================================================

def generate_gradcam(
    image: Image.Image
):

    if not isinstance(
        image,
        Image.Image
    ):
        image = Image.fromarray(
            image
        )

    image = image.convert("RGB")

    original = np.array(
        image
    )


    # =====================================================
    # PREPARE INPUT
    # =====================================================

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )


    # =====================================================
    # TARGET LAYER
    # =====================================================

    target_layer = (
        gradcam_model.layer4[-1]
    )


    activations = []
    gradients = []


    # =====================================================
    # FORWARD HOOK
    # =====================================================

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output
        )


    # =====================================================
    # BACKWARD HOOK
    # =====================================================

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

        # =================================================
        # FORWARD PASS
        # =================================================

        output = gradcam_model(
            input_tensor
        )


        predicted_class = (
            torch.argmax(
                output,
                dim=1
            ).item()
        )


        # =================================================
        # BACKWARD PASS
        # =================================================

        gradcam_model.zero_grad()

        score = output[
            0,
            predicted_class
        ]

        score.backward()


        # =================================================
        # CHECK HOOK RESULTS
        # =================================================

        if not activations:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )


        activation = activations[0]

        gradient = gradients[0]


        # =================================================
        # GLOBAL AVERAGE POOLING
        # =================================================

        weights = torch.mean(
            gradient,
            dim=(2, 3),
            keepdim=True
        )


        # =================================================
        # CAM
        # =================================================

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


        # =================================================
        # NORMALIZE CAM
        # =================================================

        cam -= cam.min()

        cam_max = cam.max()

        if cam_max > 0:

            cam /= cam_max


        # =================================================
        # RESIZE CAM
        # =================================================

        height, width = (
            original.shape[:2]
        )

        cam = cv2.resize(
            cam,
            (width, height)
        )


        # =================================================
        # CREATE HEATMAP
        # =================================================

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        # =================================================
        # RGB → BGR
        # =================================================

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )


        # =================================================
        # OVERLAY
        # =================================================

        overlay = cv2.addWeighted(
            original_bgr,
            0.55,
            heatmap,
            0.45,
            0
        )


        # =================================================
        # SAVE FILE
        # =================================================

        filename = (
            f"gradcam_"
            f"{uuid.uuid4().hex}.jpg"
        )

        output_path = (
            GRADCAM_DIR
            / filename
        )


        success = cv2.imwrite(
            str(output_path),
            overlay
        )


        if not success:

            raise RuntimeError(
                "Failed to save Grad-CAM image."
            )


        return output_path


    finally:

        forward_handle.remove()

        backward_handle.remove()