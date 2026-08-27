from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.inference import model, DEVICE, CLASS_NAMES, TRANSFORM


BASE_DIR = Path(__file__).resolve().parent.parent
GRADCAM_DIR = BASE_DIR / "assets" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def generate_gradcam(image: Image.Image):
    image = image.convert("RGB")
    original = np.asarray(image)

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_index].item() * 100

        output[0, predicted_index].backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM activations/gradients were not captured.")

        activation = activations[0]
        gradient = gradients[0]

        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()

        maximum = cam.max()
        if maximum > 0:
            cam /= maximum

        height, width = original.shape[:2]
        cam = cv2.resize(cam, (width, height))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET,
        )

        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(
            original_bgr,
            0.55,
            heatmap,
            0.45,
            0,
        )

        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        output_path = GRADCAM_DIR / filename

        if not cv2.imwrite(str(output_path), overlay):
            raise RuntimeError("Failed to save Grad-CAM image.")

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 2),
            "url": f"/assets/gradcam/{filename}",
            "filename": filename,
            "path": str(output_path),
        }

    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)
