
from pathlib import Path
import uuid

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="PneumoVision | Pneumonia Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "chest_xray_resnet18.pth"
GRADCAM_DIR = PROJECT_ROOT / "assets" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Confidence below this is treated as uncertain.
CONFIDENCE_THRESHOLD = 65.0

# Basic input-quality limits.
MIN_WIDTH = 160
MIN_HEIGHT = 160
MIN_FILE_BYTES = 8 * 1024


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --pv-border: rgba(127,127,127,.20);
        --pv-muted: rgba(127,127,127,.72);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.1rem;
        border: 1px solid var(--pv-border);
        border-radius: 26px;
        background:
            radial-gradient(circle at 85% 15%, rgba(80,130,255,.20), transparent 35%),
            linear-gradient(135deg, #111827, #1f2937 58%, #26364b);
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        font-size: .74rem;
        letter-spacing: .18em;
        text-transform: uppercase;
        opacity: .70;
        font-weight: 700;
    }

    .hero-title {
        margin: .45rem 0 .7rem;
        font-size: clamp(2.1rem, 5vw, 4rem);
        line-height: .98;
        font-weight: 850;
    }

    .hero-copy {
        max-width: 780px;
        line-height: 1.65;
        opacity: .78;
        font-size: 1rem;
    }

    .badge {
        display: inline-block;
        margin-top: 1rem;
        padding: .42rem .72rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.15);
        background: rgba(255,255,255,.07);
        font-size: .78rem;
    }

    .card {
        border: 1px solid var(--pv-border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(127,127,127,.045);
        min-height: 104px;
    }

    .label {
        text-transform: uppercase;
        letter-spacing: .10em;
        font-size: .70rem;
        opacity: .60;
        font-weight: 700;
    }

    .value {
        margin-top: .45rem;
        font-size: 1.38rem;
        font-weight: 800;
    }

    .section {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 1.35rem 0 .25rem;
    }

    .muted {
        color: var(--pv-muted);
        margin-bottom: .8rem;
    }

    .valid {
        border: 1px solid rgba(52,168,112,.30);
        background: rgba(52,168,112,.08);
        border-radius: 16px;
        padding: .9rem 1rem;
        margin: .8rem 0 1rem;
    }

    .invalid {
        border: 1px solid rgba(220,80,80,.32);
        background: rgba(220,80,80,.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 1rem 0;
    }

    .uncertain {
        border: 1px solid rgba(220,160,60,.35);
        background: rgba(220,160,60,.09);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 1rem 0;
    }

    .result-normal {
        border-left: 5px solid #35a56a;
        background: rgba(53,165,106,.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }

    .result-pneumonia {
        border-left: 5px solid #d95c5c;
        background: rgba(217,92,92,.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }

    .footer-note {
        border: 1px solid var(--pv-border);
        border-radius: 15px;
        padding: 1rem 1.15rem;
        margin-top: 2rem;
        font-size: .84rem;
        opacity: .78;
        line-height: 1.55;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: .8rem 1rem 2rem;
        }

        .hero {
            padding: 1.35rem;
            border-radius: 20px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TRANSFORM
# =========================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# BASIC IMAGE VALIDATION
# =========================================================

def basic_image_validation(uploaded_file, image):
    """Reject obviously unusable uploads before model inference."""

    if uploaded_file.size < MIN_FILE_BYTES:
        return False, "The uploaded file is too small to be a valid X-ray."

    width, height = image.size

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, "The image resolution is too small. Please upload a chest X-ray."

    if image.getbbox() is None:
        return False, "The uploaded image appears to be blank."

    arr = np.asarray(image.convert("L"))

    if arr.size == 0:
        return False, "The image could not be read correctly."

    # Very low dynamic range usually means a blank/near-blank image.
    if float(arr.std()) < 8.0:
        return False, "The image appears blank or unusable. Please upload a chest X-ray."

    return True, ""


# =========================================================
# X-RAY-LIKE HEURISTIC
# =========================================================

def xray_like_check(image):
    """
    Conservative non-X-ray gate for the demo.

    This is intentionally designed to reject obvious phone photos,
    hands, screenshots, colorful objects and other non-X-ray inputs.
    It is a heuristic gate, not a medical image classifier.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if gray.ndim != 2 or gray.size == 0:
        return False, 0.0, "Unsupported image structure."

    h, w = gray.shape

    # Strong color rejection. Typical chest X-rays are grayscale.
    channel_spread = float(
        np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))
    )

    if channel_spread > 28:
        return False, 0.0, "The image appears to be a color photograph, not a chest X-ray."

    # Very small / extremely wide images are poor candidates.
    if min(h, w) < 224:
        return False, 0.0, "The image resolution is too small for reliable X-ray analysis."

    aspect = w / max(h, 1)

    if not 0.55 <= aspect <= 1.90:
        return False, 0.0, "The image proportions do not look suitable for a chest X-ray."

    mean = float(gray.mean())
    std = float(gray.std())
    p01, p05, p95, p99 = np.percentile(gray, [1, 5, 95, 99])
    dynamic_range = float(p95 - p05)

    # Reject near-white/near-black or extremely flat images.
    if std < 15 or dynamic_range < 45:
        return False, 0.0, "The image does not contain enough X-ray-like intensity structure."

    # Edge structure.
    edges = cv2.Canny(
        gray.astype(np.uint8),
        40,
        120,
    )
    edge_density = float(np.mean(edges > 0))

    if edge_density < 0.008 or edge_density > 0.30:
        return False, 0.0, "The image does not have a suitable radiographic structure."

    # X-ray-like images generally have substantial mid-tone content,
    # rather than being dominated by a single photographic background.
    midtone_fraction = float(
        np.mean((gray >= 25) & (gray <= 235))
    )

    if midtone_fraction < 0.55:
        return False, 0.0, "The image does not resemble a chest radiograph."

    # Combine weak signals only after hard rejection rules.
    score = 0.0

    if channel_spread <= 18:
        score += 0.20
    elif channel_spread <= 28:
        score += 0.10

    if 0.70 <= aspect <= 1.65:
        score += 0.15

    if 20 <= std <= 90:
        score += 0.20

    if dynamic_range >= 80:
        score += 0.15

    if 0.012 <= edge_density <= 0.22:
        score += 0.15

    if midtone_fraction >= 0.65:
        score += 0.15

    accepted = score >= 0.68

    if not accepted:
        return (
            False,
            score,
            "The image does not look sufficiently like a chest X-ray.",
        )

    return True, score, ""


# =========================================================
# PREDICTION + GRAD-CAM
# =========================================================

def predict_and_gradcam(model, image):
    image = image.convert("RGB")
    original = np.asarray(image)

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        confidence = (
            probabilities[
                0,
                predicted_index,
            ].item()
            * 100
        )

        score = output[
            0,
            predicted_index,
        ]

        score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "Grad-CAM activations/gradients were not captured."
            )

        activation = activations[0]
        gradient = gradients[0]

        weights = torch.mean(
            gradient,
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.sum(
            weights * activation,
            dim=1,
        )

        cam = F.relu(cam)

        cam = (
            cam.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        cam_max = cam.max()

        if cam_max > 0:
            cam /= cam_max

        height, width = original.shape[:2]

        cam = cv2.resize(
            cam,
            (width, height),
        )

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET,
        )

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR,
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.55,
            heatmap,
            0.45,
            0,
        )

        overlay_rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB,
        )

        output_path = (
            GRADCAM_DIR
            / f"gradcam_{uuid.uuid4().hex}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            overlay,
        )

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "gradcam": Image.fromarray(overlay_rgb),
        }

    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-assisted chest X-ray analysis</div>
        <div class="hero-title">PneumoVision</div>
        <div class="hero-copy">
            Upload a chest X-ray to screen the image with a trained
            ResNet18 classifier and visualize the model's attention
            with Grad-CAM.
        </div>
        <div class="badge">
            ResNet18 &nbsp;•&nbsp; 2-class classification &nbsp;•&nbsp; Grad-CAM
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL STATUS
# =========================================================

try:
    model = load_model()
except Exception as exc:
    st.error(
        "The trained ResNet18 model could not be loaded."
    )
    st.code(str(exc))
    st.stop()


c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
            <div class="label">Model</div>
            <div class="value">ResNet18</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <div class="label">Classes</div>
            <div class="value">NORMAL / PNEUMONIA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="card">
            <div class="label">Runtime</div>
            <div class="value">{str(DEVICE).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section">Upload chest X-ray</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="muted">Accepted formats: JPG, JPEG, PNG</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# =========================================================
# PROCESS
# =========================================================

if uploaded_file is not None:

    try:
        image = Image.open(
            uploaded_file
        ).convert("RGB")
    except Exception:
        st.error(
            "This image could not be opened. Please upload a valid chest X-ray."
        )
        st.stop()

    st.markdown(
        '<div class="section">Image preview</div>',
        unsafe_allow_html=True,
    )

    st.image(
        image,
        width="stretch",
    )

    # Basic validation immediately after upload.
    valid_basic, basic_message = basic_image_validation(
        uploaded_file,
        image,
    )

    if not valid_basic:

        st.markdown(
            f"""
            <div class="invalid">
                <strong>Invalid image</strong><br>
                {basic_message}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.stop()

    xray_ok, xray_score, xray_reason = xray_like_check(
        image
    )

    if not xray_ok:

        st.markdown(
            f"""
            <div class="invalid">
                <strong>Please upload a chest X-ray.</strong><br>
                {xray_reason}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Input validation stopped the model before prediction."
        )

        st.stop()

    st.markdown(
        """
        <div class="valid">
            <strong>Image accepted</strong><br>
            The uploaded image passed the basic chest X-ray input checks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    analyze = st.button(
        "Analyze X-ray",
        type="primary",
        width="stretch",
    )

    if analyze:

        with st.spinner(
            "Analyzing image and generating Grad-CAM..."
        ):

            try:
                result = predict_and_gradcam(
                    model,
                    image,
                )
            except Exception as exc:
                st.error(
                    "Analysis failed."
                )
                st.code(str(exc))
                st.stop()

        prediction = result["prediction"]
        confidence = result["confidence"]

        # -----------------------------------------------------
        # CONFIDENCE GUARD
        # -----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            st.markdown(
                f"""
                <div class="uncertain">
                    <strong>Prediction uncertain</strong><br>
                    The model confidence is only {confidence:.2f}%.
                    Please upload a clearer chest X-ray rather than
                    treating this result as a reliable classification.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        confidence / 100,
                    ),
                )
            )

            st.stop()

        # -----------------------------------------------------
        # RESULT HEADER
        # -----------------------------------------------------

        st.markdown(
            '<div class="section">Analysis result</div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="label">Prediction</div>
                    <div class="value">{prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="label">Confidence</div>
                    <div class="value">{confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            st.markdown(
                """
                <div class="card">
                    <div class="label">Model</div>
                    <div class="value">ResNet18</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    confidence / 100,
                ),
            )
        )

        if prediction == "PNEUMONIA":

            st.markdown(
                """
                <div class="result-pneumonia">
                    <strong>PNEUMONIA prediction</strong><br>
                    The trained model classified this chest X-ray
                    as PNEUMONIA.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="result-normal">
                    <strong>NORMAL prediction</strong><br>
                    The trained model classified this chest X-ray
                    as NORMAL.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------
        # GRAD-CAM
        # -----------------------------------------------------

        st.markdown(
            '<div class="section">Model explanation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="muted">
                Grad-CAM highlights image regions that contributed
                to the model's prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )

        g1, g2 = st.columns(2)

        with g1:
            st.image(
                image,
                caption="Original chest X-ray",
                width="stretch",
            )

        with g2:
            st.image(
                result["gradcam"],
                caption="Grad-CAM visualization",
                width="stretch",
            )

        st.info(
            "Grad-CAM is an explanation aid and not a clinical finding."
        )

        st.markdown(
            """
            <div class="footer-note">
                <strong>Important:</strong>
                This application provides an AI-assisted prediction
                from a chest X-ray. It is not a clinical diagnosis and
                should not replace evaluation by a qualified healthcare
                professional.
            </div>
            """,
            unsafe_allow_html=True,
        )
