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


st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "chest_xray_resnet18.pth"
GRADCAM_DIR = PROJECT_ROOT / "assets" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:1.8rem 2rem;border:1px solid rgba(120,120,120,.18);
border-radius:22px;margin-bottom:1.2rem;background:linear-gradient(135deg,rgba(20,30,48,.96),rgba(38,55,80,.92))}
.hero-kicker{font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;opacity:.75;margin-bottom:.55rem}
.hero-title{font-size:clamp(2rem,5vw,3.4rem);font-weight:800;line-height:1.05;margin:0}
.hero-subtitle{margin-top:.8rem;font-size:1rem;opacity:.82;max-width:760px;line-height:1.55}
.metric-card{border:1px solid rgba(120,120,120,.18);border-radius:18px;padding:1.1rem 1.2rem;min-height:110px;background:rgba(127,127,127,.055)}
.metric-label{font-size:.76rem;text-transform:uppercase;letter-spacing:.09em;opacity:.65}
.metric-value{margin-top:.45rem;font-size:1.65rem;font-weight:750}
.section-title{font-size:1.35rem;font-weight:750;margin-top:1.3rem;margin-bottom:.2rem}
.section-subtitle{opacity:.65;margin-bottom:1rem}
.result-normal{border-left:5px solid #35a56a;padding:1rem 1.2rem;border-radius:12px;background:rgba(53,165,106,.08);margin:1rem 0}
.result-pneumonia{border-left:5px solid #d95c5c;padding:1rem 1.2rem;border-radius:12px;background:rgba(217,92,92,.08);margin:1rem 0}
.disclaimer{margin-top:2rem;padding:1rem 1.2rem;border-radius:14px;border:1px solid rgba(120,120,120,.16);font-size:.86rem;opacity:.8}
@media(max-width:700px){.block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}.hero{padding:1.35rem;border-radius:17px}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    clean_state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(clean_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def predict_and_gradcam(model, image):
    image = image.convert("RGB")
    original = np.array(image)
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    target_layer = model.layer4[-1]
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        output = model(input_tensor)
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
        cam = F.relu(cam).squeeze().detach().cpu().numpy()

        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max

        height, width = original.shape[:2]
        cam = cv2.resize(cam, (width, height))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET,
        )

        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(
            original_bgr, 0.55, heatmap, 0.45, 0
        )
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        output_path = GRADCAM_DIR / filename
        cv2.imwrite(str(output_path), overlay)

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "gradcam": Image.fromarray(overlay_rgb),
        }

    finally:
        fh.remove()
        bh.remove()
        model.zero_grad(set_to_none=True)


st.markdown("""
<div class="hero">
  <div class="hero-kicker">AI-assisted chest X-ray analysis</div>
  <div class="hero-title">Pneumonia Detection</div>
  <div class="hero-subtitle">
    Upload a chest X-ray and get a ResNet18 prediction, confidence score,
    and Grad-CAM visual explanation.
  </div>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as exc:
    st.error("The trained ResNet18 model could not be loaded.")
    st.code(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">ResNet18</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">NORMAL / PNEUMONIA</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Runtime</div><div class="metric-value">{str(DEVICE).upper()}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Upload X-ray</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">JPG, JPEG or PNG chest X-ray</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="section-title">Image Preview</div>', unsafe_allow_html=True)
    st.image(image, width="stretch")

    if st.button("Analyze X-ray", type="primary", width="stretch"):
        with st.spinner("Running ResNet18 inference and generating Grad-CAM..."):
            try:
                result = predict_and_gradcam(model, image)
            except Exception as exc:
                st.error("Analysis failed.")
                st.exception(exc)
                st.stop()

        prediction = result["prediction"]
        confidence = result["confidence"]

        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Prediction</div><div class="metric-value">{prediction}</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence:.2f}%</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                '<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">ResNet18</div></div>',
                unsafe_allow_html=True,
            )

        st.progress(max(0.0, min(1.0, confidence / 100.0)))

        if prediction == "PNEUMONIA":
            st.markdown(
                '<div class="result-pneumonia"><strong>Pneumonia prediction</strong><br>The model classified this X-ray as PNEUMONIA.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-normal"><strong>Normal prediction</strong><br>The model classified this X-ray as NORMAL.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Grad-CAM Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Highlighted regions show areas that contributed to the model prediction.</div>',
            unsafe_allow_html=True,
        )

        g1, g2 = st.columns(2)
        with g1:
            st.image(image, caption="Original X-ray", width="stretch")
        with g2:
            st.image(result["gradcam"], caption="Grad-CAM", width="stretch")

        st.info("Grad-CAM is an explanation aid, not a clinical finding.")

        st.markdown(
            '<div class="disclaimer"><strong>Important:</strong><br>This application provides an AI-assisted prediction from a chest X-ray. It is not a clinical diagnosis and should not replace evaluation by a qualified healthcare professional.</div>',
            unsafe_allow_html=True,
        )
