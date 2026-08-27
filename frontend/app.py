import os

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="PneumoVision | Pneumonia Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem 2.1rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #111827, #26364b);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: .5rem 0;
    }
    .card {
        border: 1px solid rgba(127,127,127,.2);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 100px;
    }
    .label {
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: .7rem;
        opacity: .6;
        font-weight: 700;
    }
    .value {
        margin-top: .45rem;
        font-size: 1.35rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div>AI-ASSISTED CHEST X-RAY ANALYSIS</div>
        <div class="hero-title">PneumoVision</div>
        <div>
            ResNet18 pneumonia classification with Grad-CAM
            explanation and prediction history.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        return response.ok
    except requests.RequestException:
        return False


with st.sidebar:
    st.subheader("Backend")
    st.code(API_URL)

    if api_health():
        st.success("FastAPI: Connected")
    else:
        st.error("FastAPI: Not connected")


st.markdown("### Upload chest X-ray")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Image Preview")
    st.image(image, width="stretch")

    if st.button(
        "Analyze X-ray",
        type="primary",
        width="stretch",
    ):

        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            with st.spinner(
                "Analyzing X-ray..."
            ):
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120,
                )

        except requests.RequestException as exc:
            st.error(
                "Could not connect to FastAPI."
            )
            st.code(str(exc))
            st.stop()

        try:
            data = response.json()
        except ValueError:
            st.error(
                f"FastAPI returned an invalid response ({response.status_code})."
            )
            st.code(response.text)
            st.stop()

        if response.status_code != 200:

            st.error(
                data.get(
                    "detail",
                    "Analysis failed.",
                )
            )

        else:

            result = data["result"]

            prediction = result["prediction"]
            confidence = float(
                result["confidence"]
            )
            uncertain = result["uncertain"]

            st.markdown("### Analysis Result")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="label">Prediction</div>
                        <div class="value">{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="label">Confidence</div>
                        <div class="value">{confidence:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
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
                max(0.0, min(1.0, confidence / 100))
            )

            if uncertain:
                st.warning(
                    "Prediction is uncertain because confidence is below 65%."
                )
            elif prediction == "PNEUMONIA":
                st.error(
                    "Model prediction: PNEUMONIA"
                )
            else:
                st.success(
                    "Model prediction: NORMAL"
                )

            st.markdown("### Grad-CAM")

            gradcam = data.get("gradcam")

            if gradcam and gradcam.get("url"):

                gradcam_url = (
                    f"{API_URL}{gradcam['url']}"
                )

                try:
                    gradcam_response = requests.get(
                        gradcam_url,
                        timeout=30,
                    )

                    if gradcam_response.ok:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.image(
                                image,
                                caption="Original X-ray",
                                width="stretch",
                            )

                        with col2:
                            st.image(
                                gradcam_response.content,
                                caption="Grad-CAM",
                                width="stretch",
                            )
                    else:
                        st.warning(
                            "Grad-CAM image could not be loaded."
                        )

                except requests.RequestException:
                    st.warning(
                        "Grad-CAM image could not be loaded."
                    )

            else:
                st.warning(
                    "Grad-CAM was not generated."
                )

            db_status = data.get(
                "database",
                {},
            ).get(
                "status",
                "unknown",
            )

            if db_status == "saved":
                st.success(
                    "Prediction saved to PostgreSQL."
                )
            else:
                st.warning(
                    "Prediction completed, but database save failed."
                )

            st.info(
                "This application is an AI-assisted screening tool and not a clinical diagnosis."
            )


st.markdown("### Prediction History")

if st.button("Refresh History"):

    try:
        response = requests.get(
            f"{API_URL}/predictions",
            timeout=15,
        )

        if response.ok:

            history = response.json().get(
                "predictions",
                [],
            )

            if history:
                st.dataframe(
                    history,
                    width="stretch",
                )
            else:
                st.info(
                    "No predictions saved yet."
                )

        else:
            st.error(
                "Could not load prediction history."
            )

    except requests.RequestException as exc:
        st.error(
            "Could not connect to FastAPI."
        )
        st.code(str(exc))
