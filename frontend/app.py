import os

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# CUSTOM CSS
# ============================================================

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
    margin: 0.5rem 0;
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


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div>AI-ASSISTED CHEST X-RAY ANALYSIS</div>'
    '<div class="hero-title">PneumoVision</div>'
    '<div>ResNet18 pneumonia classification with Grad-CAM explanation.</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND HEALTH
# ============================================================

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


# ============================================================
# UPLOAD
# ============================================================

st.markdown("### Upload chest X-ray")

uploaded_file = st.file_uploader(
    "Upload image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("### Image Preview")

    st.image(
        image,
        width="stretch",
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

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


        # ====================================================
        # RESPONSE JSON
        # ====================================================

        try:

            data = response.json()

        except ValueError:

            st.error(
                "FastAPI returned an invalid response "
                f"({response.status_code})."
            )

            st.code(response.text)

            st.stop()


        # ====================================================
        # API ERROR
        # ====================================================

        if response.status_code != 200:

            st.error(
                data.get(
                    "detail",
                    "Analysis failed.",
                )
            )

            st.stop()


        # ====================================================
        # RESULT
        # ====================================================

        result = data["result"]

        prediction = result["prediction"]

        confidence = float(
            result["confidence"]
        )

        uncertain = bool(
            result.get(
                "uncertain",
                False,
            )
        )

        supported = bool(
            result.get(
                "supported",
                not uncertain,
            )
        )

        reason = result.get(
            "reason",
            "",
        )


        # ====================================================
        # ANALYSIS RESULT
        # ====================================================

        st.markdown("### Analysis Result")

        c1, c2, c3 = st.columns(3)


        # Prediction
        with c1:

            st.markdown(
                f"""
                <div class="card">
                    <div class="label">
                        Prediction
                    </div>
                    <div class="value">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # Confidence
        with c2:

            st.markdown(
                f"""
                <div class="card">
                    <div class="label">
                        Confidence
                    </div>
                    <div class="value">
                        {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # Model
        with c3:

            st.markdown(
                """
                <div class="card">
                    <div class="label">
                        Model
                    </div>
                    <div class="value">
                        ResNet18
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # Confidence bar
        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    confidence / 100,
                ),
            )
        )


        # ====================================================
        # STATUS MESSAGE
        # ====================================================

        if prediction == "PNEUMONIA":

            st.error(
                "Model prediction: PNEUMONIA"
            )


        elif prediction == "NORMAL":

            st.success(
                "Model prediction: NORMAL"
            )


        elif prediction == "UNCERTAIN / UNSUPPORTED":

            st.warning(
                "The uploaded X-ray could not be "
                "reliably classified as NORMAL or PNEUMONIA."
            )

            if reason:

                st.info(reason)

            st.info(
                "Please upload a clear chest X-ray "
                "suitable for this pneumonia detection model."
            )


        else:

            st.warning(
                "Prediction could not be supported."
            )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        st.markdown("### Grad-CAM")


        # Grad-CAM only for supported predictions
        if supported:

            gradcam = data.get("gradcam")


            if (
                gradcam
                and gradcam.get("url")
            ):

                gradcam_url = (
                    f"{API_URL}"
                    f"{gradcam['url']}"
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


        else:

            st.info(
                "Grad-CAM is not generated because "
                "the prediction did not pass the "
                "reliability check."
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.info(
            "This application is an AI-assisted "
            "screening tool and not a clinical diagnosis."
        )