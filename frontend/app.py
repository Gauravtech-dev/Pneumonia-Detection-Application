import os

import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = os.getenv(
    "PNEUMONIA_API_URL",
    "https://pneumonia-detection-application.onrender.com",
).rstrip("/")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(14, 165, 233, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 25%,
                rgba(34, 197, 94, 0.06),
                transparent 25%
            ),
            #06101d;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 24px;
        padding-bottom: 60px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(13, 27, 43, 0.92);
        border: 1px solid #23384e;
        border-radius: 18px;
        padding: 14px 20px;
        margin-bottom: 42px;
    }

    .logo {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
    }

    .logo-blue {
        color: #38bdf8;
    }

    .online {
        color: #a7b8ca;
        font-size: 13px;
    }

    .green-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 7px;
        box-shadow: 0 0 10px rgba(34,197,94,0.7);
    }

    .hero {
        text-align: center;
        margin-bottom: 42px;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 15px;
        border-radius: 30px;
        background: rgba(14, 165, 233, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #7dd3fc;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
        margin-bottom: 18px;
    }

    .hero-title {
        color: #ffffff;
        font-size: 50px;
        line-height: 1.08;
        font-weight: 850;
        margin-bottom: 15px;
    }

    .hero-title span {
        color: #38bdf8;
    }

    .hero-text {
        max-width: 720px;
        margin: auto;
        color: #8fa5ba;
        font-size: 16px;
        line-height: 1.7;
    }

    .section-title {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #8298ad;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .upload-card {
        background: rgba(13, 27, 43, 0.92);
        border: 1px solid #23384e;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
    }

    .upload-title {
        color: #ffffff;
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .upload-text {
        color: #8499ae;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    [data-testid="stFileUploader"] {
        background: #081624;
        border: 1px dashed #31516d;
        border-radius: 16px;
        padding: 12px;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            #0e2032,
            #0a1726
        );
        border: 1px solid #29425a;
        border-radius: 18px;
        padding: 23px;
        min-height: 125px;
    }

    .metric-label {
        color: #7890a7;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 850;
        margin-top: 10px;
    }

    .metric-blue {
        color: #38bdf8;
    }

    .info-card {
        background: #0b1928;
        border: 1px solid #21374d;
        border-radius: 15px;
        padding: 18px;
        height: 100%;
    }

    .info-label {
        color: #71869c;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .info-value {
        color: #ffffff;
        font-size: 16px;
        font-weight: 700;
        margin-top: 7px;
    }

    .disclaimer {
        background: #0b1928;
        border: 1px solid #21374d;
        border-radius: 15px;
        padding: 18px;
        margin-top: 25px;
        color: #8298ad;
        line-height: 1.7;
        font-size: 13px;
    }

    .disclaimer strong {
        color: #ffffff;
    }

    .footer {
        text-align: center;
        color: #61758b;
        font-size: 12px;
        line-height: 1.8;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid #182b3e;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NAVBAR
# =========================================================

st.html(
    """
    <div class="topbar">
        <div class="logo">
            🫁 Pneumo<span class="logo-blue">Scan</span> AI
        </div>

        <div class="online">
            <span class="green-dot"></span>
            Backend Online
        </div>
    </div>
    """
)


# =========================================================
# HERO
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="hero-tag">
            AI • CHEST X-RAY ANALYSIS
        </div>

        <div class="hero-title">
            AI-Powered <span>Pneumonia</span><br>
            Screening
        </div>

        <div class="hero-text">
            Analyze a chest X-ray using a ResNet18
            deep-learning model and receive an AI-based
            pneumonia prediction with confidence.
        </div>

    </div>
    """
)


# =========================================================
# UPLOAD CARD
# =========================================================

st.html(
    """
    <div class="upload-card">

        <div class="upload-title">
            Start a New Analysis
        </div>

        <div class="upload-text">
            Upload a chest X-ray image. The system will
            return the model prediction and confidence.
        </div>

    </div>
    """
)


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Drop your X-ray here",
    type=["jpg", "jpeg", "png"],
)


# =========================================================
# IMAGE PREVIEW
# =========================================================

if uploaded_file is not None:

    st.html(
        """
        <div class="section-title">
            X-Ray Preview
        </div>

        <div class="section-subtitle">
            Review the image before sending it to the model.
        </div>
        """
    )

    preview_col, action_col = st.columns(
        [1.35, 0.65],
        gap="large",
    )

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with preview_col:

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # MODEL INFO
    # -----------------------------------------------------

    with action_col:

        st.html(
            """
            <div class="info-card">

                <div class="info-label">
                    Model
                </div>

                <div class="info-value">
                    Chest ResNet18
                </div>

                <br>

                <div class="info-label">
                    Input
                </div>

                <div class="info-value">
                    Chest X-Ray
                </div>

                <br>

                <div class="info-label">
                    Backend
                </div>

                <div class="info-value">
                    FastAPI + PostgreSQL
                </div>

            </div>
            """
        )

        st.write("")

        analyze = st.button(
            "🔍 Analyze X-Ray",
            type="primary",
            use_container_width=True,
        )


    # =====================================================
    # ANALYSIS
    # =====================================================

    if analyze:

        with st.spinner(
            "Running AI analysis..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=180,
                )

                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    data = response.json()

                    result = data.get("result", {})

                    if not isinstance(result, dict):
                        result = {}

                    prediction = result.get(
                        "prediction",
                        data.get("prediction", "Unknown"),
                    )

                    confidence = result.get(
                        "confidence",
                        data.get("confidence", 0),
                    )

                    gradcam_data = data.get("gradcam")
                    database_data = data.get("database", {})

                    # -------------------------------------------------
                    # Confidence conversion
                    # -------------------------------------------------

                    try:

                        confidence_value = float(
                            confidence
                        )

                        if 0 <= confidence_value <= 1:
                            confidence_value *= 100

                    except Exception:

                        confidence_value = 0.0


                    # =================================================
                    # RESULT
                    # =================================================

                    st.html(
                        """
                        <div class="section-title">
                            Analysis Result
                        </div>

                        <div class="section-subtitle">
                            Output generated by the Chest ResNet18 model.
                        </div>
                        """
                    )


                    result_col1, result_col2, result_col3 = st.columns(
                        3,
                        gap="medium",
                    )


                    # -------------------------------------------------
                    # Prediction
                    # -------------------------------------------------

                    with result_col1:

                        st.html(
                            f"""
                            <div class="metric-card">

                                <div class="metric-label">
                                    Prediction
                                </div>

                                <div class="metric-value">
                                    {prediction}
                                </div>

                            </div>
                            """
                        )


                    # -------------------------------------------------
                    # Confidence
                    # -------------------------------------------------

                    with result_col2:

                        st.html(
                            f"""
                            <div class="metric-card">

                                <div class="metric-label">
                                    Confidence
                                </div>

                                <div class="metric-value metric-blue">
                                    {confidence_value:.2f}%
                                </div>

                            </div>
                            """
                        )


                    # -------------------------------------------------
                    # Model
                    # -------------------------------------------------

                    with result_col3:

                        st.html(
                            """
                            <div class="metric-card">

                                <div class="metric-label">
                                    Model
                                </div>

                                <div class="metric-value">
                                    ResNet18
                                </div>

                            </div>
                            """
                        )


                    # =================================================
                    # CONFIDENCE BAR
                    # =================================================

                    st.write("")

                    st.caption("Model confidence")

                    st.progress(
                        min(
                            max(
                                confidence_value / 100,
                                0.0,
                            ),
                            1.0,
                        )
                    )


                    # =================================================
                    # PREDICTION STATUS
                    # =================================================

                    prediction_text = str(
                        prediction
                    ).upper()


                    if "PNEUMONIA" in prediction_text:

                        st.warning(
                            "⚠️ The model prediction indicates Pneumonia."
                        )

                    elif "NORMAL" in prediction_text:

                        st.success(
                            "✓ The model prediction indicates Normal."
                        )

                    else:

                        st.info(
                            f"Model prediction: {prediction}"
                        )


                    # =================================================
                    # GRAD-CAM
                    # =================================================

                    if isinstance(gradcam_data, dict):

                        gradcam_url = gradcam_data.get("url")

                        if gradcam_url:

                            st.html(
                                """
                                <div class="section-title">
                                    Grad-CAM Explanation
                                </div>

                                <div class="section-subtitle">
                                    Visual explanation of the image regions
                                    that influenced the model prediction.
                                </div>
                                """
                            )

                            if gradcam_url.startswith("/"):
                                gradcam_full_url = (
                                    f"{API_URL}{gradcam_url}"
                                )
                            else:
                                gradcam_full_url = gradcam_url

                            try:

                                gradcam_response = requests.get(
                                    gradcam_full_url,
                                    timeout=30,
                                )

                                if gradcam_response.status_code == 200:

                                    st.image(
                                        gradcam_response.content,
                                        caption="Grad-CAM",
                                        use_container_width=True,
                                    )

                                else:

                                    st.warning(
                                        "Grad-CAM image could not be loaded."
                                    )

                            except requests.exceptions.RequestException:

                                st.warning(
                                    "Grad-CAM image could not be loaded."
                                )


                    # =================================================
                    # DATABASE STATUS
                    # =================================================

                    if isinstance(database_data, dict):

                        database_status = database_data.get("status")

                        if database_status == "saved":

                            st.success(
                                "Prediction saved to database."
                            )

                        elif database_status == "failed":

                            st.warning(
                                "Prediction completed, but database save failed."
                            )


                    # =================================================
                    # DISCLAIMER
                    # =================================================

                    st.html(
                        """
                        <div class="disclaimer">

                            <strong>Important:</strong><br>

                            This application provides an
                            AI-assisted prediction based on a
                            chest X-ray image. It is not a
                            clinical diagnosis and should not
                            replace evaluation by a qualified
                            healthcare professional.

                        </div>
                        """
                    )


                # =================================================
                # API ERROR
                # =================================================

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except Exception:

                        st.write(
                            response.text
                        )


            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    f"Cannot connect to the FastAPI backend: {API_URL}"
                )


            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "The analysis request timed out. "
                    "Please try again."
                )


            # =====================================================
            # OTHER ERROR
            # =====================================================

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="footer">

        🫁 PneumoScan AI<br>

        ResNet18 • FastAPI • Docker • Render<br>

    </div>
    """
)