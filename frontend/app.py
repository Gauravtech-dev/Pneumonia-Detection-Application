import streamlit as st
import requests


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

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


    /* ---------- NAVBAR ---------- */

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


    /* ---------- HERO ---------- */

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


    /* ---------- SECTION ---------- */

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


    /* ---------- UPLOAD CARD ---------- */

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


    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #081624;
        border: 1px dashed #31516d;
        border-radius: 16px;
        padding: 12px;
    }


    /* ---------- RESULT CARDS ---------- */

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

    .metric-green {
        color: #4ade80;
    }


    /* ---------- INFO CARDS ---------- */

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


    /* ---------- EXPLANATION ---------- */

    .explain-box {
        background: #0b1928;
        border: 1px solid #21374d;
        border-radius: 16px;
        padding: 18px;
        color: #91a6ba;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 15px;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        background: #159bd1;
        color: white;
        border: 1px solid #38bdf8;
        border-radius: 11px;
        padding: 12px 18px;
        font-size: 15px;
        font-weight: 750;
    }

    .stButton > button:hover {
        background: #0d82b2;
        color: white;
    }


    /* ---------- FOOTER ---------- */

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
    unsafe_allow_html=True
)


# =========================================================
# NAVBAR
# =========================================================

st.markdown(
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
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">
            EXPLAINABLE AI • CHEST X-RAY ANALYSIS
        </div>
        <div class="hero-title">
            AI-Powered <span>Pneumonia</span><br>
            Screening
        </div>
        <div class="hero-text">
            Analyze a chest X-ray using a ResNet18 deep-learning
            model and visualize the regions influencing its
            prediction with Grad-CAM.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    """
    <div class="upload-card">
        <div class="upload-title">Start a New Analysis</div>
        <div class="upload-text">
            Upload a chest X-ray image. The system will return
            the model prediction, confidence score and
            explainability visualization.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Drop your X-ray here",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)


# =========================================================
# IMAGE PREVIEW
# =========================================================

if uploaded_file is not None:

    st.markdown(
        '<div class="section-title">X-Ray Preview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Review the image before sending it to the model.</div>',
        unsafe_allow_html=True
    )

    preview_col, action_col = st.columns(
        [1.35, 0.65],
        gap="large"
    )

    with preview_col:

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True
        )

    with action_col:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-label">Model</div>
                <div class="info-value">Chest ResNet18</div>
                <br>
                <div class="info-label">Input</div>
                <div class="info-value">Chest X-Ray</div>
                <br>
                <div class="info-label">Explainability</div>
                <div class="info-value">Grad-CAM</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        analyze = st.button(
            "🔍 Analyze X-Ray",
            type="primary"
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
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120
                )

                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    data = response.json()

                    prediction = data.get(
                        "prediction"
                    )

                    confidence = data.get(
                        "confidence"
                    )

                    gradcam = data.get(
                        "gradcam"
                    )


                    # ---------------------------------------------
                    # CONFIDENCE
                    # ---------------------------------------------

                    try:

                        confidence_value = float(
                            confidence
                        )

                        if 0 <= confidence_value <= 1:
                            confidence_value *= 100

                    except Exception:

                        confidence_value = 0


                    # =============================================
                    # RESULT
                    # =============================================

                    st.markdown(
                        '<div class="section-title">Analysis Result</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="section-subtitle">Output generated by the Chest ResNet18 model.</div>',
                        unsafe_allow_html=True
                    )


                    result_col1, result_col2, result_col3 = st.columns(
                        3,
                        gap="medium"
                    )


                    with result_col1:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    Prediction
                                </div>
                                <div class="metric-value">
                                    {prediction}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    with result_col2:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    Confidence
                                </div>
                                <div class="metric-value metric-blue">
                                    {confidence_value:.2f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    with result_col3:

                        st.markdown(
                            """
                            <div class="metric-card">
                                <div class="metric-label">
                                    Model
                                </div>
                                <div class="metric-value">
                                    ResNet18
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # ---------------------------------------------
                    # CONFIDENCE BAR
                    # ---------------------------------------------

                    st.write("")

                    st.caption(
                        "Model confidence"
                    )

                    st.progress(
                        min(
                            max(
                                confidence_value / 100,
                                0.0
                            ),
                            1.0
                        )
                    )


                    # ---------------------------------------------
                    # STATUS
                    # ---------------------------------------------

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


                    # =============================================
                    # GRAD-CAM
                    # =============================================

                    if gradcam:

                        st.markdown(
                            '<div class="section-title">Explainable AI</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '<div class="section-subtitle">See which image regions influenced the prediction.</div>',
                            unsafe_allow_html=True
                        )


                        if str(gradcam).startswith(
                            "http://"
                        ) or str(gradcam).startswith(
                            "https://"
                        ):

                            gradcam_url = str(
                                gradcam
                            )

                        else:

                            gradcam_url = (
                                f"{API_URL}"
                                f"{gradcam}"
                            )


                        original_col, gradcam_col = st.columns(
                            2,
                            gap="large"
                        )


                        with original_col:

                            st.markdown(
                                "**Original X-Ray**"
                            )

                            st.image(
                                uploaded_file,
                                use_container_width=True
                            )


                        with gradcam_col:

                            st.markdown(
                                "**Grad-CAM Attention Map**"
                            )

                            st.image(
                                gradcam_url,
                                use_container_width=True
                            )


                        st.markdown(
                            """
                            <div class="explain-box">
                                <strong>How this helps:</strong><br>
                                Grad-CAM highlights image regions that
                                contributed to the model's prediction.
                                This makes the deep-learning model more
                                interpretable, but the visualization is
                                not a clinical diagnosis.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # =================================================
                # ERROR
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


            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the Docker backend."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The analysis request timed out."
                )


            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )


# =========================================================
# HISTORY
# =========================================================

st.markdown(
    '<div class="section-title">Scan History</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Previous predictions stored in PostgreSQL.</div>',
    unsafe_allow_html=True
)


if st.button(
    "📋 Load Scan History"
):

    try:

        response = requests.get(
            f"{API_URL}/predictions",
            timeout=30
        )

        if response.status_code == 200:

            data = response.json()

            predictions = data.get(
                "predictions",
                []
            )


            if predictions:

                # ---------------------------------------------
                # Statistics
                # ---------------------------------------------

                total_scans = len(
                    predictions
                )

                pneumonia_count = 0
                normal_count = 0


                for row in predictions:

                    try:

                        prediction_value = str(
                            row[3]
                        ).upper()

                        if "PNEUMONIA" in prediction_value:

                            pneumonia_count += 1

                        elif "NORMAL" in prediction_value:

                            normal_count += 1

                    except Exception:
                        pass


                stat1, stat2, stat3 = st.columns(
                    3
                )


                with stat1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Total Scans
                            </div>
                            <div class="metric-value">
                                {total_scans}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with stat2:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Pneumonia Predictions
                            </div>
                            <div class="metric-value">
                                {pneumonia_count}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with stat3:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Normal Predictions
                            </div>
                            <div class="metric-value metric-green">
                                {normal_count}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.write("")

                st.dataframe(
                    predictions,
                    use_container_width=True,
                    hide_index=True
                )


            else:

                st.info(
                    "No scan history found."
                )


        else:

            st.error(
                f"Could not load history: "
                f"{response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the Docker backend."
        )


    except Exception as e:

        st.error(
            f"Error loading history: {str(e)}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🫁 PneumoScan AI<br>
        ResNet18 • Grad-CAM • PostgreSQL • FastAPI • Docker<br>
        AI-assisted screening tool — not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)