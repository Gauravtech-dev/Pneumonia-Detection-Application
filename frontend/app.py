import streamlit as st
import requests


# =========================
# CONFIGURATION
# =========================

API_URL = "http://127.0.0.1:8000"


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="CuraVision AI",
    page_icon="🩺",
    layout="wide"
)


# =========================
# TITLE
# =========================

st.title("CuraVision AI")

st.write(
    "AI-assisted Chest X-Ray analysis using ResNet18"
)


# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Chest X-Ray",
        width=500
    )

    if st.button("Analyze X-Ray"):

        with st.spinner("Analyzing X-Ray..."):

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

                if response.status_code == 200:

                    data = response.json()

                    # =========================
                    # RESULT
                    # =========================

                    result = data["result"]

                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    st.subheader("Prediction")

                    st.success(prediction)

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    # =========================
                    # GRAD-CAM
                    # =========================

                    gradcam = data.get("gradcam")

                    if gradcam:

                        gradcam_url = (
                            f"{API_URL}"
                            f"{gradcam['url']}"
                        )

                        st.subheader(
                            "Grad-CAM Explanation"
                        )

                        st.image(
                            gradcam_url,
                            caption=(
                                "Regions influencing "
                                "the model prediction"
                            )
                        )

                else:

                    st.error(
                        f"API Error: "
                        f"{response.status_code}"
                    )

                    try:

                        st.write(
                            response.json()
                        )

                    except Exception:

                        st.write(
                            response.text
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the "
                    "CuraVision AI backend. "
                    "Please make sure FastAPI "
                    "is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The prediction request "
                    "timed out."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )


# =========================
# PREDICTION HISTORY
# =========================

st.divider()

st.subheader("Prediction History")

if st.button("Load History"):

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

                st.dataframe(
                    predictions,
                    use_container_width=True
                )

            else:

                st.info(
                    "No prediction history found."
                )

        else:

            st.error(
                f"Could not load history: "
                f"{response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the backend."
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )