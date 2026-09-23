# 🩻 PneumoVision — AI-Powered Pneumonia Detection

PneumoVision is an AI-assisted chest X-ray analysis application that uses a **ResNet18 deep learning model** to classify chest X-ray images into **NORMAL** or **PNEUMONIA**.

The application provides a web-based interface using **Streamlit** and exposes the trained model through a **FastAPI REST API**.

> ⚠️ This project is intended for educational and AI-assisted screening purposes only. It is not a clinical diagnostic system and should not replace professional medical evaluation.

---

## 🚀 Live Application

### Frontend
**Streamlit:**  
`Add your deployed Streamlit URL here`

### Backend API
**FastAPI:**  
https://pneumonia-detection-application.onrender.com

### API Documentation
https://pneumonia-detection-application.onrender.com/docs

---

## ✨ Features

- 🩻 Chest X-ray image upload
- 🤖 ResNet18-based pneumonia classification
- 📊 Prediction confidence
- 🟢 NORMAL classification
- 🔴 PNEUMONIA classification
- 🟡 UNCERTAIN / UNSUPPORTED handling
- ⚡ FastAPI REST API
- 🎨 Streamlit interactive frontend
- 🔍 Basic image validation
- 🧠 GPU-supported model training
- ☁️ Cloud deployment using Render
- 📦 ONNX Runtime optimized inference for lightweight deployment
- 🛡️ Conservative confidence-based prediction handling

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │     Streamlit UI     │
                 │      Frontend        │
                 └──────────┬───────────┘
                            │
                            │ HTTP POST
                            ▼
                 ┌──────────────────────┐
                 │      FastAPI        │
                 │      Backend        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Image Processing   │
                 │ Resize + Normalize   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      ResNet18       │
                 │  Pneumonia Model    │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ NORMAL / PNEUMONIA /       │
              │ UNCERTAIN / UNSUPPORTED    │
              └────────────────────────────┘