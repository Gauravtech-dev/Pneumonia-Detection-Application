CURAVISION AI — Pneumonia Detection Application

An AI-assisted chest X-ray screening application that uses a ResNet18 deep learning model to classify chest X-ray images as NORMAL or PNEUMONIA.

The application combines PyTorch, ResNet18, FastAPI, Streamlit, and Grad-CAM to provide an end-to-end ML inference workflow with visual model explainability.

Medical Disclaimer: This project is an AI-assisted screening prototype for educational and portfolio purposes. It is not a clinical diagnostic system and should not be used as a substitute for evaluation by a qualified healthcare professional.

Features

Chest X-ray classification using ResNet18

Binary classification:

NORMAL

PNEUMONIA

GPU inference with CUDA when available

FastAPI backend for model inference

Streamlit frontend for an interactive interface

Grad-CAM visualization for model explainability

Input image validation

Basic X-ray-like image validation

Confidence score display

Uncertainty warning for lower-confidence predictions

Model error analysis on the test dataset

Confusion matrix and misclassified-image analysis

No database dependency

Simple local deployment without Docker

Project Architecture

                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │     frontend/       │
                    └──────────┬──────────┘
                               │
                               │ HTTP POST /predict
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI API      │
                    │     backend/        │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             ┌─────────────┐      ┌─────────────┐
             │  ResNet18   │      │  Grad-CAM   │
             │   PyTorch   │      │ Explainable │
             └──────┬──────┘      └──────┬──────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Prediction +        │
                    │ Confidence +        │
                    │ Visualization       │
                    └─────────────────────┘

Technology Stack

Category

Technology

Programming Language

Python

Deep Learning

PyTorch

CNN Architecture

ResNet18

Computer Vision

OpenCV, Pillow

Model Explainability

Grad-CAM

Backend

FastAPI

API Server

Uvicorn

Frontend

Streamlit

Data Processing

NumPy

ML Utilities

scikit-learn

Dataset Handling

torchvision ImageFolder

Version Control

Git / GitHub

Project Structure

Pneumonia-Detection-Application/
│
├── .streamlit/
│
├── assets/
│
├── backend/
│   ├── gradcam.py
│   ├── inference.py
│   └── main.py
│
├── data/
│   └── not_chest_xray/
│
├── frontend/
│   └── app.py
│
├── model/
│   ├── chest_xray_resnet18.pth
│   ├── confusion_matrix.png
│   └── error_analysis/
│       └── misclassified_images.txt
│
├── notebooks/
│
├── train_resnet18.py
├── analyze_errors.py
├── requirements.txt
├── .gitignore
└── README.md

Machine Learning Pipeline

1. Dataset

The model was trained using a chest X-ray pneumonia dataset containing two classes:

NORMAL
PNEUMONIA

The dataset is organized using the ImageFolder structure expected by torchvision.

2. Preprocessing

Input images are resized to:

224 × 224

The training pipeline includes augmentation such as:

Random horizontal flip

Small random rotation

Color jitter

Image normalization

Evaluation uses resizing and ImageNet normalization.

3. Model

The project uses ResNet18, a convolutional neural network architecture based on residual learning.

The final classification layer is configured for two classes:

NORMAL
PNEUMONIA

The training pipeline also uses class balancing through a weighted sampler because the training dataset contains more pneumonia images than normal images.

4. Training

The training pipeline includes:

Transfer learning with pretrained ResNet18 weights

AdamW optimizer

ReduceLROnPlateau learning-rate scheduling

Validation split

Early stopping

Macro Precision

Macro Recall

Macro F1

Confusion matrix generation

Best-model checkpointing

Model Performance

Evaluation was performed on the held-out test set containing:

Test images: 624

Current error-analysis results:

Metric

Result

Test Images

624

Correct Predictions

543

Incorrect Predictions

81

Accuracy

87.02%

The detailed error-analysis report is stored in:

model/error_analysis/misclassified_images.txt

Error Analysis

The error analysis identified:

79 cases where a NORMAL image was predicted as PNEUMONIA

2 cases where a PNEUMONIA image was predicted as NORMAL

This analysis is useful for understanding model behavior beyond overall accuracy.

Some incorrect predictions also had high model confidence, showing that the displayed confidence score should not be interpreted as clinical certainty.

Grad-CAM Explainability

The application uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize image regions that contribute to the model's prediction.

The interface displays:

Original X-ray
       +
Grad-CAM visualization

This provides a visual explanation of the model's attention and makes the prediction pipeline more interpretable.

Grad-CAM is an explanatory visualization and does not establish that a highlighted region is medically diagnostic.

Backend API

The FastAPI backend provides the inference service.

Health Check

GET /health

Example response:

{
  "status": "ok"
}

Prediction

POST /predict

The endpoint accepts an uploaded image and returns the model prediction, confidence information, model name, and Grad-CAM output when successfully generated.

Running the Project Locally

1. Clone the repository

git clone <repository-url>
cd Pneumonia-Detection-Application

2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Start the FastAPI backend

From the project root:

python backend\main.py

The backend runs on:

http://127.0.0.1:8000

5. Start the Streamlit frontend

Open another terminal in the project root:

streamlit run frontend\app.py

The Streamlit interface will open in the browser.

Hardware

The model has been tested with CUDA acceleration on:

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

The application can also run on CPU when CUDA is unavailable, although inference may be slower.

API Workflow

Upload X-ray
     │
     ▼
Image validation
     │
     ▼
X-ray-like input check
     │
     ▼
Image preprocessing
     │
     ▼
ResNet18 inference
     │
     ├──────────────► Prediction
     │
     ├──────────────► Confidence
     │
     └──────────────► Grad-CAM
                         │
                         ▼
                  Streamlit result

Error Analysis

The project includes:

analyze_errors.py

This script evaluates the existing trained model on the test dataset and records misclassified images.

Run:

python analyze_errors.py

The report is generated at:

model/error_analysis/misclassified_images.txt

This allows individual incorrect predictions to be inspected instead of relying only on aggregate metrics.

Important Limitations

This project has several important limitations:

It is a portfolio/educational ML prototype, not a clinical diagnostic tool.

The model can produce incorrect predictions.

High confidence does not guarantee correctness.

The test evaluation contains both false-positive and false-negative predictions.

Dataset characteristics may not represent all patient populations, imaging equipment, or clinical settings.

Grad-CAM provides an interpretability visualization but should not be treated as medical evidence.

Real clinical deployment would require extensive external validation, calibration, safety testing, regulatory review, and expert clinical evaluation.

Future Improvements

Potential improvements include:

External validation on an independent dataset

Better calibration of confidence scores

More extensive error analysis

Threshold analysis for different clinical use cases

Improved data augmentation

Experimentation with stronger CNN/vision architectures

Model ensemble methods

Automated experiment tracking

Unit and API integration tests

Production-grade monitoring and logging

Why This Project

This project demonstrates an end-to-end machine learning workflow rather than only model training:

Dataset
   ↓
Preprocessing
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Evaluation
   ↓
Error Analysis
   ↓
Grad-CAM Explainability
   ↓
FastAPI Inference API
   ↓
Streamlit Application

It demonstrates practical skills in:

Deep Learning

Computer Vision

Transfer Learning

PyTorch

Model Evaluation

Error Analysis

Explainable AI

REST API development

Streamlit

Git/GitHub

Disclaimer

This application is developed for educational, research, and portfolio purposes only.

It is an AI-assisted screening prototype and is not intended to provide medical diagnosis, treatment recommendations, or clinical decisions.

Always consult a qualified healthcare professional for medical interpretation of chest X-rays.

Author

Gaurav Gangwar

B.Tech — Artificial Intelligence & Machine Learning