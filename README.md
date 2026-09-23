# PneumoVision - Pneumonia Detection Application

PneumoVision is a chest X-ray classification project built using deep learning. The application takes a chest X-ray image as input and predicts whether the image belongs to the `NORMAL` or `PNEUMONIA` class.

The project uses a ResNet18 model for image classification, FastAPI for serving the model, and Streamlit for the frontend.

> Note: This is an academic/portfolio project and is not intended for clinical diagnosis

# Frontend: 
https://fr-4dzc.onrender.com



# Backend
FastAPI: https://pneumonia-detection-application.onrender.com

# API Documentation
https://pneumonia-detection-application.onrender.com/docs

# Features

- Upload chest X-ray images in JPG, JPEG, or PNG format
- Classify images as NORMAL or PNEUMONIA
- Display prediction confidence
- Handle uncertain predictions
- Image validation before inference
- REST API using FastAPI
- Streamlit-based user interface
- ResNet18 transfer learning
- ONNX Runtime for lightweight inference deployment
- Deployed backend using Render

# Architecture

Streamlit Frontend
       |
       | POST /predict
       v
FastAPI Backend
       |
       v
Image Preprocessing
       |
       v
ResNet18 Model
       |
       v
Prediction
       |
       +---- NORMAL
       |
       +---- PNEUMONIA
       |
       +---- UNCERTAIN / UNSUPPORTED

Tech Stack

Machine Learning

Python
PyTorch
Torchvision
ResNet18
Scikit-learn
NumPy

Backend

FastAPI
Uvicorn
Pillow

Frontend

Streamlit
HTML
CSS

Deployment

GitHub
Render
ONNX Runtime       