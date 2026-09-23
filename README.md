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

#Tech stack

#Machine Learning

Python
PyTorch
Torchvision
ResNet18
Scikit-learn
NumPy

#Backend

FastAPI
Uvicorn
Pillow

#Frontend

Streamlit
HTML
CSS

#Deployment

GitHub
Render
ONNX Runtime       

#Model

The application uses ResNet18, a convolutional neural network architecture commonly used for image classification.

The model was trained for binary classification:

0 → NORMAL
1 → PNEUMONIA

For deployment, the trained PyTorch model was converted to ONNX format.

Why ONNX Runtime?

The original model was trained using PyTorch. For cloud deployment, ONNX Runtime is used to avoid loading the complete PyTorch runtime during inference.

This makes the backend more suitable for a lightweight CPU deployment.

#Model Performance

Evaluation on the test dataset:

#Metric	Score
Accuracy	88.46%

#macro Precision	91.98%

#Macro Recall	84.70%

#Macro F1-Score	86.72%

Classification Report
Class	Precision	Recall	F1-Score
NORMAL	0.99	0.70	0.82
PNEUMONIA	0.85	1.00	0.92
Confusion Matrix
                 Predicted
               NORMAL  PNEUMONIA

Actual NORMAL     163       71
Actual PNEUMONIA    1      389

The test results show that the model performs differently across the two classes, so the application also supports an uncertain outcome for predictions that do not meet the configured confidence conditions.


Future Improvements

Improve model performance using additional training data
Add better class balancing and augmentation
Improve false-positive handling
Add model explainability
Add automated model monitoring
Improve frontend error handling
Add additional medical imaging datasets for evaluation
Disclaimer

This project is developed for educational and demonstration purposes only.

It is not a medical diagnostic system and should not be used to make medical decisions. Any medical interpretation should be performed by a qualified healthcare professional.

Author

Gaurav Gangwar