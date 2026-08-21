# Pneumonia Detection Application

An AI-assisted medical imaging application that uses **Deep Learning and Computer Vision** to detect pneumonia from **Chest X-ray images**.

The application uses a trained **ResNet18** model to classify Chest X-ray images into two classes:

- **NORMAL**
- **PNEUMONIA**

The project also uses **Grad-CAM** to provide a visual explanation of the model's prediction. A *Fast API* backend handles model inference, **PostgreSQL** stores prediction history, and **Stream lit** provides the user interface.

> Medical Disclaimer: This project is an educational/research prototype. It is not a clinically validated diagnostic system and should not replace a qualified doctor or radiologist.

---

## Features

- Chest X-ray image upload
- Pneumonia detection using ResNet18
- Binary image classification
- NORMAL vs PNEUMONIA prediction
- Prediction confidence score
- Grad-CAM visualization
- Explainable AI
- Fast API REST API
- PostgreSQL database integration
- Prediction history
- Stream lit web interface
- Interactive API documentation with Swagger UI

---

## Project Workflow

```text
                 Chest X-ray
                      |
                      ↓
             Image Preprocessing
                      |
                      ↓
                  ResNet18
                      |
              ┌───────┴───────┐
              ↓               ↓
           NORMAL         PNEUMONIA
              |               |
              └───────┬───────┘
                      ↓
              Confidence Score
                      |
                      ↓
                 Grad-CAM
                      |
                      ↓
                 Fast API
                      |
                      ↓
                PostgreSQL
                      |
                      ↓
                 Stream lit
                      |
                      ↓
                  User
