# CuraVision AI

### AI-Powered Multimodal Medical Assistant

> An AI-assisted medical platform combining Chest X-Ray analysis with a conversational medical assistant.

---

## 🚧 Project Status

**Currently under development**

CuraVision AI is being developed as a major academic project focused on combining Deep Learning, Computer Vision, Generative AI, Backend Development, and Database Management into a single application.

---

## 🎯 Project Vision

CuraVision AI aims to provide an AI-assisted platform where users can:

- Upload Chest X-Ray images
- Analyze X-Ray images using Deep Learning
- View model predictions and confidence scores
- Visualize model attention using Grad-CAM
- Ask medical questions through a conversational AI
- Get information from trusted medical knowledge sources
- Maintain analysis and chat history

> **Medical Disclaimer:** This project is intended for educational and research purposes. AI predictions are not a substitute for professional medical diagnosis.

---

##  Planned Architecture

```text
                    CURAVISION AI
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Streamlit UI            Medical Chat
             │                       │
             └───────────┬───────────┘
                         ▼
                    FastAPI API
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Chest X-Ray AI          RAG + LLM
       PyTorch / ResNet        Medical QA
              │                     │
              └──────────┬──────────┘
                         ▼
                    PostgreSQL
