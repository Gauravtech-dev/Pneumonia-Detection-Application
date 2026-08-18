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

## Future Scope

CuraVision AI is designed with a modular architecture so that additional
medical imaging capabilities can be integrated progressively.

### Medical Vision Expansion

The current implementation focuses on Chest X-ray analysis. In future
versions, the Medical Vision module will be extended with:

- Body-Part Classification
  - Chest
  - Hand
  - Knee
  - Spine
  - Dental
  - Other X-ray categories

- Dedicated AI Models
  - Chest X-ray model
  - Hand X-ray model
  - Knee X-ray model
  - Spine X-ray model
  - Dental X-ray model

The planned workflow is:

```text
Uploaded X-ray
      ↓
Body-Part Classifier
      ↓
Identify X-ray Category
      ↓
Corresponding Medical Vision Model
      ↓
Prediction / Findings
      ↓
Result + Explanation
