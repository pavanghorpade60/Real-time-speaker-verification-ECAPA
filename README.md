# Real-Time Speaker Identification using ECAPA-TDNN
git clone <repo_link>
cd real-time-speaker-identification
pip install -r requirements.txt
## Overview

This project implements a real-time speaker identification system using the ECAPA-TDNN deep learning architecture in PyTorch.

The system captures live microphone audio, extracts discriminative speaker embeddings, and performs instant speaker classification with low latency.

An optional gender classification head is also integrated as an auxiliary task.

The goal of this project is to design and deploy a complete end-to-end speech AI pipeline — from audio capture to real-time inference.

---

## Key Features

- Real-time microphone streaming  
- ECAPA-TDNN based speaker embedding extraction  
- Multi-speaker classification  
- Optional gender classification module  
- Modular and extensible architecture  
- Low-latency inference pipeline  

---

## System Architecture

```mermaid
flowchart LR
    A[Live Microphone Input] --> B[Audio Preprocessing]
    B --> C[Mel Spectrogram Extraction]
    C --> D[ECAPA-TDNN Backbone]
    D --> E[Speaker Embedding Vector]
    E --> F[Speaker Classification Head]
    E --> G[Gender Classification Head (Optional)]
    F --> H[Predicted Speaker]
    G --> I[Predicted Gender]
    How the Pipeline Works

Live audio is captured using the system microphone.

Audio is normalized and preprocessed.

Log-Mel Spectrogram features are extracted.

The ECAPA-TDNN backbone generates a fixed-length speaker embedding.

The embedding is passed to:

A speaker classification layer

(Optional) a gender classification layer

Predictions are displayed in real time.

Technical Stack

Python

PyTorch

Torchaudio

NumPy

SoundDevice

Hugging Face (pretrained ECAPA components)

Project Structure
real-time-speaker-identification/
│
├── model.py                # ECAPA-TDNN architecture + classifier heads
├── build_voiceprint.py     # Voiceprint generation
├── record_samples.py       # Audio data collection
├── test.py                 # Real-time inference
├── utils.py                # Helper utilities
├── requirements.txt
├── README.md
└── data/                   # Stored speaker samples
Installation
git clone https://github.com/your-username/real-time-speaker-identification.git
cd real-time-speaker-identification
pip install -r requirements.txt
Running Real-Time Inference
python test.py

Speak into the microphone to receive live speaker predictions.

Performance & Observations

Stable real-time inference

Effective speaker discrimination using learned embeddings

Modular design allows easy scaling to multiple speakers

Embedding-based architecture supports future unknown speaker detection

Future Enhancements

Unknown speaker rejection using cosine similarity threshold

Larger multi-speaker dataset

Confidence score visualization

Web deployment (Streamlit / FastAPI)

REST API integration

Model quantization for edge deployment

Learning Outcomes

Through this project, I gained hands-on experience in:

Speaker embedding extraction

Deep learning for speech applications

Real-time inference system design

Audio preprocessing pipelines

Model integration and deployment workflows

Author

Pavan Ghorpade
Machine Learning & AI Enthusiast

This project represents a practical implementation of modern speaker recognition systems using state-of-the-art deep learning architectures.