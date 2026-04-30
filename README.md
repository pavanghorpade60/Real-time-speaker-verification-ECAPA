# Real-Time Speaker Verification System using ECAPA-TDNN

A **real-time biometric speaker verification system** built using the pretrained **ECAPA-TDNN deep learning model** from SpeechBrain.

The system records live microphone audio, extracts **speaker embeddings**, compares them with enrolled voice samples, and verifies identity using **cosine similarity scoring**.

In addition, the system performs **speech transcription, acoustic feature analysis, and automatic logging of results into Excel**, making it useful for both **demonstration and evaluation purposes**.

---

# Project Overview

This project implements a complete **end-to-end speaker verification pipeline**.

Main capabilities:

*  Real-time microphone audio capture
*  Speaker embedding extraction using ECAPA-TDNN
*  Cosine similarity comparison against enrolled samples
*  Threshold-based identity verification
*  Speech transcription using OpenAI Whisper
*  Voice feature analysis (Pitch, Spectral Centroid, Bandwidth, Energy, MFCC)
*  Automatic logging of verification results to Excel
*  Playback links for both input and matched audio samples

The system is designed following **machine learning engineering best practices** with modular components and reproducible results.

---

# System Architecture

### End-to-End Pipeline

```
Microphone Input
      ↓
Voice Activity Detection
      ↓
Audio Normalization
      ↓
ECAPA-TDNN Speaker Embedding Extraction
      ↓
Embedding Normalization
      ↓
Cosine Similarity Comparison
      ↓
Threshold Decision
      ↓
Speaker Identified (Pavan / Unknown)
      ↓
Speech Transcription + Feature Analysis
      ↓
Results Logged to Excel
```

---

# Model Details

Model: **ECAPA-TDNN**

Pretrained on: **VoxCeleb Dataset**

Key Properties:

| Property             | Value             |
| -------------------- | ----------------- |
| Embedding Dimension  | 192               |
| Similarity Metric    | Cosine Similarity |
| Speaker Verification | Threshold Based   |
| Inference Device     | CPU / GPU         |
| Speech Recognition   | Whisper Medium    |

---

# Project Structure

```
Real-Time Speaker Identification using ECAPA-TDNN
│
├── data/
│   └── pavan/
│       └── processed_audio/        # Enrolled speaker samples
│
├── recordings/                     # Live microphone recordings
│
├── logs/
│   └── speaker_verification_log.xlsx   # Automatic result logs
│
├── pretrained_models/              # SpeechBrain downloaded models
│
├── test.py                         # Main real-time verification system
├── enroll_150.py                   # Enrollment embedding generation
├── record_samples.py               # Audio recording script
├── convert_audio.py                # Audio preprocessing utility
├── model.py                        # Model related utilities
│
├── pavan_master_embedding.pt       # Final enrolled speaker embedding
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE
└── .gitignore
```

---

# Installation

### 1. Clone the Repository

```
git clone https://github.com/pavanghorpade60/Real-time-speaker-verification-ECAPA.git
cd Real-time-speaker-verification-ECAPA
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

Main libraries used:

* PyTorch
* SpeechBrain
* Transformers
* Whisper
* Librosa
* SoundDevice
* Pandas
* OpenPyXL

---

# Enrollment Process

### Step 1 — Record Voice Samples

```
python record_samples.py
```

This records multiple voice samples for enrollment.

---

### Step 2 — Convert Audio Format

```
python convert_audio.py
```

Standardizes audio into:

* 16kHz sample rate
* Mono channel
* WAV format

---

### Step 3 — Generate Speaker Embeddings

```
python enroll_150.py
```

This script:

* Extracts ECAPA embeddings
* Normalizes embeddings
* Saves them as

```
pavan_master_embedding.pt
```

---

# Real-Time Speaker Verification

Run the system:

```
python test.py
```

The system will:

1. Listen to microphone input
2. Detect speech
3. Extract speaker embedding
4. Compare with enrolled samples
5. Transcribe speech
6. Analyze voice features
7. Log results to Excel

---

# Example Console Output

```
SMART VERIFICATION RESULT

Spoken Text           : Can you hear what I am saying?
Detected Speaker      : PAVAN

Best Matching Sample  : pavan_06.wav
Best Similarity Score : 0.5549
Average Similarity    : 0.4550

VOICE FEATURE ANALYSIS

Pitch Difference      : 22.01
Spectral Centroid Diff: 364.28
Bandwidth Difference  : 143.63
Energy Difference     : 0.0577
MFCC Distance         : 212.66
```

---

# Excel Logging System

Each verification run is automatically logged in:

```
logs/speaker_verification_log.xlsx
```

The Excel sheet contains:

| Column               | Description                      |
| -------------------- | -------------------------------- |
| Timestamp            | Time of verification             |
| Spoken Text          | Whisper transcription            |
| Detected Speaker     | Predicted identity               |
| Input Audio          | Recorded microphone audio        |
| Matched Audio        | Closest matching enrolled sample |
| Best Similarity      | Highest cosine similarity        |
| Average Similarity   | Mean similarity across samples   |
| Pitch Difference     | Pitch comparison                 |
| Spectral Centroid    | Voice brightness difference      |
| Bandwidth Difference | Frequency spread difference      |
| Energy Difference    | Loudness difference              |
| MFCC Distance        | Spectral feature difference      |

Both **input audio and matched audio are clickable** inside the Excel file.

---

# Similarity Score Interpretation

Typical cosine similarity ranges:

| Similarity Score | Meaning           |
| ---------------- | ----------------- |
| 0.65 – 0.90      | Same Speaker      |
| 0.45 – 0.65      | Possible Match    |
| 0.20 – 0.45      | Different Speaker |

Threshold used in this system:

```
0.50
```

---

# Engineering Highlights

Key engineering practices used in this project:

* Real-time microphone streaming
* Voice activity detection
* Audio normalization
* Pretrained ECAPA deep speaker embeddings
* Cosine similarity verification
* Feature analysis for interpretability
* Excel-based logging system
* Modular Python pipeline
* GPU-aware inference support

---

# Future Improvements

Potential upgrades for this system:

* Multi-speaker enrollment database
* Equal Error Rate (EER) evaluation
* ROC curve visualization
* Web API deployment
* Real-time streaming inference
* Adaptive threshold learning
* Speaker diarization support
* Noise robustness improvements

---

# Applications

This system can be used for:

* Biometric authentication
* Secure voice access systems
* Personalized voice assistants
* Voice-controlled automation
* Speaker verification research
* Audio forensics

---

# License

MIT License

---

# Author

**Pavan Ghorpade**

Machine Learning Engineer
Speech Processing & AI Systems
