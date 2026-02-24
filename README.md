#  Real-Time Speaker Verification using ECAPA-TDNN

A real-time biometric speaker verification system built using a pretrained **ECAPA-TDNN** model and cosine similarity scoring.

The system performs structured speaker enrollment, extracts 192-dimensional speaker embeddings, and verifies identity using threshold-based comparison. Multi-trial averaging is implemented to improve robustness and reduce embedding variance.

---

##  Project Overview

This project implements a complete speaker verification pipeline:

- 🎤 Structured voice sample collection  
- 🔄 Audio preprocessing & standardization  
- 🧠 Speaker embedding extraction (ECAPA-TDNN)  
- 💾 Enrollment embedding storage  
- 📊 Cosine similarity scoring  
- 🔐 Threshold-based identity verification  
- 📈 Multi-trial stabilization for improved accuracy  

The system is modular, scalable, and designed following production-oriented ML engineering practices.

---

##  System Architecture

###  End-to-End Pipeline

```
Raw Audio / Microphone
        ↓
Audio Standardization (16kHz, Mono)
        ↓
ECAPA-TDNN Embedding Extraction
        ↓
L2 Normalization
        ↓
Cosine Similarity Comparison
        ↓
Threshold Decision
        ↓
Speaker: Pavan / Unknown
```

---

##  Model Details

- Architecture: ECAPA-TDNN  
- Pretrained on: VoxCeleb dataset  
- Embedding Dimension: 192  
- Similarity Metric: Cosine Similarity  
- Decision Strategy: Threshold-based classification  
- Optional Stabilization: Multi-trial embedding averaging  

---

##  Project Structure

```
Real-time-speaker-verification-ECAPA/
│
├── model.py                # Custom ECAPA-style embedding model
├── utils.py                # Preprocessing & similarity utilities
├── build_voiceprint.py     # Enrollment pipeline
├── record_samples.py       # Structured audio recording
├── convert_audio.py        # Audio format standardization
├── test.py                 # Single-sample real-time verification
├── test2.py                # Multi-trial stabilized verification
├── requirements.txt        # Dependencies
├── .gitignore              # Ignore unnecessary files
└── README.md               # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/pavanghorpade60/Real-time-speaker-verification-ECAPA.git
cd Real-time-speaker-verification-ECAPA
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎤 Enrollment Process

### Step 1 – Record Samples

```bash
python record_samples.py
```

This records multiple 16kHz mono samples for enrollment.

---

### Step 2 – Convert Audio (If Required)

```bash
python convert_audio.py
```

Standardizes audio to 16kHz mono WAV format.

---

### Step 3 – Build Enrollment Embeddings

```bash
python build_voiceprint.py
```

This:
- Extracts embeddings using ECAPA-TDNN
- Normalizes embeddings
- Saves them as `pavan_embeddings.pt`

---

##  Real-Time Verification

### Single Sample Mode

```bash
python test.py
```

Example Output:

```
🔎 Audio Energy: 0.0832
📊 Average Similarity: 0.7814
📊 Max Similarity:     0.8429
🗣️ Speaker: Pavan
```

---

### Multi-Trial Stabilized Mode

```bash
python test2.py
```

Example Output:

```
📊 Average Similarity: 0.8035
📊 Max Similarity:     0.8712
🗣️ Speaker: Pavan
```

Multi-trial averaging reduces embedding variance and improves robustness.

---

##  Similarity Interpretation

Cosine similarity range:

- Same Speaker → ~0.65 to 0.90  
- Different Speaker → ~0.20 to 0.50  

Threshold is empirically calibrated (≈ 0.50–0.52) to balance:

- False Acceptance Rate (FAR)  
- False Rejection Rate (FRR)

---

## 🛠 Engineering Highlights

- GPU-aware inference  
- Deterministic preprocessing  
- Defensive amplitude normalization  
- Signal energy validation  
- Multi-sample enrollment strategy  
- Modular and extensible architecture  

---

##  Future Improvements

- Equal Error Rate (EER) evaluation  
- ROC curve visualization  
- Multi-speaker database support  
- REST API deployment  
- Streaming real-time inference  
- Voice Activity Detection (VAD)  
- Adaptive thresholding  

---

##  Use Cases

- Biometric authentication  
- Secure voice access systems  
- Personalized voice assistants  
- Voice-controlled applications  
- Speaker verification research  

---

## 📄 License

MIT License

---

##  Author

**Pavan Ghorpade**  
Machine Learning Engineer | Speech & Audio Processing  

---