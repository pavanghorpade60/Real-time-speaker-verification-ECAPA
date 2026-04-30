import os
import glob
import numpy as np
import librosa
import torch
import io
import contextlib
import warnings
import logging

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from speechbrain.pretrained import EncoderClassifier

print("\nBuilding MULTI-REFERENCE Vivek Speaker Embeddings...")

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models",
        run_opts={"device": DEVICE}
    )

# -------------------------------------------------
# PATH
# -------------------------------------------------
SPEAKER_NAME = "VIVEK"
AUDIO_FOLDER = "recordings/vivek"
OUTPUT_FILE = "vivek_reference_embeddings.pt"

audio_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.wav")))

print("Samples found:", len(audio_files))

if len(audio_files) == 0:
    print("No audio samples found in recordings/vivek")
    raise SystemExit

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
TARGET_SR = 16000
MIN_DURATION = 1.5
MIN_PEAK = 0.05
MIN_RMS = 0.01
TOP_DB = 25

KEEP_PERCENT = 0.70
MIN_KEEP_COUNT = 6
MAX_KEEP_COUNT = 12

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def prepare_audio(audio: np.ndarray, sr: int):
    if audio is None or len(audio) == 0:
        return None

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    audio = np.asarray(audio, dtype=np.float32)

    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)
    if len(audio) == 0:
        return None

    duration = len(audio) / TARGET_SR
    if duration < MIN_DURATION:
        return None

    audio = librosa.util.normalize(audio)

    peak = float(np.max(np.abs(audio)))
    if peak < MIN_PEAK:
        return None

    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    if rms < MIN_RMS:
        return None

    return np.ascontiguousarray(audio, dtype=np.float32)


def extract_embedding(audio_np: np.ndarray) -> np.ndarray:
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = classifier.encode_batch(waveform).squeeze().detach().cpu().numpy()
    return l2_normalize(emb)


# -------------------------------------------------
# STEP 1: CLEAN + EXTRACT PER-SAMPLE EMBEDDINGS
# -------------------------------------------------
embeddings = []
valid_files = []
sample_info = []

for file in audio_files:
    try:
        audio, sr = librosa.load(file, sr=TARGET_SR)
        prepared = prepare_audio(audio, sr)

        if prepared is None:
            print(f"Skipped weak/short file: {os.path.basename(file)}")
            continue

        duration = len(prepared) / TARGET_SR
        peak = float(np.max(np.abs(prepared)))
        rms = float(np.sqrt(np.mean(prepared ** 2) + 1e-12))

        emb = extract_embedding(prepared)

        embeddings.append(emb)
        valid_files.append(file)
        sample_info.append({
            "file": file,
            "duration": duration,
            "peak": peak,
            "rms": rms
        })

        print(f"Used: {os.path.basename(file)} | dur={duration:.2f}s | rms={rms:.4f}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

print(f"\nValid samples used: {len(embeddings)}")

if len(embeddings) < MIN_KEEP_COUNT:
    print("Not enough good samples. Record more clean samples.")
    raise SystemExit

embeddings = np.asarray(embeddings, dtype=np.float32)

# -------------------------------------------------
# STEP 2: BUILD ROBUST CENTER
# -------------------------------------------------
initial_center = l2_normalize(np.mean(embeddings, axis=0))

# -------------------------------------------------
# STEP 3: SCORE EACH SAMPLE AGAINST CENTER
# -------------------------------------------------
scores = np.asarray(
    [cosine_similarity(e, initial_center) for e in embeddings],
    dtype=np.float32
)

sorted_idx = np.argsort(scores)[::-1]

keep_count = int(len(embeddings) * KEEP_PERCENT)
keep_count = max(MIN_KEEP_COUNT, keep_count)
keep_count = min(MAX_KEEP_COUNT, keep_count)
keep_count = min(keep_count, len(embeddings))

selected_idx = sorted_idx[:keep_count]

selected_embeddings = embeddings[selected_idx]
selected_files = [valid_files[i] for i in selected_idx]
selected_scores = [float(scores[i]) for i in selected_idx]

print(f"Selected strong reference embeddings: {len(selected_embeddings)}")
print(f"Removed weaker/outlier samples: {len(embeddings) - len(selected_embeddings)}")

# -------------------------------------------------
# STEP 4: SAVE MULTI-REFERENCE EMBEDDINGS
# -------------------------------------------------
payload = {
    "speaker": SPEAKER_NAME,
    "embeddings": torch.tensor(selected_embeddings, dtype=torch.float32),
    "files": selected_files,
    "scores_to_center": selected_scores,
    "num_embeddings": len(selected_embeddings),
    "center_embedding": torch.tensor(
        l2_normalize(np.mean(selected_embeddings, axis=0)),
        dtype=torch.float32
    ),
}

torch.save(payload, OUTPUT_FILE)

print(f"\n{SPEAKER_NAME} reference embeddings saved successfully!")
print(f"Output file: {OUTPUT_FILE}")

# -------------------------------------------------
# STEP 5: DEBUG SUMMARY
# -------------------------------------------------
best_score = float(np.max(scores))
worst_score = float(np.min(scores))
avg_score = float(np.mean(scores))

print("\nEmbedding quality summary:")
print(f"Best sample similarity   : {best_score:.4f}")
print(f"Average sample similarity: {avg_score:.4f}")
print(f"Worst sample similarity  : {worst_score:.4f}")

print("\nSelected files:")
for rank, i in enumerate(selected_idx, start=1):
    print(f"{rank}. {os.path.basename(valid_files[i])} | sim={scores[i]:.4f}")