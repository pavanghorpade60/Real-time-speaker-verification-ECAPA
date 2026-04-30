import os
import shutil
import warnings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa

from speechbrain.pretrained import EncoderClassifier

# -------------------------------------------------
# CLEAN WARNINGS
# -------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

print("\nBuilding STRONG Virat Speaker Embeddings...")

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# Input cleaned Virat audio (voice-only)
INPUT_AUDIO_FILE = "virat_audio/virat_uvr_audio.wav"

# Output
CHUNKS_FOLDER = "virat_chunks"
MASTER_EMBEDDING_FILE = "virat_master_embedding.pt"
STRONG_EMBEDDING_FILE = "virat_strong_embedding.pt"

# Chunking
CHUNK_DURATION_SEC = 4.0
CHUNK_HOP_SEC = 2.0
MIN_CHUNK_DURATION_SEC = 2.5

# Quality thresholds
MIN_RMS_ENERGY = 0.010
MIN_VOICE_RATIO = 0.55
MAX_SPECTRAL_FLATNESS = 0.20
TOP_DB = 24

# Final selection
MIN_KEEP_CHUNKS = 30
MAX_KEEP_CHUNKS = 80
STRONG_KEEP_RATIO = 0.45   # strongest 45% of final selected chunks

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print(f"Using device: {DEVICE}")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.abspath("pretrained_models/spkrec-ecapa-voxceleb"),
    run_opts={"device": DEVICE}
)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def l2_normalize_torch(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.norm(x, p=2) + 1e-10)


def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    return audio, target_sr


def save_audio(audio_np, sr, out_path):
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    sf.write(out_path, audio_np, sr)


def normalize_audio(audio_np):
    peak = np.max(np.abs(audio_np)) + 1e-8
    return (audio_np / peak).astype(np.float32)


def trim_silence(audio_np, top_db=24):
    trimmed, _ = librosa.effects.trim(audio_np, top_db=top_db)
    if len(trimmed) == 0:
        return audio_np.astype(np.float32)
    return trimmed.astype(np.float32)


def compute_rms(audio_np):
    return float(np.sqrt(np.mean(audio_np ** 2) + 1e-10))


def estimate_voice_ratio(audio_np, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(
        y=audio_np,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=audio_np,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    if len(rms) == 0:
        return 0.0

    rms_thresh = max(0.18 * np.max(rms), 0.004)
    voiced = (rms > rms_thresh) & (zcr > 0.005) & (zcr < 0.22)

    return float(np.mean(voiced))


def spectral_flatness_score(audio_np):
    flatness = librosa.feature.spectral_flatness(y=audio_np)[0]
    if len(flatness) == 0:
        return 1.0
    return float(np.mean(flatness))


def chunk_quality_score(audio_np, sr):
    rms = compute_rms(audio_np)
    voice_ratio = estimate_voice_ratio(audio_np, sr)
    flatness = spectral_flatness_score(audio_np)

    rms_score = min(rms / 0.06, 1.0)
    voice_score = min(voice_ratio / 0.70, 1.0)
    flatness_score = 1.0 - min(flatness / 0.30, 1.0)

    final_score = (
        0.30 * rms_score +
        0.50 * voice_score +
        0.20 * flatness_score
    )

    return {
        "rms": rms,
        "voice_ratio": voice_ratio,
        "flatness": flatness,
        "final_score": float(final_score)
    }


def is_valid_chunk(metrics):
    if metrics["rms"] < MIN_RMS_ENERGY:
        return False, "low energy"

    if metrics["voice_ratio"] < MIN_VOICE_RATIO:
        return False, "low voice ratio"

    if metrics["flatness"] > MAX_SPECTRAL_FLATNESS:
        return False, "too noisy / music-like"

    return True, "ok"


def extract_embedding_from_audio(audio_np):
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform).squeeze().detach().cpu().float()

    embedding = l2_normalize_torch(embedding)
    return embedding


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception:
                pass
    os.makedirs(folder_path, exist_ok=True)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    if not os.path.exists(INPUT_AUDIO_FILE):
        print(f"Input audio file not found: {INPUT_AUDIO_FILE}")
        return

    clear_folder(CHUNKS_FOLDER)

    print(f"\nLoading audio: {INPUT_AUDIO_FILE}")
    audio_np, sr = load_audio(INPUT_AUDIO_FILE, TARGET_SR)

    audio_np = trim_silence(audio_np, top_db=TOP_DB)
    audio_np = normalize_audio(audio_np)

    total_samples = len(audio_np)
    total_duration_sec = total_samples / sr
    chunk_samples = int(CHUNK_DURATION_SEC * sr)
    hop_samples = int(CHUNK_HOP_SEC * sr)

    if total_samples < chunk_samples:
        print("Audio is too short after preprocessing.")
        return

    num_chunks = 1 + max(0, (total_samples - chunk_samples) // hop_samples)

    print(f"Audio duration after trim: {total_duration_sec:.2f} sec")
    print(f"Creating overlapping chunks: {CHUNK_DURATION_SEC}s window / {CHUNK_HOP_SEC}s hop")
    print(f"Total candidate chunks: {num_chunks}\n")

    candidate_records = []

    for i in range(num_chunks):
        start = i * hop_samples
        end = start + chunk_samples
        chunk = audio_np[start:end]

        if len(chunk) == 0:
            continue

        chunk = trim_silence(chunk, top_db=TOP_DB)

        chunk_duration = len(chunk) / sr
        if chunk_duration < MIN_CHUNK_DURATION_SEC:
            continue

        chunk = normalize_audio(chunk)
        metrics = chunk_quality_score(chunk, sr)
        valid, reason = is_valid_chunk(metrics)

        if not valid:
            print(
                f"Rejected chunk {i+1:03d}: {reason} | "
                f"RMS={metrics['rms']:.6f}, "
                f"VoiceRatio={metrics['voice_ratio']:.3f}, "
                f"Flatness={metrics['flatness']:.3f}, "
                f"Score={metrics['final_score']:.3f}"
            )
            continue

        try:
            emb = extract_embedding_from_audio(chunk)

            candidate_records.append({
                "index": i + 1,
                "audio": chunk,
                "embedding": emb,
                "rms": metrics["rms"],
                "voice_ratio": metrics["voice_ratio"],
                "flatness": metrics["flatness"],
                "score": metrics["final_score"]
            })

            print(
                f"Accepted chunk {i+1:03d} | "
                f"Dur={chunk_duration:.2f}s | "
                f"RMS={metrics['rms']:.6f} | "
                f"VoiceRatio={metrics['voice_ratio']:.3f} | "
                f"Flatness={metrics['flatness']:.3f} | "
                f"Score={metrics['final_score']:.3f}"
            )
        except Exception as e:
            print(f"Failed chunk {i+1:03d}: {e}")

    if len(candidate_records) == 0:
        print("\nNo valid chunks found.")
        return

    # -------------------------------------------------
    # STEP 2: INITIAL RANKING BY QUALITY
    # -------------------------------------------------
    candidate_records = sorted(
        candidate_records,
        key=lambda x: (
            x["score"],
            x["voice_ratio"],
            x["rms"],
            -x["flatness"]
        ),
        reverse=True
    )

    preselect_count = min(max(MAX_KEEP_CHUNKS + 20, 100), len(candidate_records))
    preselected = candidate_records[:preselect_count]

    print(f"\nInitial high-quality pool selected: {len(preselected)}")

    # -------------------------------------------------
    # STEP 3: EMBEDDING OUTLIER REMOVAL
    # -------------------------------------------------
    emb_stack = torch.stack([x["embedding"] for x in preselected])
    mean_emb = l2_normalize_torch(torch.mean(emb_stack, dim=0))

    for record in preselected:
        cosine_sim = torch.dot(record["embedding"], mean_emb).item()
        record["center_similarity"] = float(cosine_sim)

    similarity_values = np.array([x["center_similarity"] for x in preselected], dtype=np.float32)
    similarity_threshold = np.percentile(similarity_values, 25)

    filtered = []
    for record in preselected:
        if record["center_similarity"] >= similarity_threshold:
            filtered.append(record)

    print(f"Outliers removed after embedding check: {len(preselected) - len(filtered)}")
    print(f"Remaining after outlier removal: {len(filtered)}")

    if len(filtered) < MIN_KEEP_CHUNKS:
        print("\nToo few chunks after outlier filtering.")
        print("Try relaxing thresholds slightly or use a cleaner Virat source.")
        return

    # -------------------------------------------------
    # STEP 4: FINAL SELECT BEST CHUNKS
    # -------------------------------------------------
    filtered = sorted(
        filtered,
        key=lambda x: (
            x["score"],
            x["center_similarity"],
            x["voice_ratio"],
            x["rms"],
            -x["flatness"]
        ),
        reverse=True
    )

    final_records = filtered[:min(MAX_KEEP_CHUNKS, len(filtered))]

    if len(final_records) < MIN_KEEP_CHUNKS:
        print(f"\nOnly {len(final_records)} chunks available, which is below required minimum {MIN_KEEP_CHUNKS}.")
        return

    # -------------------------------------------------
    # STEP 5: SAVE FINAL CHUNKS
    # -------------------------------------------------
    for idx, record in enumerate(final_records, start=1):
        chunk_file = os.path.join(CHUNKS_FOLDER, f"virat_chunk_{idx:03d}.wav")
        save_audio(record["audio"], sr, chunk_file)
        record["chunk_file"] = chunk_file

    # -------------------------------------------------
    # STEP 6: BUILD VIRAT MASTER EMBEDDING
    # -------------------------------------------------
    final_embeddings = torch.stack([x["embedding"] for x in final_records])

    weights = []
    for x in final_records:
        weight = (
            0.40 * x["score"] +
            0.30 * min(x["voice_ratio"], 1.0) +
            0.15 * min(x["rms"] / 0.08, 1.0) +
            0.15 * min(max(x["center_similarity"], 0.0), 1.0)
        )
        weights.append(max(weight, 0.10))

    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    weighted_embedding = torch.sum(final_embeddings * weights, dim=0) / torch.sum(weights)
    virat_master_embedding = l2_normalize_torch(weighted_embedding)

    # -------------------------------------------------
    # STEP 7: BUILD VIRAT STRONG EMBEDDING
    # -------------------------------------------------
    strong_keep_count = max(12, int(len(final_records) * STRONG_KEEP_RATIO))

    strong_records = sorted(
        final_records,
        key=lambda x: (
            x["score"],
            x["center_similarity"],
            x["voice_ratio"],
            x["rms"]
        ),
        reverse=True
    )[:strong_keep_count]

    strong_embeddings = torch.stack([x["embedding"] for x in strong_records])

    strong_weights = []
    for x in strong_records:
        weight = (
            0.45 * x["score"] +
            0.30 * min(x["center_similarity"], 1.0) +
            0.15 * min(x["voice_ratio"], 1.0) +
            0.10 * min(x["rms"] / 0.08, 1.0)
        )
        strong_weights.append(max(weight, 0.10))

    strong_weights = torch.tensor(strong_weights, dtype=torch.float32).unsqueeze(1)
    strong_weighted_embedding = torch.sum(strong_embeddings * strong_weights, dim=0) / torch.sum(strong_weights)
    virat_strong_embedding = l2_normalize_torch(strong_weighted_embedding)

    # -------------------------------------------------
    # STEP 8: SAVE EMBEDDINGS
    # -------------------------------------------------
    torch.save(virat_master_embedding, MASTER_EMBEDDING_FILE)
    torch.save(virat_strong_embedding, STRONG_EMBEDDING_FILE)

    # -------------------------------------------------
    # DONE
    # -------------------------------------------------
    print("\n--------------------------------------------------")
    print("Strong Virat embeddings created successfully")
    print(f"Master embedding       : {MASTER_EMBEDDING_FILE}")
    print(f"Strong embedding       : {STRONG_EMBEDDING_FILE}")
    print(f"Final selected chunks  : {len(final_records)}")
    print(f"Strong chunks used     : {len(strong_records)}")
    print(f"Chunks saved in folder : {CHUNKS_FOLDER}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()