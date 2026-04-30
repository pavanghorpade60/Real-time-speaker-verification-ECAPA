import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 6
NUM_SAMPLES = 50
CHANNELS = 1
DTYPE = "float32"

MIC_DEVICE = None

MIN_RMS_ENERGY = 0.008
MIN_DURATION_AFTER_TRIM = 2.0
TRIM_THRESHOLD = 0.01
CLIP_THRESHOLD = 0.98

OUTPUT_FOLDER = "recordings/vivek"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"\nRecording {NUM_SAMPLES} clean samples for VIVEK...\n")


def compute_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2) + 1e-10))


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float = TRIM_THRESHOLD) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])
    return audio[start:end]


def is_clipped(audio: np.ndarray, threshold: float = CLIP_THRESHOLD) -> bool:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return False
    return bool(np.max(np.abs(audio)) >= threshold)


def validate_raw_audio(audio: np.ndarray):
    rms = compute_rms(audio)

    if rms < MIN_RMS_ENERGY:
        return False, "Low energy / unclear audio", rms

    if is_clipped(audio):
        return False, "Audio clipped (too loud / too close to mic)", rms

    return True, "OK", rms


if MIC_DEVICE is not None:
    sd.default.device = MIC_DEVICE

i = 1
while i <= NUM_SAMPLES:
    input(f"\nPress ENTER to record VIVEK sample {i}/{NUM_SAMPLES}")

    print("Recording starts in 2 seconds...")
    time.sleep(2)

    print("Speak clearly now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=MIC_DEVICE
    )
    sd.wait()

    audio = audio.flatten().astype(np.float32)

    valid, reason, rms = validate_raw_audio(audio)
    print(f"RMS Energy: {rms:.6f}")

    if not valid:
        print(f"{reason}. Retry.")
        continue

    trimmed = trim_silence(audio, threshold=TRIM_THRESHOLD)

    trimmed_duration = len(trimmed) / SAMPLE_RATE
    if trimmed_duration < MIN_DURATION_AFTER_TRIM:
        print("Too short after trimming. Retry.")
        continue

    trimmed = normalize_audio(trimmed)

    file_path = os.path.join(OUTPUT_FOLDER, f"vivek_{i:02d}.wav")
    sf.write(file_path, trimmed, SAMPLE_RATE)

    print(f"Saved: vivek_{i:02d}.wav | Duration: {trimmed_duration:.2f}s")
    i += 1

print(f"\nAll {NUM_SAMPLES} VIVEK samples recorded successfully!")