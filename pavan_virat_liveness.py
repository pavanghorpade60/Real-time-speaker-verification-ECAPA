import os
import io
import sys
import time
import warnings
import logging
import contextlib
import ctypes
from collections import deque, Counter

# -------------------------------------------------
# CLEAN CONSOLE / WARNINGS
# -------------------------------------------------
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)

import torch
import sounddevice as sd
import numpy as np
import librosa

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from speechbrain.pretrained import EncoderClassifier

# -------------------------------------------------
# WINDOWS CMD COLOR SUPPORT
# -------------------------------------------------
def enable_windows_ansi():
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle == 0:
            return
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            enable_vt = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | enable_vt)
    except Exception:
        pass


enable_windows_ansi()

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m}"

# Fix typo if accidentally used above
MAGENTA = "\033[95m"

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECAPA_CACHE_DIR = os.path.join(BASE_DIR, "ecapa_cache")

PAVAN_MASTER_FILE = os.path.join(BASE_DIR, "pavan_master_embedding.pt")
PAVAN_STRONG_FILE = os.path.join(BASE_DIR, "pavan_strong_embedding.pt")
VIRAT_MASTER_FILE = os.path.join(BASE_DIR, "virat_master_embedding.pt")
VIRAT_STRONG_FILE = os.path.join(BASE_DIR, "virat_strong_embedding.pt")

os.makedirs(ECAPA_CACHE_DIR, exist_ok=True)

# -------------------------------------------------
# AUDIO SETTINGS
# -------------------------------------------------
MIC_DEVICE = 1
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1
DTYPE = "float32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# SPEECH / SEGMENT SETTINGS
# -------------------------------------------------
START_ENERGY_THRESHOLD = 0.0038
CONTINUE_ENERGY_THRESHOLD = 0.0028

SILENCE_LIMIT = 2.40
MIN_DECISION_AUDIO_SECONDS = 0.28
MIN_PREPARED_AUDIO_SECONDS = 0.22

PREBUFFER_CHUNKS = 10
LIVE_UPDATE_INTERVAL = 0.06
LIVE_WINDOWS = [0.40, 0.70, 1.00, 1.35]

VOICE_CHUNKS_TO_START = 1
SILENT_CHUNKS_TO_STOP = int(SILENCE_LIMIT / (BLOCK_SIZE / SAMPLE_RATE)) + 1
RESET_LIVE_BUFFER_SILENT_CHUNKS = 12

# -------------------------------------------------
# THRESHOLDS
# -------------------------------------------------
START_THRESHOLDS = {
    "PAVAN": 0.36,
    "VIRAT KOHLI": 0.30,
}

KEEP_THRESHOLDS = {
    "PAVAN": 0.32,
    "VIRAT KOHLI": 0.24,
}

SWITCH_THRESHOLDS = {
    "PAVAN": 0.47,
    "VIRAT KOHLI": 0.33,
}

START_MARGIN = 0.020
SWITCH_MARGIN = 0.030

UNKNOWN_SCORE_FLOOR = 0.21

INITIAL_CONFIRMATIONS = 1
SWITCH_CONFIRMATIONS = 2
UNKNOWN_CONFIRMATIONS = 4

HISTORY_SIZE = 6
MIN_HISTORY_VOTE = 2

FAST_SWITCH_EXTRA_GAP = 0.090
BAD_FRAMES_TO_RELEASE = 7
LABEL_HOLD_SECONDS = 1.35

SMOOTHING_ALPHA = 0.42

# -------------------------------------------------
# VIRAT PLAYBACK STABILITY
# -------------------------------------------------
VIRAT_STICKY_HOLD_SECONDS = 2.80
VIRAT_RELEASE_MIN_SCORE = 0.20
VIRAT_RELEASE_MARGIN = -0.035
PAVAN_TAKEOVER_FROM_VIRAT_SCORE = 0.52
PAVAN_TAKEOVER_FROM_VIRAT_GAP = 0.16

# -------------------------------------------------
# MIXED / OVERLAP
# -------------------------------------------------
OVERLAP_MARGIN = 0.020

# -------------------------------------------------
# AUDIO QUALITY SETTINGS
# -------------------------------------------------
TRIM_TOP_DB = 20
TARGET_PEAK = 0.95
MIN_AUDIO_RMS = 0.008
MAX_ANALYSIS_SECONDS = 1.8

# -------------------------------------------------
# SOURCE TYPE (LIVE / PLAYBACK)
# FIX: much stronger playback lock, slower LIVE recovery
# -------------------------------------------------
SOURCE_BUFFER_SECONDS = 3.8
SOURCE_WINDOWS = [1.6, 2.4, 3.4]
SOURCE_SMOOTHING_ALPHA = 0.92

PLAYBACK_ENTER_THRESHOLD = 0.55
PLAYBACK_STAY_THRESHOLD = 0.30
LIVE_ENTER_THRESHOLD = 0.10

SOURCE_CONFIRMATIONS = 2
SOURCE_SWITCH_CONFIRMATIONS = 8
PLAYBACK_HOLD_SECONDS = 10.0
LIVE_HOLD_SECONDS = 2.5

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def color_for_label(label):
    if label == "PAVAN":
        return GREEN
    if label == "VIRAT KOHLI":
        return RED
    if label == "UNKNOWN":
        return YELLOW
    if label == "MIXED":
        return CYAN
    return RESET


def source_color(source_label):
    if source_label == "LIVE":
        return GREEN
    if source_label == "PLAYBACK":
        return MAGENTA
    return YELLOW


def speaker_line(label, source_label=None, confidence=None):
    if confidence is None:
        confidence_text = ""
    else:
        confidence_text = f" ({confidence}%)"

    source_text = ""
    if source_label is not None:
        source_text = f" | SOURCE={source_color(source_label)}{source_label}{RESET}"

    if label == "PAVAN":
        return f"{GREEN}Pavan is speaking{confidence_text}...{RESET}{source_text}"
    if label == "VIRAT KOHLI":
        return f"{RED}Virat Kohli is speaking{confidence_text}...{RESET}{source_text}"
    if label == "MIXED":
        return f"{CYAN}Mixed / overlapping speech detected{confidence_text}...{RESET}{source_text}"
    return f"{YELLOW}Unknown speaker is speaking{confidence_text}...{RESET}{source_text}"


def compute_confidence(label, scores):
    p = float(scores.get("PAVAN", 0.0))
    v = float(scores.get("VIRAT KOHLI", 0.0))

    best = max(p, v)
    second = min(p, v)
    margin = best - second

    if label == "UNKNOWN":
        if best < 0.21:
            return 95
        if best < 0.28:
            return 88
        if best < 0.34:
            return 78
        return 68

    if label == "MIXED":
        closeness = 1.0 - min(abs(p - v) / 0.08, 1.0)
        mixed_conf = 60 + int(closeness * 35)
        return max(60, min(95, mixed_conf))

    if label in ("PAVAN", "VIRAT KOHLI"):
        score_conf = min(max((best - 0.30) / 0.35, 0.0), 1.0)
        margin_conf = min(max(margin / 0.20, 0.0), 1.0)
        conf = int(55 + 30 * score_conf + 15 * margin_conf)
        return max(55, min(99, conf))

    return 70


def emit_label_if_changed(label, ui_state, scores=None, source_label=None):
    if label is None:
        return

    confidence = None
    if scores is not None:
        confidence = compute_confidence(label, scores)

    display_signature = (label, confidence, source_label)

    if ui_state["last_printed_label"] != display_signature:
        print(speaker_line(label, source_label, confidence))
        ui_state["last_printed_label"] = display_signature


def print_live_scores(scores, best_label, best_score, margin, source_label=None, source_score=None):
    best_color = color_for_label(best_label)

    if source_label is None:
        source_text = " | SOURCE=UNCERTAIN"
    else:
        if source_score is not None:
            source_text = (
                f" | SOURCE={source_color(source_label)}{source_label}{RESET}"
                f" {source_score:.2f}"
            )
        else:
            source_text = f" | SOURCE={source_color(source_label)}{source_label}{RESET}"

    print(
        f"[LIVE] "
        f"PAVAN={scores['PAVAN']:.4f} | "
        f"VIRAT={scores['VIRAT KOHLI']:.4f} | "
        f"BEST={best_color}{best_label} {best_score:.4f}{RESET} | "
        f"MARGIN={margin:.4f}"
        f"{source_text}"
    )


def l2_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float().flatten()
    norm = torch.norm(x, p=2)
    if norm.item() == 0:
        return x
    return x / norm


def load_single_embedding(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        sys.exit()
    emb = torch.load(path, map_location="cpu")
    return l2_normalize_tensor(emb)


def load_reference_embeddings():
    return {
        "PAVAN": {
            "master": load_single_embedding(PAVAN_MASTER_FILE),
            "strong": load_single_embedding(PAVAN_STRONG_FILE),
        },
        "VIRAT KOHLI": {
            "master": load_single_embedding(VIRAT_MASTER_FILE),
            "strong": load_single_embedding(VIRAT_STRONG_FILE),
        },
    }


def safe_flatten_mono(audio_np) -> np.ndarray:
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim > 1:
        if audio_np.shape[1] == 1:
            audio_np = audio_np[:, 0]
        else:
            audio_np = np.mean(audio_np, axis=1)
    return np.ascontiguousarray(audio_np.flatten(), dtype=np.float32)


def safe_concat_chunks(chunks) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(
        np.concatenate([np.asarray(c, dtype=np.float32).copy() for c in chunks]),
        dtype=np.float32,
    )


def chunk_rms(chunk: np.ndarray) -> float:
    chunk = np.asarray(chunk, dtype=np.float32).flatten()
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))


def chunk_zero_crossing_rate(chunk: np.ndarray) -> float:
    chunk = np.asarray(chunk, dtype=np.float32).flatten()
    if len(chunk) < 2:
        return 0.0
    frame_length = min(len(chunk), 512)
    hop_length = max(1, min(len(chunk) // 2, 256))
    zcr = librosa.feature.zero_crossing_rate(
        y=chunk,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    return float(np.mean(zcr))


def chunk_spectral_flatness(chunk: np.ndarray) -> float:
    chunk = np.asarray(chunk, dtype=np.float32).flatten()
    if len(chunk) < 64:
        return 0.0
    try:
        return float(np.mean(librosa.feature.spectral_flatness(y=chunk)))
    except Exception:
        return 0.0


def is_voice_chunk(chunk: np.ndarray, is_currently_speaking: bool) -> bool:
    energy = chunk_rms(chunk)
    zcr = chunk_zero_crossing_rate(chunk)
    flatness = chunk_spectral_flatness(chunk)

    threshold = CONTINUE_ENERGY_THRESHOLD if is_currently_speaking else START_ENERGY_THRESHOLD

    if energy < threshold:
        return False

    if zcr < 0.0008 or zcr > 0.45:
        return False

    if flatness > 0.55:
        return False

    return True


def audio_rms(audio_np: np.ndarray) -> float:
    audio_np = np.asarray(audio_np, dtype=np.float32).flatten()
    if len(audio_np) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_np)) + 1e-12))


def voiced_chunk_count(chunks) -> int:
    count = 0
    for c in chunks:
        if chunk_rms(c) > CONTINUE_ENERGY_THRESHOLD:
            count += 1
    return count


def voiced_ratio(chunks) -> float:
    if not chunks:
        return 0.0
    return voiced_chunk_count(chunks) / len(chunks)


def prepare_audio(audio_np):
    if audio_np is None or len(audio_np) == 0:
        return None

    audio_np = safe_flatten_mono(audio_np)
    if len(audio_np) == 0:
        return None

    peak = np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0.0
    if peak < 1e-4:
        return None

    audio_np, _ = librosa.effects.trim(audio_np, top_db=TRIM_TOP_DB)
    if len(audio_np) == 0:
        return None

    max_len = int(MAX_ANALYSIS_SECONDS * SAMPLE_RATE)
    if len(audio_np) > max_len:
        audio_np = audio_np[-max_len:]

    duration_sec = len(audio_np) / SAMPLE_RATE
    if duration_sec < MIN_PREPARED_AUDIO_SECONDS:
        return None

    audio_np = audio_np - np.mean(audio_np)

    if len(audio_np) >= 2:
        audio_np = np.append(audio_np[0], audio_np[1:] - 0.97 * audio_np[:-1])

    rms = np.sqrt(np.mean(audio_np ** 2) + 1e-12)
    if rms < 0.008:
        return None

    audio_np = audio_np / rms

    peak = np.max(np.abs(audio_np))
    if peak > 1e-6:
        audio_np = TARGET_PEAK * (audio_np / peak)

    audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

    final_rms = audio_rms(audio_np)
    if final_rms < MIN_AUDIO_RMS:
        return None

    return np.ascontiguousarray(audio_np, dtype=np.float32)


def is_valid_audio_for_scoring(audio_np):
    if audio_np is None or len(audio_np) == 0:
        return False

    rms = np.sqrt(np.mean(audio_np ** 2) + 1e-12)
    if rms < 0.012:
        return False

    peak = np.max(np.abs(audio_np))
    if peak < 0.08:
        return False

    return True


def create_input_embedding(classifier, audio_np):
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform).squeeze()
    return l2_normalize_tensor(embedding)


def cosine_score(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()


def speaker_fused_score(input_emb, refs_for_speaker):
    master_score = cosine_score(input_emb, refs_for_speaker["master"])
    strong_score = cosine_score(input_emb, refs_for_speaker["strong"])

    best = max(master_score, strong_score)
    mean_score = (master_score + strong_score) / 2.0
    consistency_bonus = 0.015 if min(master_score, strong_score) >= 0.30 else 0.0

    fused = 0.78 * best + 0.22 * mean_score + consistency_bonus
    fused = float(min(fused, 1.0))

    return {
        "master": float(master_score),
        "strong": float(strong_score),
        "fused": fused,
    }


def score_single_window(classifier, reference_embeddings, audio_np):
    prepared = prepare_audio(audio_np)
    if prepared is None:
        return None

    emb = create_input_embedding(classifier, prepared)

    pavan_scores = speaker_fused_score(emb, reference_embeddings["PAVAN"])
    virat_scores = speaker_fused_score(emb, reference_embeddings["VIRAT KOHLI"])

    return {
        "PAVAN": pavan_scores["fused"],
        "VIRAT KOHLI": virat_scores["fused"],
    }


def blend_scores(prev_scores, new_scores, alpha=SMOOTHING_ALPHA):
    if prev_scores is None:
        return new_scores.copy()

    out = {}
    for k in ("PAVAN", "VIRAT KOHLI"):
        out[k] = float(alpha * prev_scores.get(k, 0.0) + (1.0 - alpha) * new_scores.get(k, 0.0))
    return out


def score_multiwindow(classifier, reference_embeddings, live_chunks):
    if len(live_chunks) == 0:
        return None

    total_audio_len = sum(len(c) for c in live_chunks)
    total_audio_sec = total_audio_len / SAMPLE_RATE

    if total_audio_sec < MIN_DECISION_AUDIO_SECONDS:
        return None

    if voiced_ratio(live_chunks) < 0.12:
        return None

    full_audio = safe_concat_chunks(list(live_chunks))

    if audio_rms(full_audio) < 0.006:
        return None

    prepared_full = prepare_audio(full_audio)
    if prepared_full is None:
        return None

    if not is_valid_audio_for_scoring(prepared_full):
        return None

    scores_list = []

    for win_sec in LIVE_WINDOWS:
        target_len = int(win_sec * SAMPLE_RATE)
        if len(prepared_full) >= target_len:
            win_audio = prepared_full[-target_len:]
        else:
            win_audio = prepared_full

        result = score_single_window(classifier, reference_embeddings, win_audio)
        if result is not None:
            scores_list.append(result)

    if not scores_list:
        return None

    p_scores = np.array([x["PAVAN"] for x in scores_list], dtype=np.float32)
    v_scores = np.array([x["VIRAT KOHLI"] for x in scores_list], dtype=np.float32)

    return {
        "PAVAN": 0.78 * float(np.max(p_scores)) + 0.22 * float(np.mean(p_scores)),
        "VIRAT KOHLI": 0.78 * float(np.max(v_scores)) + 0.22 * float(np.mean(v_scores)),
    }


def stable_vote(history):
    if len(history) < MIN_HISTORY_VOTE:
        return None

    counts = Counter(history)
    label, count = counts.most_common(1)[0]

    if count >= MIN_HISTORY_VOTE:
        return label
    return None


def get_overlap_aware_label(scores):
    p = float(scores["PAVAN"])
    v = float(scores["VIRAT KOHLI"])

    if p >= v:
        best_label = "PAVAN"
        best_score = p
        second_score = v
    else:
        best_label = "VIRAT KOHLI"
        best_score = v
        second_score = p

    margin = best_score - second_score

    if best_score < UNKNOWN_SCORE_FLOOR:
        return "UNKNOWN", best_score, margin

    if p >= 0.38 and v >= 0.34 and margin < OVERLAP_MARGIN:
        return "MIXED", best_score, margin

    if best_label == "PAVAN":
        if best_score >= START_THRESHOLDS["PAVAN"] and margin >= START_MARGIN:
            return "PAVAN", best_score, margin
        return "UNKNOWN", best_score, margin

    if best_label == "VIRAT KOHLI":
        if best_score >= START_THRESHOLDS["VIRAT KOHLI"] and margin >= START_MARGIN:
            return "VIRAT KOHLI", best_score, margin
        return "UNKNOWN", best_score, margin

    return "UNKNOWN", best_score, margin


def should_accept_start(label, label_score, margin):
    if label not in START_THRESHOLDS:
        return False

    if label_score >= START_THRESHOLDS[label] and margin >= START_MARGIN:
        return True

    if label == "VIRAT KOHLI" and label_score >= 0.29 and margin >= 0.025:
        return True

    return False


def should_accept_switch(label, label_score, margin, scores, current_label=None):
    if label not in SWITCH_THRESHOLDS:
        return False

    if label_score < SWITCH_THRESHOLDS[label]:
        return False

    if margin < SWITCH_MARGIN:
        return False

    if current_label in ("PAVAN", "VIRAT KOHLI"):
        current_score_now = float(scores.get(current_label, -1.0))
        if label_score <= current_score_now + 0.015:
            return False

    if current_label == "VIRAT KOHLI" and label == "PAVAN":
        if label_score < PAVAN_TAKEOVER_FROM_VIRAT_SCORE:
            return False
        if margin < PAVAN_TAKEOVER_FROM_VIRAT_GAP:
            return False

    return True


def should_fast_switch(target_label, scores, current_label):
    if current_label not in ("PAVAN", "VIRAT KOHLI"):
        return False
    if target_label not in ("PAVAN", "VIRAT KOHLI"):
        return False
    if target_label == current_label:
        return False

    target_score = float(scores.get(target_label, 0.0))
    current_score = float(scores.get(current_label, 0.0))
    gap = target_score - current_score

    if current_label == "VIRAT KOHLI" and target_label == "PAVAN":
        if target_score < PAVAN_TAKEOVER_FROM_VIRAT_SCORE:
            return False
        if gap < PAVAN_TAKEOVER_FROM_VIRAT_GAP:
            return False

    if target_score < SWITCH_THRESHOLDS[target_label]:
        return False

    if target_label == "VIRAT KOHLI":
        return gap >= FAST_SWITCH_EXTRA_GAP and target_score >= 0.39

    return gap >= FAST_SWITCH_EXTRA_GAP and target_score >= 0.48

# -------------------------------------------------
# SOURCE LAYER
# -------------------------------------------------
def safe_spectral_centroid(audio_np, sr):
    try:
        val = librosa.feature.spectral_centroid(y=audio_np, sr=sr)
        return float(np.mean(val))
    except Exception:
        return 0.0


def safe_spectral_rolloff(audio_np, sr):
    try:
        val = librosa.feature.spectral_rolloff(y=audio_np, sr=sr, roll_percent=0.85)
        return float(np.mean(val))
    except Exception:
        return 0.0


def safe_spectral_flatness(audio_np):
    try:
        val = librosa.feature.spectral_flatness(y=audio_np)
        return float(np.mean(val))
    except Exception:
        return 0.0


def safe_band_energy_ratio(audio_np, sr, low_hz, high_hz):
    try:
        fft = np.fft.rfft(audio_np)
        mag = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(audio_np), d=1.0 / sr)
        total = np.sum(mag) + 1e-12
        band = np.sum(mag[(freqs >= low_hz) & (freqs < high_hz)])
        return float(band / total)
    except Exception:
        return 0.0


def frame_rms_stats(audio_np, frame_length=512, hop_length=256):
    try:
        rms = librosa.feature.rms(y=audio_np, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) == 0:
            return 0.0, 0.0, 0.0
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        cv_rms = float(std_rms / (mean_rms + 1e-9))
        return mean_rms, std_rms, cv_rms
    except Exception:
        return 0.0, 0.0, 0.0


def analyze_source_window(audio_np):
    if audio_np is None or len(audio_np) < int(1.0 * SAMPLE_RATE):
        return None

    x = safe_flatten_mono(audio_np)
    if len(x) == 0:
        return None

    x = x - np.mean(x)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = audio_rms(x)

    if rms < 0.007 or peak < 0.04:
        return None

    x = np.clip(x, -1.0, 1.0)

    centroid = safe_spectral_centroid(x, SAMPLE_RATE)
    rolloff = safe_spectral_rolloff(x, SAMPLE_RATE)
    flatness = safe_spectral_flatness(x)
    zcr = chunk_zero_crossing_rate(x)

    low_ratio = safe_band_energy_ratio(x, SAMPLE_RATE, 80, 1200)
    high_ratio = safe_band_energy_ratio(x, SAMPLE_RATE, 3200, 7600)

    _, _, rms_cv = frame_rms_stats(x)
    clipping_ratio = float(np.mean(np.abs(x) > 0.92))
    crest_factor = float(peak / (rms + 1e-9))
    high_low_ratio = float(high_ratio / (low_ratio + 1e-9))

    playback_evidence = 0.0
    live_evidence = 0.0

    if high_ratio > 0.16:
        playback_evidence += 0.18
    if high_ratio > 0.21:
        playback_evidence += 0.12
    if rolloff > 3000:
        playback_evidence += 0.12
    if rolloff > 3600:
        playback_evidence += 0.08
    if centroid > 1700:
        playback_evidence += 0.08
    if flatness > 0.018:
        playback_evidence += 0.10
    if clipping_ratio > 0.002:
        playback_evidence += 0.10
    if rms_cv < 0.38:
        playback_evidence += 0.12
    if crest_factor < 3.8:
        playback_evidence += 0.08
    if high_low_ratio > 0.42:
        playback_evidence += 0.08
    if zcr > 0.10:
        playback_evidence += 0.04

    if low_ratio > 0.42:
        live_evidence += 0.14
    if high_ratio < 0.13:
        live_evidence += 0.10
    if rolloff < 2700:
        live_evidence += 0.12
    if centroid < 1550:
        live_evidence += 0.08
    if flatness < 0.014:
        live_evidence += 0.10
    if rms_cv > 0.48:
        live_evidence += 0.12
    if crest_factor > 4.6:
        live_evidence += 0.08
    if clipping_ratio < 0.001:
        live_evidence += 0.05
    if high_low_ratio < 0.28:
        live_evidence += 0.06

    total = playback_evidence + live_evidence + 1e-9
    playback_score = float(np.clip(playback_evidence / total, 0.0, 1.0))

    return playback_score


def score_source_multiwindow(source_chunks):
    if len(source_chunks) == 0:
        return None

    full_audio = safe_concat_chunks(list(source_chunks))
    if len(full_audio) < int(1.0 * SAMPLE_RATE):
        return None

    scores = []
    for win_sec in SOURCE_WINDOWS:
        target_len = int(win_sec * SAMPLE_RATE)
        if len(full_audio) >= target_len:
            win_audio = full_audio[-target_len:]
        else:
            win_audio = full_audio

        result = analyze_source_window(win_audio)
        if result is not None:
            scores.append(float(result))

    if not scores:
        return None

    scores_np = np.array(scores, dtype=np.float32)
    fused = 0.62 * float(np.max(scores_np)) + 0.38 * float(np.mean(scores_np))
    return float(np.clip(fused, 0.0, 1.0))


def update_source_state(state, source_chunks):
    fused_playback_score = score_source_multiwindow(source_chunks)
    if fused_playback_score is None:
        return

    prev = state.get("source_score_smoothed")
    if prev is None:
        smoothed = fused_playback_score
    else:
        smoothed = float(
            SOURCE_SMOOTHING_ALPHA * prev +
            (1.0 - SOURCE_SMOOTHING_ALPHA) * fused_playback_score
        )

    state["source_score_smoothed"] = smoothed
    now = time.time()

    current = state["current_source_label"]

    playback_candidate = smoothed >= PLAYBACK_ENTER_THRESHOLD
    strong_live_candidate = smoothed <= LIVE_ENTER_THRESHOLD

    candidate = None

    # CURRENT = PLAYBACK
    if current == "PLAYBACK":
        if now < state["source_hold_until"]:
            return

        if smoothed >= PLAYBACK_STAY_THRESHOLD:
            return

        if strong_live_candidate:
            candidate = "LIVE"
        else:
            return

    # CURRENT = LIVE
    elif current == "LIVE":
        if now < state["source_hold_until"]:
            if not playback_candidate:
                return

        if playback_candidate:
            candidate = "PLAYBACK"
        elif strong_live_candidate:
            return
        else:
            return

    # CURRENT = NONE / startup
    else:
        if playback_candidate:
            candidate = "PLAYBACK"
        elif strong_live_candidate:
            candidate = "LIVE"
        else:
            return

    if candidate == state["pending_source_label"]:
        state["pending_source_count"] += 1
    else:
        state["pending_source_label"] = candidate
        state["pending_source_count"] = 1

    required = SOURCE_CONFIRMATIONS if current is None else SOURCE_SWITCH_CONFIRMATIONS

    if state["pending_source_count"] >= required:
        state["current_source_label"] = candidate

        if candidate == "PLAYBACK":
            state["source_hold_until"] = now + PLAYBACK_HOLD_SECONDS
        else:
            state["source_hold_until"] = now + LIVE_HOLD_SECONDS

        state["pending_source_label"] = None
        state["pending_source_count"] = 0


def reset_runtime_state():
    return {
        "is_speaking": False,
        "silence_time": 0.0,
        "silent_chunk_count": 0,
        "voice_chunk_count": 0,
        "last_live_check_time": 0.0,
        "current_label": None,
        "current_scores": {"PAVAN": 0.0, "VIRAT KOHLI": 0.0},
        "smoothed_scores": None,
        "current_best_score": 0.0,
        "current_margin": 0.0,
        "current_segment_chunks": [],
        "pending_label": None,
        "pending_count": 0,
        "last_live_signature": None,
        "bad_frame_count": 0,
        "label_hold_until": 0.0,
        "current_source_label": None,
        "pending_source_label": None,
        "pending_source_count": 0,
        "source_score_smoothed": None,
        "source_hold_until": 0.0,
    }


def stop_current_segment(state, decision_history, live_chunks, source_chunks, ui_state, reason_text):
    state = reset_runtime_state()
    decision_history.clear()
    live_chunks.clear()
    source_chunks.clear()
    ui_state["last_printed_label"] = None
    print(reason_text)
    return state


# -------------------------------------------------
# MAIN
# -------------------------------------------------
print("\nInitializing Multi-Speaker Identification System...")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_CACHE_DIR,
        run_opts={"device": DEVICE},
    )

reference_embeddings = load_reference_embeddings()

if MIC_DEVICE is not None:
    sd.default.device = (MIC_DEVICE, None)

print("System Ready.")
print("Speak into the microphone. Press Ctrl + C to stop.\n")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    blocksize=BLOCK_SIZE,
    dtype=DTYPE,
    device=MIC_DEVICE,
    latency="low",
)

pre_buffer = deque(maxlen=PREBUFFER_CHUNKS)
max_live_chunks = int((max(LIVE_WINDOWS) * SAMPLE_RATE) / BLOCK_SIZE) + 4
live_chunks = deque(maxlen=max_live_chunks)

max_source_chunks = int((SOURCE_BUFFER_SECONDS * SAMPLE_RATE) / BLOCK_SIZE) + 6
source_chunks = deque(maxlen=max_source_chunks)

state = reset_runtime_state()
decision_history = deque(maxlen=HISTORY_SIZE)
ui_state = {"last_printed_label": None}

chunk_duration = BLOCK_SIZE / SAMPLE_RATE

try:
    stream.start()

    while True:
        audio_chunk, _ = stream.read(BLOCK_SIZE)
        chunk = np.asarray(audio_chunk[:, 0], dtype=np.float32).copy()

        pre_buffer.append(chunk.copy())
        chunk_has_voice = is_voice_chunk(chunk, state["is_speaking"])

        # -------------------------------------------------
        # SILENCE
        # -------------------------------------------------
        if not chunk_has_voice:
            state["voice_chunk_count"] = 0

            if state["is_speaking"]:
                state["silent_chunk_count"] += 1
                state["silence_time"] += chunk_duration

                if state["silent_chunk_count"] <= 6:
                    live_chunks.append(chunk.copy())
                    source_chunks.append(chunk.copy())

                if (
                    state["silence_time"] >= SILENCE_LIMIT
                    or state["silent_chunk_count"] >= SILENT_CHUNKS_TO_STOP
                ):
                    state = stop_current_segment(
                        state,
                        decision_history,
                        live_chunks,
                        source_chunks,
                        ui_state,
                        "[SILENCE DETECTED - STOPPED]"
                    )
            continue

        # -------------------------------------------------
        # VOICE PRESENT
        # -------------------------------------------------
        state["voice_chunk_count"] += 1

        if not state["is_speaking"]:
            state["silent_chunk_count"] = 0
            state["silence_time"] = 0.0

            if state["voice_chunk_count"] >= VOICE_CHUNKS_TO_START:
                state["is_speaking"] = True
                state["current_segment_chunks"] = [c.copy() for c in pre_buffer]

                live_chunks.clear()
                for c in pre_buffer:
                    live_chunks.append(c.copy())

                source_chunks.clear()
                for c in pre_buffer:
                    source_chunks.append(c.copy())

                live_chunks.append(chunk.copy())
                source_chunks.append(chunk.copy())
                state["current_segment_chunks"].append(chunk.copy())

                state["current_label"] = None
                state["current_scores"] = {"PAVAN": 0.0, "VIRAT KOHLI": 0.0}
                state["smoothed_scores"] = None
                state["current_best_score"] = 0.0
                state["current_margin"] = 0.0
                state["pending_label"] = None
                state["pending_count"] = 0
                state["last_live_check_time"] = 0.0
                state["last_live_signature"] = None
                state["bad_frame_count"] = 0
                state["label_hold_until"] = 0.0
                state["current_source_label"] = None
                state["pending_source_label"] = None
                state["pending_source_count"] = 0
                state["source_score_smoothed"] = None
                state["source_hold_until"] = 0.0
                decision_history.clear()
                ui_state["last_printed_label"] = None

                print("[VOICE DETECTED]")

                state["last_live_check_time"] = 0.0

        else:
            state["silent_chunk_count"] = 0
            state["silence_time"] = 0.0

            state["current_segment_chunks"].append(chunk.copy())
            live_chunks.append(chunk.copy())
            source_chunks.append(chunk.copy())

        now = time.time()

        if state["is_speaking"] and (now - state["last_live_check_time"] >= LIVE_UPDATE_INTERVAL):
            raw_scores = score_multiwindow(classifier, reference_embeddings, live_chunks)
            update_source_state(state, source_chunks)

            if raw_scores is not None:
                state["smoothed_scores"] = blend_scores(state["smoothed_scores"], raw_scores, SMOOTHING_ALPHA)
                scores = state["smoothed_scores"].copy()

                best_label, best_score, margin = get_overlap_aware_label(scores)

                current_source = state["current_source_label"]
                source_score_display = state["source_score_smoothed"]

                live_signature = (
                    round(scores["PAVAN"], 4),
                    round(scores["VIRAT KOHLI"], 4),
                    best_label,
                    round(best_score, 4),
                    round(margin, 4),
                    current_source,
                    None if source_score_display is None else round(source_score_display, 2),
                )

                if live_signature != state["last_live_signature"]:
                    print_live_scores(
                        scores,
                        best_label,
                        best_score,
                        margin,
                        source_label=current_source,
                        source_score=source_score_display,
                    )
                    state["last_live_signature"] = live_signature

                decision_history.append(best_label)
                voted_label = stable_vote(decision_history)

                if state["current_label"] is None:
                    if voted_label in ("PAVAN", "VIRAT KOHLI"):
                        voted_score = float(scores.get(voted_label, 0.0))
                        voted_margin = abs(scores["PAVAN"] - scores["VIRAT KOHLI"])

                        if should_accept_start(voted_label, voted_score, voted_margin):
                            if state["pending_label"] == voted_label:
                                state["pending_count"] += 1
                            else:
                                state["pending_label"] = voted_label
                                state["pending_count"] = 1

                            if state["pending_count"] >= INITIAL_CONFIRMATIONS:
                                state["current_label"] = voted_label
                                state["current_scores"] = scores.copy()
                                state["current_best_score"] = voted_score
                                state["current_margin"] = voted_margin
                                state["pending_label"] = None
                                state["pending_count"] = 0
                                state["bad_frame_count"] = 0
                                state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS
                                if voted_label == "VIRAT KOHLI":
                                    state["label_hold_until"] = time.time() + VIRAT_STICKY_HOLD_SECONDS
                                emit_label_if_changed(
                                    state["current_label"],
                                    ui_state,
                                    state["current_scores"],
                                    source_label=state["current_source_label"],
                                )
                        else:
                            state["pending_label"] = None
                            state["pending_count"] = 0

                    elif voted_label == "UNKNOWN":
                        if state["pending_label"] == "UNKNOWN":
                            state["pending_count"] += 1
                        else:
                            state["pending_label"] = "UNKNOWN"
                            state["pending_count"] = 1

                        if state["pending_count"] >= UNKNOWN_CONFIRMATIONS:
                            state["current_label"] = "UNKNOWN"
                            state["current_scores"] = scores.copy()
                            state["current_best_score"] = best_score
                            state["current_margin"] = margin
                            state["pending_label"] = None
                            state["pending_count"] = 0
                            state["bad_frame_count"] = 0
                            state["label_hold_until"] = 0.0
                            emit_label_if_changed(
                                state["current_label"],
                                ui_state,
                                state["current_scores"],
                                source_label=state["current_source_label"],
                            )

                elif state["current_label"] in ("PAVAN", "VIRAT KOHLI"):
                    current_label = state["current_label"]
                    current_speaker_score = float(scores.get(current_label, -1.0))
                    hold_active = time.time() < state["label_hold_until"]

                    if voted_label == current_label and current_speaker_score >= KEEP_THRESHOLDS[current_label]:
                        state["current_scores"] = scores.copy()
                        state["current_best_score"] = current_speaker_score
                        state["current_margin"] = abs(scores["PAVAN"] - scores["VIRAT KOHLI"])
                        state["pending_label"] = None
                        state["pending_count"] = 0
                        state["bad_frame_count"] = 0

                        if current_label == "VIRAT KOHLI":
                            state["label_hold_until"] = time.time() + VIRAT_STICKY_HOLD_SECONDS
                        elif current_speaker_score > 0.44:
                            state["label_hold_until"] = time.time() + 1.6

                        emit_label_if_changed(
                            state["current_label"],
                            ui_state,
                            state["current_scores"],
                            source_label=state["current_source_label"],
                        )

                    elif should_fast_switch(best_label, scores, current_label):
                        state["current_label"] = best_label
                        state["current_scores"] = scores.copy()
                        state["current_best_score"] = float(scores.get(best_label, 0.0))
                        state["current_margin"] = abs(scores["PAVAN"] - scores["VIRAT KOHLI"])
                        state["pending_label"] = None
                        state["pending_count"] = 0
                        state["bad_frame_count"] = 0
                        if best_label == "VIRAT KOHLI":
                            state["label_hold_until"] = time.time() + VIRAT_STICKY_HOLD_SECONDS
                        else:
                            state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS
                        decision_history.clear()
                        emit_label_if_changed(
                            state["current_label"],
                            ui_state,
                            state["current_scores"],
                            source_label=state["current_source_label"],
                        )

                    elif voted_label in ("PAVAN", "VIRAT KOHLI") and voted_label != current_label and not hold_active:
                        voted_score = float(scores.get(voted_label, 0.0))
                        voted_margin = abs(scores["PAVAN"] - scores["VIRAT KOHLI"])

                        if should_accept_switch(
                            voted_label,
                            voted_score,
                            voted_margin,
                            scores,
                            current_label=current_label,
                        ):
                            if state["pending_label"] == voted_label:
                                state["pending_count"] += 1
                            else:
                                state["pending_label"] = voted_label
                                state["pending_count"] = 1

                            if state["pending_count"] >= SWITCH_CONFIRMATIONS:
                                state["current_label"] = voted_label
                                state["current_scores"] = scores.copy()
                                state["current_best_score"] = voted_score
                                state["current_margin"] = voted_margin
                                state["pending_label"] = None
                                state["pending_count"] = 0
                                state["bad_frame_count"] = 0
                                if voted_label == "VIRAT KOHLI":
                                    state["label_hold_until"] = time.time() + VIRAT_STICKY_HOLD_SECONDS
                                else:
                                    state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS
                                decision_history.clear()
                                emit_label_if_changed(
                                    state["current_label"],
                                    ui_state,
                                    state["current_scores"],
                                    source_label=state["current_source_label"],
                                )
                        else:
                            state["pending_label"] = None
                            state["pending_count"] = 0

                    else:
                        good_enough = current_speaker_score >= KEEP_THRESHOLDS[current_label]
                        other_label = "VIRAT KOHLI" if current_label == "PAVAN" else "PAVAN"
                        other_score = float(scores.get(other_label, 0.0))
                        current_margin_vs_other = current_speaker_score - other_score

                        if current_label == "VIRAT KOHLI":
                            if current_speaker_score >= VIRAT_RELEASE_MIN_SCORE and current_margin_vs_other >= VIRAT_RELEASE_MARGIN:
                                state["bad_frame_count"] = 0
                                state["label_hold_until"] = max(
                                    state["label_hold_until"],
                                    time.time() + 0.45
                                )
                            else:
                                state["bad_frame_count"] += 1
                        else:
                            if good_enough or hold_active or current_margin_vs_other >= 0.06:
                                state["bad_frame_count"] = 0
                            else:
                                state["bad_frame_count"] += 1

                        if (not hold_active) and state["bad_frame_count"] >= BAD_FRAMES_TO_RELEASE:
                            state["current_label"] = "UNKNOWN"
                            state["current_scores"] = scores.copy()
                            state["current_best_score"] = best_score
                            state["current_margin"] = margin
                            state["pending_label"] = None
                            state["pending_count"] = 0
                            state["bad_frame_count"] = 0
                            state["label_hold_until"] = 0.0
                            decision_history.clear()
                            emit_label_if_changed(
                                state["current_label"],
                                ui_state,
                                state["current_scores"],
                                source_label=state["current_source_label"],
                            )

                else:
                    if voted_label in ("PAVAN", "VIRAT KOHLI"):
                        voted_score = float(scores.get(voted_label, 0.0))
                        voted_margin = abs(scores["PAVAN"] - scores["VIRAT KOHLI"])

                        if should_accept_start(voted_label, voted_score, voted_margin):
                            if state["pending_label"] == voted_label:
                                state["pending_count"] += 1
                            else:
                                state["pending_label"] = voted_label
                                state["pending_count"] = 1

                            if state["pending_count"] >= INITIAL_CONFIRMATIONS:
                                state["current_label"] = voted_label
                                state["current_scores"] = scores.copy()
                                state["current_best_score"] = voted_score
                                state["current_margin"] = voted_margin
                                state["pending_label"] = None
                                state["pending_count"] = 0
                                state["bad_frame_count"] = 0
                                if voted_label == "VIRAT KOHLI":
                                    state["label_hold_until"] = time.time() + VIRAT_STICKY_HOLD_SECONDS
                                else:
                                    state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS
                                decision_history.clear()
                                emit_label_if_changed(
                                    state["current_label"],
                                    ui_state,
                                    state["current_scores"],
                                    source_label=state["current_source_label"],
                                )
                        else:
                            state["pending_label"] = None
                            state["pending_count"] = 0

            state["last_live_check_time"] = now

except KeyboardInterrupt:
    print("\nStopping system...")
    print("Stopped by user.")

finally:
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass