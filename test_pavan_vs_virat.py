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

import numpy as np
import sounddevice as sd
import torch
import librosa

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from speechbrain.pretrained import EncoderClassifier

# -------------------------------------------------
# WINDOWS ANSI COLORS
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
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


enable_windows_ansi()

RESET = "\033[0m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RED = "\033[91m"

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECAPA_CACHE_DIR = os.path.join(BASE_DIR, "ecapa_cache")

PAVAN_REF_FILE = os.path.join(BASE_DIR, "pavan_reference_embeddings.pt")
SUMANTH_REF_FILE = os.path.join(BASE_DIR, "sumanth_reference_embeddings.pt")
VIVEK_REF_FILE = os.path.join(BASE_DIR, "vivek_reference_embeddings.pt")

os.makedirs(ECAPA_CACHE_DIR, exist_ok=True)

# -------------------------------------------------
# AUDIO SETTINGS
# -------------------------------------------------
MIC_DEVICE = 1              # change only if your mic device index is different
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1
DTYPE = "float32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# VAD / SEGMENT CONTROL
# -------------------------------------------------
START_ENERGY_THRESHOLD = 0.0016
CONTINUE_ENERGY_THRESHOLD = 0.00105
NOISE_MULTIPLIER_START = 1.30
NOISE_MULTIPLIER_CONTINUE = 1.08

VOICE_CHUNKS_TO_START = 2
SILENCE_LIMIT = 1.25
SILENT_CHUNKS_TO_SOFT_RESET = 10
PREBUFFER_CHUNKS = 10

# -------------------------------------------------
# LIVE WINDOW SETTINGS
# -------------------------------------------------
LIVE_UPDATE_INTERVAL = 0.08

# Stable scoring uses moderate context.
STABLE_WINDOWS = (0.55, 0.85, 1.15)

# Transition scoring uses very short context so old speaker dies quickly.
FAST_WINDOWS = (0.32, 0.48, 0.65)

MAX_ANALYSIS_SECONDS = 1.25
POST_SWITCH_KEEP_CHUNKS = 4
FAST_MODE_KEEP_CHUNKS = 3

# -------------------------------------------------
# AUDIO CLEANUP
# -------------------------------------------------
TRIM_TOP_DB = 24
TARGET_PEAK = 0.92
MIN_PREPARED_AUDIO_SECONDS = 0.26
MIN_DECISION_AUDIO_SECONDS = 0.28
MIN_AUDIO_RMS = 0.0038
MIN_VALID_SCORE_RMS = 0.0048

# -------------------------------------------------
# MULTI-REFERENCE SCORING
# -------------------------------------------------
TOPK_MEAN = 2
TOPK_MIN = 2
BEST_WEIGHT = 0.42
TOPK_WEIGHT = 0.48
CENTER_WEIGHT = 0.10
CONSISTENCY_BONUS = 0.010

# -------------------------------------------------
# DECISION THRESHOLDS
# -------------------------------------------------
START_THRESHOLDS = {
    "PAVAN": 0.435,
    "SUMANTH": 0.350,
    "VIVEK": 0.405,
}

KEEP_THRESHOLDS = {
    "PAVAN": 0.365,
    "SUMANTH": 0.300,
    "VIVEK": 0.348,
}

SOFT_KEEP_THRESHOLDS = {
    "PAVAN": 0.325,
    "SUMANTH": 0.270,
    "VIVEK": 0.305,
}

SWITCH_THRESHOLDS = {
    "PAVAN": 0.405,
    "SUMANTH": 0.332,
    "VIVEK": 0.382,
}

START_MARGINS = {
    "PAVAN": 0.034,
    "SUMANTH": 0.016,
    "VIVEK": 0.030,
}

SWITCH_MARGINS = {
    "PAVAN": 0.026,
    "SUMANTH": 0.014,
    "VIVEK": 0.024,
}

UNKNOWN_SCORE_FLOOR = 0.265
UNKNOWN_MARGIN_FLOOR = 0.008
UNKNOWN_NEAR_GAP = 0.016
UNKNOWN_STRONG_KNOWN_CAP = 0.325

INITIAL_CONFIRMATIONS = 2
SWITCH_CONFIRMATIONS = 2
UNKNOWN_CONFIRMATIONS = 6

# -------------------------------------------------
# STABILITY / SMOOTHING / FAST SWITCH
# -------------------------------------------------
HISTORY_SIZE = 6
STABLE_HISTORY_WINDOW = 4
MIN_HISTORY_VOTE_KNOWN = 2
MIN_HISTORY_VOTE_UNKNOWN = 3

SMOOTHING_ALPHA_STABLE = 0.38
SMOOTHING_ALPHA_FAST = 0.10

LABEL_HOLD_SECONDS = 0.22
STRONG_LABEL_HOLD_SECONDS = 0.34
BAD_FRAMES_TO_RELEASE = 5

FAST_SWITCH_SCORE_DROP = 0.050
FAST_SWITCH_SCORE_RISE = 0.050
FAST_SWITCH_GAP = 0.030
FAST_SWITCH_MIN_NEW_SCORE = 0.325
FAST_MODE_SECONDS = 0.55

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def color_for_label(label):
    return {
        "PAVAN": GREEN,
        "SUMANTH": MAGENTA,
        "VIVEK": CYAN,
        "UNKNOWN": YELLOW,
    }.get(label, RESET)


def speaker_line(label, confidence=None):
    confidence_text = "" if confidence is None else f" ({confidence}%)"
    if label == "PAVAN":
        return f"{GREEN}Pavan is speaking{confidence_text}...{RESET}"
    if label == "SUMANTH":
        return f"{MAGENTA}Sumanth is speaking{confidence_text}...{RESET}"
    if label == "VIVEK":
        return f"{CYAN}Vivek is speaking{confidence_text}...{RESET}"
    return f"{YELLOW}Unknown speaker is speaking{confidence_text}...{RESET}"


def print_debug(msg):
    print(f"{RED}{msg}{RESET}")


def get_top2_margin(scores):
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_label, second_score = ordered[1]
    margin = best_score - second_score
    return best_label, float(best_score), second_label, float(second_score), float(margin)


def compute_confidence(label, scores):
    ordered_vals = sorted(
        [float(scores.get("PAVAN", 0.0)), float(scores.get("SUMANTH", 0.0)), float(scores.get("VIVEK", 0.0))],
        reverse=True,
    )
    best = ordered_vals[0]
    second = ordered_vals[1]
    margin = best - second

    if label == "UNKNOWN":
        if best < 0.22:
            return 96
        if best < 0.28:
            return 90
        if best < 0.34:
            return 82
        return 72

    score_conf = min(max((best - 0.28) / 0.28, 0.0), 1.0)
    margin_conf = min(max(margin / 0.16, 0.0), 1.0)
    conf = int(52 + 30 * score_conf + 17 * margin_conf)
    return max(55, min(99, conf))


def emit_label_if_changed(label, ui_state, scores=None):
    if label is None:
        return
    confidence = None if scores is None else compute_confidence(label, scores)
    if ui_state["last_printed_label"] != label:
        print(speaker_line(label, confidence))
        ui_state["last_printed_label"] = label


def print_live_scores(scores, best_label, best_score, margin, fast_mode=False):
    best_color = color_for_label(best_label)
    mode_text = "FAST" if fast_mode else "STABLE"
    print(
        f"[LIVE-{mode_text}] "
        f"PAVAN={scores['PAVAN']:.4f} | "
        f"SUMANTH={scores['SUMANTH']:.4f} | "
        f"VIVEK={scores['VIVEK']:.4f} | "
        f"BEST={best_color}{best_label} {best_score:.4f}{RESET} | "
        f"MARGIN={margin:.4f}"
    )


def l2_normalize_tensor(x):
    x = x.detach().cpu().float().flatten()
    norm = torch.norm(x, p=2)
    if norm.item() == 0:
        return x
    return x / norm


def safe_flatten_mono(audio_np):
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim > 1:
        if audio_np.shape[1] == 1:
            audio_np = audio_np[:, 0]
        else:
            audio_np = np.mean(audio_np, axis=1)
    return np.ascontiguousarray(audio_np.flatten(), dtype=np.float32)


def safe_concat_chunks(chunks):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks]), dtype=np.float32)


def chunk_rms(chunk):
    chunk = np.asarray(chunk, dtype=np.float32).flatten()
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))


def chunk_zero_crossing_rate(chunk):
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


def audio_rms(audio_np):
    audio_np = np.asarray(audio_np, dtype=np.float32).flatten()
    if len(audio_np) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_np)) + 1e-12))


def estimate_noise_floor(pre_buffer):
    if not pre_buffer:
        return START_ENERGY_THRESHOLD
    rms_vals = [chunk_rms(c) for c in list(pre_buffer)]
    if len(rms_vals) < 4:
        return START_ENERGY_THRESHOLD
    return float(np.percentile(rms_vals, 35))


def is_voice_chunk(chunk, is_currently_speaking, pre_buffer=None):
    energy = chunk_rms(chunk)
    zcr = chunk_zero_crossing_rate(chunk)
    noise_floor = estimate_noise_floor(pre_buffer) if pre_buffer is not None else 0.0

    if is_currently_speaking:
        threshold = max(CONTINUE_ENERGY_THRESHOLD, noise_floor * NOISE_MULTIPLIER_CONTINUE)
    else:
        threshold = max(START_ENERGY_THRESHOLD, noise_floor * NOISE_MULTIPLIER_START)

    if energy < threshold:
        return False

    if zcr < 0.0010 or zcr > 0.52:
        return False

    return True


def prepare_audio(audio_np):
    if audio_np is None or len(audio_np) == 0:
        return None

    audio_np = safe_flatten_mono(audio_np)
    if len(audio_np) == 0:
        return None

    peak = np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0.0
    if peak < 1e-5:
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

    rms = np.sqrt(np.mean(audio_np ** 2) + 1e-12)
    if rms < 0.0035:
        return None

    audio_np = audio_np / max(rms, 1e-8)

    peak = np.max(np.abs(audio_np))
    if peak > 1e-6:
        audio_np = TARGET_PEAK * (audio_np / peak)

    audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

    if audio_rms(audio_np) < MIN_AUDIO_RMS:
        return None

    return np.ascontiguousarray(audio_np, dtype=np.float32)


def is_valid_audio_for_scoring(audio_np):
    if audio_np is None or len(audio_np) == 0:
        return False
    rms = np.sqrt(np.mean(audio_np ** 2) + 1e-12)
    if rms < MIN_VALID_SCORE_RMS:
        return False
    peak = np.max(np.abs(audio_np))
    if peak < 0.03:
        return False
    return True


def load_reference_bank(path):
    if not os.path.exists(path):
        print(f"Missing reference file: {path}")
        sys.exit(1)

    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict) or "embeddings" not in payload:
        print(f"Invalid reference file format: {path}")
        sys.exit(1)

    emb = payload["embeddings"]
    if isinstance(emb, np.ndarray):
        emb = torch.tensor(emb, dtype=torch.float32)
    emb = emb.detach().cpu().float()

    if emb.ndim == 1:
        emb = emb.unsqueeze(0)

    norms = torch.norm(emb, dim=1, keepdim=True).clamp(min=1e-12)
    emb = emb / norms

    center = payload.get("center_embedding", None)
    if center is None:
        center = torch.mean(emb, dim=0)
    if isinstance(center, np.ndarray):
        center = torch.tensor(center, dtype=torch.float32)
    center = l2_normalize_tensor(center)

    return {
        "speaker": payload.get("speaker", "UNKNOWN"),
        "embeddings": emb,
        "center_embedding": center,
        "files": payload.get("files", []),
        "scores_to_center": payload.get("scores_to_center", []),
        "num_embeddings": int(payload.get("num_embeddings", emb.shape[0])),
    }


def load_reference_embeddings():
    return {
        "PAVAN": load_reference_bank(PAVAN_REF_FILE),
        "SUMANTH": load_reference_bank(SUMANTH_REF_FILE),
        "VIVEK": load_reference_bank(VIVEK_REF_FILE),
    }


def create_input_embedding(classifier, audio_np):
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform).squeeze()
    return l2_normalize_tensor(embedding)


def speaker_multiref_score(input_emb, ref_bank):
    ref_embs = ref_bank["embeddings"]
    sims = torch.matmul(ref_embs, input_emb.cpu())
    sims = sims.detach().cpu().float()

    n = sims.numel()
    k = min(TOPK_MEAN, n)
    k = max(TOPK_MIN, k) if n >= TOPK_MIN else n

    if k <= 0:
        return {"best": 0.0, "topk_mean": 0.0, "center_score": 0.0, "final": 0.0}

    topk_vals = torch.topk(sims, k=k, largest=True).values
    best_score = float(topk_vals[0].item())
    topk_mean = float(torch.mean(topk_vals).item())
    center_score = float(torch.dot(input_emb.cpu(), ref_bank["center_embedding"]).item())

    spread_penalty = 0.0
    if k >= 2:
        gap12 = float(topk_vals[0].item() - topk_vals[1].item())
        if gap12 > 0.10:
            spread_penalty = 0.015

    final = (
        BEST_WEIGHT * best_score
        + TOPK_WEIGHT * topk_mean
        + CENTER_WEIGHT * center_score
        - spread_penalty
    )

    if k >= 2 and float(topk_vals[-1].item()) >= 0.38:
        final += CONSISTENCY_BONUS

    return {
        "best": best_score,
        "topk_mean": topk_mean,
        "center_score": center_score,
        "final": float(min(final, 1.0)),
    }


def score_single_window(classifier, reference_embeddings, audio_np):
    prepared = prepare_audio(audio_np)
    if prepared is None:
        return None

    emb = create_input_embedding(classifier, prepared)

    pavan_scores = speaker_multiref_score(emb, reference_embeddings["PAVAN"])
    sumanth_scores = speaker_multiref_score(emb, reference_embeddings["SUMANTH"])
    vivek_scores = speaker_multiref_score(emb, reference_embeddings["VIVEK"])

    return {
        "PAVAN": pavan_scores["final"],
        "SUMANTH": sumanth_scores["final"],
        "VIVEK": vivek_scores["final"],
    }


def blend_scores(prev_scores, new_scores, alpha):
    if prev_scores is None:
        return new_scores.copy()
    out = {}
    for k in ("PAVAN", "SUMANTH", "VIVEK"):
        out[k] = float(alpha * prev_scores.get(k, 0.0) + (1.0 - alpha) * new_scores.get(k, 0.0))
    return out


def voiced_ratio(chunks):
    if not chunks:
        return 0.0
    voiced = 0
    for c in chunks:
        if chunk_rms(c) > CONTINUE_ENERGY_THRESHOLD:
            voiced += 1
    return voiced / len(chunks)


def score_multiwindow(classifier, reference_embeddings, live_chunks, windows):
    if len(live_chunks) == 0:
        return None

    total_audio = safe_concat_chunks(list(live_chunks))
    total_audio_sec = len(total_audio) / SAMPLE_RATE
    if total_audio_sec < MIN_DECISION_AUDIO_SECONDS:
        return None

    if voiced_ratio(live_chunks) < 0.15:
        return None

    prepared_full = prepare_audio(total_audio)
    if prepared_full is None:
        return None

    if not is_valid_audio_for_scoring(prepared_full):
        return None

    scores_list = []
    for win_sec in windows:
        target_len = int(win_sec * SAMPLE_RATE)
        if len(prepared_full) < target_len:
            continue
        win_audio = prepared_full[-target_len:]
        result = score_single_window(classifier, reference_embeddings, win_audio)
        if result is not None:
            scores_list.append((win_sec, result))

    if not scores_list:
        return None

    if windows == FAST_WINDOWS:
        weight_map = {0.32: 0.50, 0.48: 0.32, 0.65: 0.18}
    else:
        weight_map = {0.55: 0.30, 0.85: 0.38, 1.15: 0.32}

    scores = {"PAVAN": 0.0, "SUMANTH": 0.0, "VIVEK": 0.0}
    total_w = 0.0
    for win_sec, result in scores_list:
        w = weight_map.get(round(win_sec, 2), 0.33)
        total_w += w
        for spk in scores:
            scores[spk] += w * float(result[spk])

    if total_w <= 0:
        return None

    for spk in scores:
        scores[spk] = float(scores[spk] / total_w)

    return scores


def stable_vote(history):
    if len(history) < STABLE_HISTORY_WINDOW:
        return None
    recent = list(history)[-STABLE_HISTORY_WINDOW:]
    counts = Counter(recent)
    label, count = counts.most_common(1)[0]
    needed = MIN_HISTORY_VOTE_UNKNOWN if label == "UNKNOWN" else MIN_HISTORY_VOTE_KNOWN
    if count >= needed:
        return label
    return None


def get_start_margin_for_label(label):
    return START_MARGINS.get(label, 0.03)


def get_switch_margin_for_label(label):
    return SWITCH_MARGINS.get(label, 0.024)


def get_decision_label(scores):
    best_label, best_score, second_label, second_score, margin = get_top2_margin(scores)

    if best_score < UNKNOWN_SCORE_FLOOR:
        return "UNKNOWN", best_score, margin

    if best_score <= UNKNOWN_STRONG_KNOWN_CAP and margin < UNKNOWN_NEAR_GAP:
        return "UNKNOWN", best_score, margin

    if margin < UNKNOWN_MARGIN_FLOOR:
        return "UNKNOWN", best_score, margin

    return best_label, best_score, margin


def should_accept_start(label, score, margin):
    if label not in START_THRESHOLDS:
        return False
    return score >= START_THRESHOLDS[label] and margin >= get_start_margin_for_label(label)


def should_accept_switch(label, score, margin, scores, current_label=None):
    if label not in SWITCH_THRESHOLDS:
        return False
    if score < SWITCH_THRESHOLDS[label]:
        return False
    if margin < get_switch_margin_for_label(label):
        return False
    if current_label in ("PAVAN", "SUMANTH", "VIVEK"):
        current_score_now = float(scores.get(current_label, 0.0))
        min_gap = 0.006 if label == "SUMANTH" else 0.010
        if score <= current_score_now + min_gap:
            return False
    return True


def recent_tail_chunks(source_chunks, keep_count):
    if not source_chunks:
        return []
    return list(source_chunks)[-keep_count:]


def refill_live_chunks_recent(live_chunks, recent_chunks):
    live_chunks.clear()
    for c in recent_chunks:
        live_chunks.append(np.asarray(c, dtype=np.float32).copy())


def should_enter_fast_mode(state, raw_scores):
    current_label = state["current_label"]
    if current_label not in ("PAVAN", "SUMANTH", "VIVEK"):
        return False, None

    prev_raw = state["prev_raw_scores"]
    if prev_raw is None:
        return False, None

    current_prev = float(prev_raw.get(current_label, 0.0))
    current_now = float(raw_scores.get(current_label, 0.0))
    current_drop = current_prev - current_now

    other_scores = {k: v for k, v in raw_scores.items() if k != current_label}
    challenger_label = max(other_scores, key=other_scores.get)
    challenger_now = float(other_scores[challenger_label])
    challenger_prev = float(prev_raw.get(challenger_label, 0.0))
    challenger_rise = challenger_now - challenger_prev
    gap = challenger_now - current_now

    enter = (
        current_drop >= FAST_SWITCH_SCORE_DROP
        and challenger_rise >= FAST_SWITCH_SCORE_RISE
        and challenger_now >= FAST_SWITCH_MIN_NEW_SCORE
        and gap >= 0.0
    )
    return enter, challenger_label


def should_fast_switch(scores, current_label):
    if current_label not in ("PAVAN", "SUMANTH", "VIVEK"):
        return False, None
    other_scores = {k: v for k, v in scores.items() if k != current_label}
    challenger_label = max(other_scores, key=other_scores.get)
    challenger_score = float(other_scores[challenger_label])
    current_score = float(scores.get(current_label, 0.0))
    gap = challenger_score - current_score

    if challenger_score < SWITCH_THRESHOLDS.get(challenger_label, 1.0):
        return False, None
    if gap < FAST_SWITCH_GAP:
        return False, None

    return True, challenger_label


def reset_runtime_state():
    return {
        "is_speaking": False,
        "silence_time": 0.0,
        "silent_chunk_count": 0,
        "voice_chunk_count": 0,
        "last_live_check_time": 0.0,
        "current_label": None,
        "current_scores": {"PAVAN": 0.0, "SUMANTH": 0.0, "VIVEK": 0.0},
        "smoothed_scores": None,
        "current_best_score": 0.0,
        "current_margin": 0.0,
        "current_segment_chunks": [],
        "pending_label": None,
        "pending_count": 0,
        "last_live_signature": None,
        "bad_frame_count": 0,
        "label_hold_until": 0.0,
        "fast_mode_until": 0.0,
        "fast_mode_candidate": None,
        "prev_raw_scores": None,
    }


def stop_current_segment(state, decision_history, live_chunks, ui_state, reason_text):
    state = reset_runtime_state()
    decision_history.clear()
    live_chunks.clear()
    ui_state["last_printed_label"] = None
    print(reason_text)
    return state


def apply_switch(state, live_chunks, ui_state, new_label, scores):
    state["current_label"] = new_label
    state["current_scores"] = scores.copy()
    state["current_best_score"] = float(scores.get(new_label, 0.0))
    _, _, _, _, state["current_margin"] = get_top2_margin(scores)
    state["pending_label"] = None
    state["pending_count"] = 0
    state["bad_frame_count"] = 0
    state["smoothed_scores"] = None
    state["last_live_signature"] = None
    state["fast_mode_candidate"] = None

    if state["current_best_score"] >= 0.52:
        state["label_hold_until"] = time.time() + STRONG_LABEL_HOLD_SECONDS
    else:
        state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS

    refill_live_chunks_recent(live_chunks, recent_tail_chunks(live_chunks, POST_SWITCH_KEEP_CHUNKS))
    emit_label_if_changed(state["current_label"], ui_state, state["current_scores"])


# -------------------------------------------------
# BOOT
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

max_live_chunks = int((max(STABLE_WINDOWS) * SAMPLE_RATE) / BLOCK_SIZE) + 12
pre_buffer = deque(maxlen=PREBUFFER_CHUNKS)
live_chunks = deque(maxlen=max_live_chunks)

state = reset_runtime_state()
decision_history = deque(maxlen=HISTORY_SIZE)
ui_state = {"last_printed_label": None}

chunk_duration = BLOCK_SIZE / SAMPLE_RATE

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    blocksize=BLOCK_SIZE,
    dtype=DTYPE,
    device=MIC_DEVICE,
    latency="low",
)

try:
    stream.start()

    while True:
        try:
            audio_chunk, _ = stream.read(BLOCK_SIZE)
        except Exception as read_error:
            print_debug(f"[AUDIO READ ERROR] {read_error}")
            time.sleep(0.05)
            continue

        if audio_chunk is None or len(audio_chunk) == 0:
            continue

        chunk = np.asarray(audio_chunk[:, 0], dtype=np.float32).copy()
        pre_buffer.append(chunk.copy())
        chunk_has_voice = is_voice_chunk(chunk, state["is_speaking"], pre_buffer)

        # -------------------------------------------------
        # SILENCE
        # -------------------------------------------------
        if not chunk_has_voice:
            state["voice_chunk_count"] = 0

            if state["is_speaking"]:
                state["silent_chunk_count"] += 1
                state["silence_time"] += chunk_duration

                if state["silent_chunk_count"] >= SILENT_CHUNKS_TO_SOFT_RESET:
                    live_chunks.clear()
                    state["pending_label"] = None
                    state["pending_count"] = 0
                    state["last_live_signature"] = None
                    state["last_live_check_time"] = 0.0
                    state["smoothed_scores"] = None
                    state["prev_raw_scores"] = None
                    state["fast_mode_until"] = 0.0
                    state["fast_mode_candidate"] = None

                if state["silence_time"] >= SILENCE_LIMIT:
                    state = stop_current_segment(
                        state,
                        decision_history,
                        live_chunks,
                        ui_state,
                        "[SILENCE DETECTED - STOPPED]",
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

                state["current_label"] = None
                state["current_scores"] = {"PAVAN": 0.0, "SUMANTH": 0.0, "VIVEK": 0.0}
                state["smoothed_scores"] = None
                state["current_best_score"] = 0.0
                state["current_margin"] = 0.0
                state["pending_label"] = None
                state["pending_count"] = 0
                state["last_live_check_time"] = 0.0
                state["last_live_signature"] = None
                state["bad_frame_count"] = 0
                state["label_hold_until"] = 0.0
                state["fast_mode_until"] = 0.0
                state["fast_mode_candidate"] = None
                state["prev_raw_scores"] = None
                decision_history.clear()
                ui_state["last_printed_label"] = None

                print("[VOICE DETECTED]")

        state["silent_chunk_count"] = 0
        state["silence_time"] = 0.0
        state["current_segment_chunks"].append(chunk.copy())
        live_chunks.append(chunk.copy())

        now = time.time()
        if now - state["last_live_check_time"] < LIVE_UPDATE_INTERVAL:
            continue

        fast_mode = now < state["fast_mode_until"]
        windows = FAST_WINDOWS if fast_mode else STABLE_WINDOWS

        raw_scores = score_multiwindow(
            classifier=classifier,
            reference_embeddings=reference_embeddings,
            live_chunks=live_chunks,
            windows=windows,
        )

        if raw_scores is None:
            state["last_live_check_time"] = now
            continue

        # Enter fast mode quickly when a challenger rises and current speaker drops.
        if not fast_mode and state["current_label"] in ("PAVAN", "SUMANTH", "VIVEK"):
            enter_fast, challenger = should_enter_fast_mode(state, raw_scores)
            if enter_fast:
                state["fast_mode_until"] = now + FAST_MODE_SECONDS
                state["fast_mode_candidate"] = challenger
                refill_live_chunks_recent(live_chunks, recent_tail_chunks(live_chunks, FAST_MODE_KEEP_CHUNKS))
                state["smoothed_scores"] = None
                state["last_live_signature"] = None
                decision_history.clear()
                state["prev_raw_scores"] = raw_scores.copy()
                state["last_live_check_time"] = now
                continue

        alpha = SMOOTHING_ALPHA_FAST if fast_mode else SMOOTHING_ALPHA_STABLE
        state["smoothed_scores"] = blend_scores(state["smoothed_scores"], raw_scores, alpha)
        scores = state["smoothed_scores"].copy()

        best_label, best_score, margin = get_decision_label(scores)

        live_signature = (
            round(scores["PAVAN"], 4),
            round(scores["SUMANTH"], 4),
            round(scores["VIVEK"], 4),
            best_label,
            round(best_score, 4),
            round(margin, 4),
            fast_mode,
        )

        if live_signature != state["last_live_signature"]:
            print_live_scores(scores, best_label, best_score, margin, fast_mode=fast_mode)
            state["last_live_signature"] = live_signature

        decision_history.append(best_label)
        voted_label = stable_vote(decision_history)

        # -------------------------------------------------
        # NO CURRENT LABEL YET
        # -------------------------------------------------
        if state["current_label"] is None:
            if voted_label in ("PAVAN", "SUMANTH", "VIVEK"):
                voted_score = float(scores.get(voted_label, 0.0))
                _, _, _, _, voted_margin = get_top2_margin(scores)

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
                        state["fast_mode_until"] = 0.0
                        state["fast_mode_candidate"] = None

                        if voted_score >= 0.52:
                            state["label_hold_until"] = time.time() + STRONG_LABEL_HOLD_SECONDS
                        else:
                            state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS

                        emit_label_if_changed(state["current_label"], ui_state, state["current_scores"])
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
                    emit_label_if_changed("UNKNOWN", ui_state, state["current_scores"])

        # -------------------------------------------------
        # CURRENT LABEL = KNOWN SPEAKER
        # -------------------------------------------------
        elif state["current_label"] in ("PAVAN", "SUMANTH", "VIVEK"):
            current_label = state["current_label"]
            current_score = float(scores.get(current_label, 0.0))
            hold_active = (time.time() < state["label_hold_until"]) and not fast_mode

            # keep same label
            if voted_label == current_label and current_score >= KEEP_THRESHOLDS[current_label]:
                state["current_scores"] = scores.copy()
                state["current_best_score"] = current_score
                _, _, _, _, current_margin = get_top2_margin(scores)
                state["current_margin"] = current_margin
                state["pending_label"] = None
                state["pending_count"] = 0
                state["bad_frame_count"] = 0

                if current_score >= 0.52:
                    state["label_hold_until"] = time.time() + STRONG_LABEL_HOLD_SECONDS
                else:
                    state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS

            else:
                # very fast switch path
                can_fast_switch, challenger_label = should_fast_switch(scores, current_label)
                if can_fast_switch and (fast_mode or not hold_active):
                    apply_switch(state, live_chunks, ui_state, challenger_label, scores)
                    decision_history.clear()
                    state["fast_mode_until"] = 0.0

                # normal confirmed switch path
                elif voted_label in ("PAVAN", "SUMANTH", "VIVEK") and voted_label != current_label and (fast_mode or not hold_active):
                    voted_score = float(scores.get(voted_label, 0.0))
                    _, _, _, _, voted_margin = get_top2_margin(scores)

                    if should_accept_switch(voted_label, voted_score, voted_margin, scores, current_label=current_label):
                        if state["pending_label"] == voted_label:
                            state["pending_count"] += 1
                        else:
                            state["pending_label"] = voted_label
                            state["pending_count"] = 1

                        if state["pending_count"] >= SWITCH_CONFIRMATIONS:
                            apply_switch(state, live_chunks, ui_state, voted_label, scores)
                            decision_history.clear()
                            state["fast_mode_until"] = 0.0
                    else:
                        state["pending_label"] = None
                        state["pending_count"] = 0

                # release current label if it is no longer convincing
                else:
                    other_scores = {k: v for k, v in scores.items() if k != current_label}
                    strongest_other_label = max(other_scores, key=other_scores.get)
                    strongest_other = float(other_scores[strongest_other_label])
                    current_margin_vs_other = current_score - strongest_other
                    soft_keep = SOFT_KEEP_THRESHOLDS[current_label]

                    if strongest_other >= SWITCH_THRESHOLDS.get(strongest_other_label, 1.0) and strongest_other > current_score + 0.026:
                        state["bad_frame_count"] += 2
                    elif current_score >= KEEP_THRESHOLDS[current_label] or current_margin_vs_other >= 0.020:
                        state["bad_frame_count"] = 0
                    elif current_score >= soft_keep and strongest_other < current_score + 0.020:
                        state["bad_frame_count"] = max(0, state["bad_frame_count"] - 1)
                    else:
                        state["bad_frame_count"] += 1

                    allow_unknown_release = best_score < 0.250 or strongest_other > current_score + 0.040

                    if (fast_mode or not hold_active) and state["bad_frame_count"] >= BAD_FRAMES_TO_RELEASE and allow_unknown_release:
                        state["current_label"] = "UNKNOWN"
                        state["current_scores"] = scores.copy()
                        state["current_best_score"] = best_score
                        state["current_margin"] = margin
                        state["pending_label"] = None
                        state["pending_count"] = 0
                        state["bad_frame_count"] = 0
                        state["label_hold_until"] = 0.0
                        decision_history.clear()
                        emit_label_if_changed("UNKNOWN", ui_state, state["current_scores"])

        # -------------------------------------------------
        # CURRENT LABEL = UNKNOWN
        # -------------------------------------------------
        else:
            if voted_label in ("PAVAN", "SUMANTH", "VIVEK"):
                voted_score = float(scores.get(voted_label, 0.0))
                _, _, _, _, voted_margin = get_top2_margin(scores)

                if should_accept_start(voted_label, voted_score, voted_margin):
                    if state["pending_label"] == voted_label:
                        state["pending_count"] += 1
                    else:
                        state["pending_label"] = voted_label
                        state["pending_count"] = 1

                    if state["pending_count"] >= INITIAL_CONFIRMATIONS:
                        apply_switch(state, live_chunks, ui_state, voted_label, scores)
                        decision_history.clear()
                        state["fast_mode_until"] = 0.0
                else:
                    state["pending_label"] = None
                    state["pending_count"] = 0
            else:
                state["current_scores"] = scores.copy()
                state["current_best_score"] = best_score
                state["current_margin"] = margin
                emit_label_if_changed("UNKNOWN", ui_state, state["current_scores"])

        state["prev_raw_scores"] = raw_scores.copy()
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
