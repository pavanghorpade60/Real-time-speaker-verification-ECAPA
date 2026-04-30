import os
import io
import sys
import time
import warnings
import logging
import contextlib
import ctypes
from collections import deque, Counter

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


def enable_windows_ansi():
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECAPA_CACHE_DIR = os.path.join(BASE_DIR, "ecapa_cache")

PAVAN_REF_FILE = os.path.join(BASE_DIR, "pavan_reference_embeddings.pt")
SUMANTH_REF_FILE = os.path.join(BASE_DIR, "sumanth_reference_embeddings.pt")
VIVEK_REF_FILE = os.path.join(BASE_DIR, "vivek_reference_embeddings.pt")

os.makedirs(ECAPA_CACHE_DIR, exist_ok=True)

MIC_DEVICE = 1
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1
DTYPE = "float32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_ENERGY_THRESHOLD = 0.0018
CONTINUE_ENERGY_THRESHOLD = 0.00105
NOISE_MULTIPLIER_START = 1.30
NOISE_MULTIPLIER_CONTINUE = 1.08

VOICE_CHUNKS_TO_START = 2
SILENCE_LIMIT = 1.25
SILENT_CHUNKS_TO_SOFT_RESET = 8
PREBUFFER_CHUNKS = 8

LIVE_UPDATE_INTERVAL = 0.035

FAST_WINDOWS = (0.20, 0.32)
MID_WINDOWS = (0.45, 0.60)
STABLE_WINDOWS = (0.75, 1.00)

MAX_ANALYSIS_SECONDS = 1.00
POST_SWITCH_KEEP_CHUNKS = 1
FAST_MODE_KEEP_CHUNKS = 2

TRIM_TOP_DB = 24
TARGET_PEAK = 0.92
MIN_PREPARED_AUDIO_SECONDS = 0.20
MIN_DECISION_AUDIO_SECONDS = 0.25
MIN_AUDIO_RMS = 0.0038
MIN_VALID_SCORE_RMS = 0.0048

TOPK_MEAN = 2
TOPK_MIN = 2
BEST_WEIGHT = 0.42
TOPK_WEIGHT = 0.48
CENTER_WEIGHT = 0.10
CONSISTENCY_BONUS = 0.010

START_THRESHOLDS = {
    "PAVAN": 0.340,
    "SUMANTH": 0.240,
    "VIVEK": 0.300,
}

KEEP_THRESHOLDS = {
    "PAVAN": 0.240,
    "SUMANTH": 0.190,
    "VIVEK": 0.220,
}

SOFT_KEEP_THRESHOLDS = {
    "PAVAN": 0.200,
    "SUMANTH": 0.160,
    "VIVEK": 0.180,
}

SWITCH_THRESHOLDS = {
    "PAVAN": 0.335,
    "SUMANTH": 0.245,
    "VIVEK": 0.300,
}

START_MARGINS = {
    "PAVAN": 0.030,
    "SUMANTH": 0.014,
    "VIVEK": 0.026,
}

SWITCH_MARGINS = {
    "PAVAN": 0.022,
    "SUMANTH": 0.012,
    "VIVEK": 0.020,
}

UNKNOWN_SCORE_FLOOR = 0.245
UNKNOWN_MARGIN_FLOOR = 0.042
UNKNOWN_NEAR_GAP = 0.055
UNKNOWN_STRONG_KNOWN_CAP = 0.300

UNKNOWN_TO_KNOWN_SCORE = {
    "PAVAN": 0.340,
    "SUMANTH": 0.255,
    "VIVEK": 0.305,
}

UNKNOWN_TO_KNOWN_MARGIN = 0.050

INITIAL_CONFIRMATIONS = 2
SWITCH_CONFIRMATIONS = 1
FAST_SWITCH_CONFIRMATIONS = 1
UNKNOWN_CONFIRMATIONS = 3

UNKNOWN_TO_KNOWN_CONFIRMATIONS = {
    "PAVAN": 2,
    "SUMANTH": 2,
    "VIVEK": 2,
}

INITIAL_DECISION_DELAY_SECONDS = 0.34

HISTORY_SIZE = 6
STABLE_HISTORY_WINDOW = 3
MIN_HISTORY_VOTE_KNOWN = 2
MIN_HISTORY_VOTE_UNKNOWN = 2

SMOOTHING_ALPHA_STABLE = 0.12
SMOOTHING_ALPHA_FAST = 0.00

LABEL_HOLD_SECONDS = 0.08
STRONG_LABEL_HOLD_SECONDS = 0.15
BAD_FRAMES_TO_RELEASE = 3

FAST_SWITCH_SCORE_DROP = 0.010
FAST_SWITCH_SCORE_RISE = 0.010
FAST_SWITCH_MIN_NEW_SCORE = 0.265
FAST_SWITCH_MIN_MARGIN = 0.030
FAST_SWITCH_MIN_LEAD = 0.015
FAST_MODE_SECONDS = 0.95

KNOWN_TO_UNKNOWN_BLOCK_SECONDS = 0.10

VIVEK_SCORE_BONUS = 0.010


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
    vals = sorted(
        [
            float(scores.get("PAVAN", 0.0)),
            float(scores.get("SUMANTH", 0.0)),
            float(scores.get("VIVEK", 0.0)),
        ],
        reverse=True,
    )
    best = vals[0]
    second = vals[1]
    margin = best - second

    if label == "UNKNOWN":
        if margin < 0.02:
            return 60
        if best < 0.15:
            return 70
        if best < 0.22:
            return 80
        return 88

    score_conf = min(max((best - 0.22) / 0.28, 0.0), 1.0)
    margin_conf = min(max(margin / 0.15, 0.0), 1.0)
    conf = int(55 + 30 * score_conf + 14 * margin_conf)
    return max(55, min(99, conf))


def emit_label_if_changed(label, ui_state, scores=None):
    if label is None:
        return

    confidence = None if scores is None else compute_confidence(label, scores)

    if ui_state["last_printed_label"] != label:
        print(speaker_line(label, confidence))
        ui_state["last_printed_label"] = label


def print_live_scores(scores, decision_label, decision_score, margin, fast_mode=False):
    decision_color = color_for_label(decision_label)
    mode_text = "FAST" if fast_mode else "STABLE"
    true_best, true_score, _, _, true_margin = get_top2_margin(scores)

    print(
        f"[LIVE-{mode_text}] "
        f"PAVAN={scores['PAVAN']:.4f} | "
        f"SUMANTH={scores['SUMANTH']:.4f} | "
        f"VIVEK={scores['VIVEK']:.4f} | "
        f"TRUE_BEST={true_best} {true_score:.4f} | "
        f"DECISION={decision_color}{decision_label} {decision_score:.4f}{RESET} | "
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

    return np.ascontiguousarray(
        np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks]),
        dtype=np.float32,
    )


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

    peak = np.max(np.abs(audio_np))

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
        return {
            "best": 0.0,
            "topk_mean": 0.0,
            "center_score": 0.0,
            "final": 0.0,
        }

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

    scores = {
        "PAVAN": pavan_scores["final"],
        "SUMANTH": sumanth_scores["final"],
        "VIVEK": vivek_scores["final"] + VIVEK_SCORE_BONUS,
    }

    return scores


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
        weight_map = {
            0.20: 0.66,
            0.32: 0.34,
        }
    elif windows == MID_WINDOWS:
        weight_map = {
            0.45: 0.58,
            0.60: 0.42,
        }
    else:
        weight_map = {
            0.75: 0.58,
            1.00: 0.42,
        }

    scores = {
        "PAVAN": 0.0,
        "SUMANTH": 0.0,
        "VIVEK": 0.0,
    }

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


def score_hierarchical_windows(classifier, reference_embeddings, live_chunks, current_label=None):
    fast_scores = score_multiwindow(
        classifier=classifier,
        reference_embeddings=reference_embeddings,
        live_chunks=live_chunks,
        windows=FAST_WINDOWS,
    )

    mid_scores = score_multiwindow(
        classifier=classifier,
        reference_embeddings=reference_embeddings,
        live_chunks=live_chunks,
        windows=MID_WINDOWS,
    )

    stable_scores = score_multiwindow(
        classifier=classifier,
        reference_embeddings=reference_embeddings,
        live_chunks=live_chunks,
        windows=STABLE_WINDOWS,
    )

    if fast_scores is None and mid_scores is None and stable_scores is None:
        return None, "NONE"

    if fast_scores is not None and mid_scores is not None:
        fast_best, fast_score, _, _, fast_margin = get_top2_margin(fast_scores)
        mid_best, mid_score, _, _, mid_margin = get_top2_margin(mid_scores)

        if (
            fast_best == mid_best
            and fast_score >= FAST_SWITCH_MIN_NEW_SCORE
            and mid_score >= FAST_SWITCH_MIN_NEW_SCORE
            and fast_margin >= FAST_SWITCH_MIN_MARGIN
            and mid_margin >= FAST_SWITCH_MIN_MARGIN
        ):
            final_scores = {
                spk: 0.68 * float(fast_scores[spk]) + 0.32 * float(mid_scores[spk])
                for spk in ("PAVAN", "SUMANTH", "VIVEK")
            }
            return final_scores, "FAST_HIERARCHY"

    if mid_scores is not None and stable_scores is not None:
        final_scores = {
            spk: 0.42 * float(mid_scores[spk]) + 0.58 * float(stable_scores[spk])
            for spk in ("PAVAN", "SUMANTH", "VIVEK")
        }
        return final_scores, "STABLE_HIERARCHY"

    if stable_scores is not None:
        return stable_scores, "STABLE_ONLY"

    if mid_scores is not None:
        return mid_scores, "MID_ONLY"

    return fast_scores, "FAST_ONLY"


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
    return START_MARGINS.get(label, 0.06)


def get_switch_margin_for_label(label):
    return SWITCH_MARGINS.get(label, 0.09)


def is_unknown_like(scores):
    true_best, true_score, _, second_score, margin = get_top2_margin(scores)

    if true_score < UNKNOWN_SCORE_FLOOR:
        return True

    if margin < UNKNOWN_MARGIN_FLOOR:
        return True

    if true_score <= UNKNOWN_STRONG_KNOWN_CAP and margin < UNKNOWN_NEAR_GAP:
        return True

    return False


def classify_frame(scores, current_label=None):
    true_best, true_score, _, _, margin = get_top2_margin(scores)

    if current_label == "UNKNOWN":
        if true_best not in UNKNOWN_TO_KNOWN_SCORE:
            return "UNKNOWN", true_score, margin

        if true_score >= UNKNOWN_TO_KNOWN_SCORE[true_best] and margin >= UNKNOWN_TO_KNOWN_MARGIN:
            return true_best, true_score, margin

        return "UNKNOWN", true_score, margin

    if current_label in ("PAVAN", "SUMANTH", "VIVEK"):
        current_score = float(scores.get(current_label, 0.0))

        if is_unknown_like(scores) and current_score < SOFT_KEEP_THRESHOLDS[current_label]:
            return "UNKNOWN", true_score, margin

        if true_best != current_label:
            if (
                true_score >= SWITCH_THRESHOLDS.get(true_best, 1.0)
                and margin >= SWITCH_MARGINS.get(true_best, 0.09)
                and true_score >= current_score + FAST_SWITCH_MIN_LEAD
            ):
                return true_best, true_score, margin

            if current_score >= SOFT_KEEP_THRESHOLDS[current_label] and margin < UNKNOWN_NEAR_GAP:
                return current_label, current_score, margin

            if is_unknown_like(scores):
                return "UNKNOWN", true_score, margin

            return current_label, current_score, margin

        if current_score >= SOFT_KEEP_THRESHOLDS[current_label]:
            return current_label, current_score, margin

        if is_unknown_like(scores):
            return "UNKNOWN", true_score, margin

        return current_label, current_score, margin

    if is_unknown_like(scores):
        return "UNKNOWN", true_score, margin

    return true_best, true_score, margin


def should_accept_start(label, score, margin):
    if label not in START_THRESHOLDS:
        return False

    return score >= START_THRESHOLDS[label] and margin >= get_start_margin_for_label(label)


def should_accept_unknown_to_known(label, score, margin):
    if label not in UNKNOWN_TO_KNOWN_SCORE:
        return False

    return score >= UNKNOWN_TO_KNOWN_SCORE[label] and margin >= UNKNOWN_TO_KNOWN_MARGIN


def should_accept_switch(label, score, margin, scores, current_label=None, fast_mode=False):
    if label not in SWITCH_THRESHOLDS:
        return False

    current_score = float(scores.get(current_label, 0.0)) if current_label else 0.0

    if score < SWITCH_THRESHOLDS[label]:
        return False

    if margin < get_switch_margin_for_label(label):
        return False

    if current_label in ("PAVAN", "SUMANTH", "VIVEK"):
        if score < current_score + FAST_SWITCH_MIN_LEAD:
            return False

    if fast_mode:
        if score < FAST_SWITCH_MIN_NEW_SCORE:
            return False

        if margin < FAST_SWITCH_MIN_MARGIN:
            return False

    return True


def should_release_to_unknown(current_label, scores, bad_frame_count):
    if current_label not in ("PAVAN", "SUMANTH", "VIVEK"):
        return False

    current_score = float(scores.get(current_label, 0.0))
    true_best, true_score, _, _, margin = get_top2_margin(scores)

    weak_current = current_score < SOFT_KEEP_THRESHOLDS[current_label]
    confused = margin < UNKNOWN_MARGIN_FLOOR
    low_known = true_score < UNKNOWN_SCORE_FLOOR
    near_unknown = true_score <= UNKNOWN_STRONG_KNOWN_CAP and margin < UNKNOWN_NEAR_GAP

    if weak_current and (confused or low_known or near_unknown):
        return True

    if bad_frame_count >= BAD_FRAMES_TO_RELEASE and (confused or low_known):
        return True

    return False


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

    true_best, true_score, _, _, margin = get_top2_margin(raw_scores)

    enter_by_drop = (
        current_drop >= FAST_SWITCH_SCORE_DROP
        and challenger_now >= FAST_SWITCH_MIN_NEW_SCORE
        and gap >= FAST_SWITCH_MIN_LEAD
    )

    enter_by_rise = (
        challenger_rise >= FAST_SWITCH_SCORE_RISE
        and challenger_now >= FAST_SWITCH_MIN_NEW_SCORE
        and gap >= FAST_SWITCH_MIN_LEAD
    )

    enter_by_clear_challenger = (
        true_best != current_label
        and true_score >= SWITCH_THRESHOLDS.get(true_best, 1.0)
        and margin >= FAST_SWITCH_MIN_MARGIN
        and true_score >= current_now + FAST_SWITCH_MIN_LEAD
    )

    enter_by_weak_current = (
        current_now < SOFT_KEEP_THRESHOLDS[current_label]
        and current_drop >= 0.008
    )

    enter = enter_by_drop or enter_by_rise or enter_by_clear_challenger or enter_by_weak_current

    return enter, challenger_label


def should_fast_switch(scores, current_label):
    if current_label not in ("PAVAN", "SUMANTH", "VIVEK"):
        return False, None

    true_best, true_score, _, _, margin = get_top2_margin(scores)

    if true_best == current_label:
        return False, None

    current_score = float(scores.get(current_label, 0.0))

    if true_score < FAST_SWITCH_MIN_NEW_SCORE:
        return False, None

    if margin < FAST_SWITCH_MIN_MARGIN:
        return False, None

    if true_score < current_score + FAST_SWITCH_MIN_LEAD:
        return False, None

    if true_score < SWITCH_THRESHOLDS.get(true_best, 1.0):
        return False, None

    return True, true_best


def reset_runtime_state():
    return {
        "is_speaking": False,
        "silence_time": 0.0,
        "silent_chunk_count": 0,
        "voice_chunk_count": 0,
        "last_live_check_time": 0.0,
        "current_label": None,
        "current_scores": {
            "PAVAN": 0.0,
            "SUMANTH": 0.0,
            "VIVEK": 0.0,
        },
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
        "segment_start_time": 0.0,
        "last_known_label": None,
        "last_known_time": 0.0,
        "transition_mode_until": 0.0,
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
    state["current_best_score"] = float(scores.get(new_label, 0.0)) if new_label != "UNKNOWN" else max(scores.values())

    _, _, _, _, state["current_margin"] = get_top2_margin(scores)

    state["pending_label"] = None
    state["pending_count"] = 0
    state["bad_frame_count"] = 0
    state["smoothed_scores"] = None
    state["last_live_signature"] = None
    state["fast_mode_candidate"] = None

    if new_label in ("PAVAN", "SUMANTH", "VIVEK"):
        state["last_known_label"] = new_label
        state["last_known_time"] = time.time()
        state["transition_mode_until"] = time.time() + KNOWN_TO_UNKNOWN_BLOCK_SECONDS
    else:
        state["transition_mode_until"] = 0.0

    if state["current_best_score"] >= 0.52:
        state["label_hold_until"] = time.time() + STRONG_LABEL_HOLD_SECONDS
    else:
        state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS

    refill_live_chunks_recent(live_chunks, recent_tail_chunks(live_chunks, POST_SWITCH_KEEP_CHUNKS))
    emit_label_if_changed(state["current_label"], ui_state, state["current_scores"])


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

max_live_chunks = int((max(STABLE_WINDOWS) * SAMPLE_RATE) / BLOCK_SIZE) + 10
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

        state["voice_chunk_count"] += 1

        if not state["is_speaking"]:
            state["silent_chunk_count"] = 0
            state["silence_time"] = 0.0

            if state["voice_chunk_count"] >= VOICE_CHUNKS_TO_START:
                state["is_speaking"] = True
                state["segment_start_time"] = time.time()
                state["current_segment_chunks"] = [c.copy() for c in pre_buffer]

                live_chunks.clear()

                for c in list(pre_buffer)[-2:]:
                    live_chunks.append(c.copy())

                state["current_label"] = None
                state["current_scores"] = {
                    "PAVAN": 0.0,
                    "SUMANTH": 0.0,
                    "VIVEK": 0.0,
                }
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
                state["last_known_label"] = None
                state["last_known_time"] = 0.0
                state["transition_mode_until"] = 0.0

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

        raw_scores, hierarchy_mode = score_hierarchical_windows(
            classifier=classifier,
            reference_embeddings=reference_embeddings,
            live_chunks=live_chunks,
            current_label=state["current_label"],
        )

        if raw_scores is None:
            state["last_live_check_time"] = now
            continue

        fast_mode = (now < state["fast_mode_until"]) or (hierarchy_mode == "FAST_HIERARCHY")

        if not fast_mode and state["current_label"] in ("PAVAN", "SUMANTH", "VIVEK"):
            enter_fast, challenger = should_enter_fast_mode(state, raw_scores)

            if enter_fast:
                state["fast_mode_until"] = now + FAST_MODE_SECONDS
                state["transition_mode_until"] = now + KNOWN_TO_UNKNOWN_BLOCK_SECONDS
                state["fast_mode_candidate"] = challenger

                refill_live_chunks_recent(
                    live_chunks,
                    recent_tail_chunks(live_chunks, FAST_MODE_KEEP_CHUNKS),
                )

                state["smoothed_scores"] = None
                state["last_live_signature"] = None
                decision_history.clear()

                fast_raw_scores = score_multiwindow(
                    classifier=classifier,
                    reference_embeddings=reference_embeddings,
                    live_chunks=live_chunks,
                    windows=FAST_WINDOWS,
                )

                if fast_raw_scores is not None:
                    raw_scores = fast_raw_scores
                    fast_mode = True

        alpha = SMOOTHING_ALPHA_FAST if fast_mode else SMOOTHING_ALPHA_STABLE
        state["smoothed_scores"] = blend_scores(state["smoothed_scores"], raw_scores, alpha)
        scores = state["smoothed_scores"].copy()

        decision_label, decision_score, margin = classify_frame(scores, state["current_label"])

        live_signature = (
            round(scores["PAVAN"], 4),
            round(scores["SUMANTH"], 4),
            round(scores["VIVEK"], 4),
            decision_label,
            round(decision_score, 4),
            round(margin, 4),
            fast_mode,
        )

        if live_signature != state["last_live_signature"]:
            print_live_scores(scores, decision_label, decision_score, margin, fast_mode=fast_mode)
            state["last_live_signature"] = live_signature

        decision_history.append(decision_label)
        voted_label = stable_vote(decision_history)

        if state["current_label"] is None:
            initial_age = time.time() - float(state.get("segment_start_time", 0.0))

            if initial_age < INITIAL_DECISION_DELAY_SECONDS:
                state["prev_raw_scores"] = raw_scores.copy()
                state["last_live_check_time"] = now
                continue

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
                    apply_switch(state, live_chunks, ui_state, "UNKNOWN", scores)

        elif state["current_label"] == "UNKNOWN":
            if voted_label in ("PAVAN", "SUMANTH", "VIVEK"):
                voted_score = float(scores.get(voted_label, 0.0))
                _, _, _, _, voted_margin = get_top2_margin(scores)

                if should_accept_unknown_to_known(voted_label, voted_score, voted_margin):
                    if state["pending_label"] == voted_label:
                        state["pending_count"] += 1
                    else:
                        state["pending_label"] = voted_label
                        state["pending_count"] = 1

                    required = UNKNOWN_TO_KNOWN_CONFIRMATIONS.get(voted_label, 2)

                    if state["pending_count"] >= required:
                        apply_switch(state, live_chunks, ui_state, voted_label, scores)
                        decision_history.clear()
                        state["fast_mode_until"] = 0.0
                else:
                    state["pending_label"] = None
                    state["pending_count"] = 0
            else:
                state["current_scores"] = scores.copy()
                state["current_best_score"] = decision_score
                state["current_margin"] = margin
                emit_label_if_changed("UNKNOWN", ui_state, state["current_scores"])

        elif state["current_label"] in ("PAVAN", "SUMANTH", "VIVEK"):
            current_label = state["current_label"]
            current_score = float(scores.get(current_label, 0.0))
            hold_active = False if fast_mode else (time.time() < state["label_hold_until"])

            if should_release_to_unknown(current_label, scores, state["bad_frame_count"]):
                state["bad_frame_count"] += 1
            else:
                state["bad_frame_count"] = max(0, state["bad_frame_count"] - 1)

            in_transition_block = time.time() < state["transition_mode_until"]

            if (
                not in_transition_block
                and decision_label == "UNKNOWN"
                and state["bad_frame_count"] >= BAD_FRAMES_TO_RELEASE
            ):
                apply_switch(state, live_chunks, ui_state, "UNKNOWN", scores)
                decision_history.clear()
                state["fast_mode_until"] = 0.0
                state["last_live_check_time"] = now
                state["prev_raw_scores"] = raw_scores.copy()
                continue

            can_fast_switch, challenger_label = should_fast_switch(scores, current_label)

            if can_fast_switch and (fast_mode or not hold_active):
                challenger_score = float(scores.get(challenger_label, 0.0))
                _, _, _, _, switch_margin = get_top2_margin(scores)

                if should_accept_switch(
                    challenger_label,
                    challenger_score,
                    switch_margin,
                    scores,
                    current_label=current_label,
                    fast_mode=fast_mode,
                ):
                    apply_switch(state, live_chunks, ui_state, challenger_label, scores)
                    decision_history.clear()
                    state["fast_mode_until"] = 0.0
                    state["last_live_check_time"] = now
                    state["prev_raw_scores"] = raw_scores.copy()
                    continue

            if (
                voted_label in ("PAVAN", "SUMANTH", "VIVEK")
                and voted_label != current_label
                and (fast_mode or not hold_active)
            ):
                voted_score = float(scores.get(voted_label, 0.0))
                _, _, _, _, voted_margin = get_top2_margin(scores)

                if should_accept_switch(
                    voted_label,
                    voted_score,
                    voted_margin,
                    scores,
                    current_label=current_label,
                    fast_mode=fast_mode,
                ):
                    apply_switch(state, live_chunks, ui_state, voted_label, scores)
                    decision_history.clear()
                    state["fast_mode_until"] = 0.0
                    state["last_live_check_time"] = now
                    state["prev_raw_scores"] = raw_scores.copy()
                    continue

            if decision_label == current_label and current_score >= KEEP_THRESHOLDS[current_label]:
                state["current_scores"] = scores.copy()
                state["current_best_score"] = current_score

                _, _, _, _, current_margin = get_top2_margin(scores)
                state["current_margin"] = current_margin

                state["pending_label"] = None
                state["pending_count"] = 0
                state["bad_frame_count"] = 0
                state["last_known_label"] = current_label
                state["last_known_time"] = time.time()

                if current_score >= 0.52:
                    state["label_hold_until"] = time.time() + STRONG_LABEL_HOLD_SECONDS
                else:
                    state["label_hold_until"] = time.time() + LABEL_HOLD_SECONDS

            else:
                true_best, true_score, _, _, true_margin = get_top2_margin(scores)

                if true_best != current_label and true_score >= SWITCH_THRESHOLDS.get(true_best, 1.0):
                    state["bad_frame_count"] += 1
                elif current_score < SOFT_KEEP_THRESHOLDS[current_label]:
                    state["bad_frame_count"] += 1
                else:
                    state["bad_frame_count"] = max(0, state["bad_frame_count"] - 1)

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