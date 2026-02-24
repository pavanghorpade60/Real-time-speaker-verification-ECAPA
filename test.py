import torch
import sounddevice as sd
from speechbrain.inference import EncoderClassifier

# -------------------------------------------------
# Settings
# -------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 7                # Slightly longer for stability
THRESHOLD = 0.52            # Tune if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Load Model
# -------------------------------------------------
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# -------------------------------------------------
# Load Enrollment Embeddings
# -------------------------------------------------
enrollment_embeddings = torch.load("pavan_embeddings.pt").to(DEVICE)
enrollment_embeddings = torch.nn.functional.normalize(enrollment_embeddings, dim=1)

print("\n🔐 Speaker Verification (Single Sample Mode)")
print("------------------------------------------------")

input("\nPress Enter to start recording...")
print("🎤 Recording... Speak clearly and continuously!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

print("✅ Recording complete.")

waveform = torch.tensor(audio.T).to(DEVICE)

# Normalize amplitude
max_val = torch.max(torch.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val

# Energy check
energy = torch.mean(torch.abs(waveform)).item()
print(f"🔎 Audio Energy: {energy:.4f}")

if energy < 0.02:
    print("⚠️ Audio too quiet. Please speak louder.")
    exit()

# -------------------------------------------------
# Extract Embedding
# -------------------------------------------------
with torch.no_grad():
    emb = classifier.encode_batch(waveform)
    emb = torch.nn.functional.normalize(emb, dim=2)

live_vec = emb.squeeze(0).squeeze(0)

# -------------------------------------------------
# Compare Against ALL Enrollment Samples
# -------------------------------------------------
scores = torch.nn.functional.cosine_similarity(
    enrollment_embeddings,
    live_vec.unsqueeze(0).repeat(enrollment_embeddings.size(0), 1)
)

average_score = scores.mean().item()
max_score = scores.max().item()

print(f"\n📊 Average Similarity: {average_score:.4f}")
print(f"📊 Max Similarity:     {max_score:.4f}")

# -------------------------------------------------
# Decision Logic
# -------------------------------------------------
if average_score >= THRESHOLD:
    print("🗣️ Speaker: Pavan")
else:
    print("🗣️ Speaker: Unknown")