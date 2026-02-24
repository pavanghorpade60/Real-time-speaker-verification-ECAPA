import torch
import sounddevice as sd
from speechbrain.inference import EncoderClassifier

# -------------------------------------------------
# Settings
# -------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 6
NUM_TRIALS = 3
THRESHOLD = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

enrollment_embeddings = torch.load("pavan_embeddings.pt").to(DEVICE)
enrollment_embeddings = torch.nn.functional.normalize(enrollment_embeddings, dim=1)

print("\n🔐 Speaker Verification System")
print("--------------------------------")

live_embeddings = []

for i in range(NUM_TRIALS):
    input(f"\nPress Enter to record sample {i+1}...")
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
        print("⚠️ Audio too quiet. Try again.")
        continue

    with torch.no_grad():
        emb = classifier.encode_batch(waveform)
        emb = torch.nn.functional.normalize(emb, dim=2)

    live_embeddings.append(emb.squeeze(0).squeeze(0))

if len(live_embeddings) == 0:
    print("❌ No valid recordings captured.")
    exit()

live_vec = torch.stack(live_embeddings).mean(dim=0)
live_vec = torch.nn.functional.normalize(live_vec.unsqueeze(0), dim=1).squeeze(0)

scores = torch.nn.functional.cosine_similarity(
    enrollment_embeddings,
    live_vec.expand_as(enrollment_embeddings)
)

average_score = scores.mean().item()
max_score = scores.max().item()

print(f"\n📊 Average Similarity: {average_score:.4f}")
print(f"📊 Max Similarity:     {max_score:.4f}")

if average_score >= THRESHOLD:
    print("🗣️ Speaker: Pavan")
else:
    print("🗣️ Speaker: Unknown")