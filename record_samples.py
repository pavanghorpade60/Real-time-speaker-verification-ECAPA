import os
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 4
NUM_SAMPLES = 40

OUTPUT_FOLDER = "data/pavan/processed_audio"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(" Recording new enrollment samples...\n")

for i in range(1, NUM_SAMPLES + 1):

    input(f"\nPress Enter to record sample {i}...")

    print("🎤 Recording...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    file_path = os.path.join(OUTPUT_FOLDER, f"pavan_{i:02d}.wav")
    sf.write(file_path, audio, SAMPLE_RATE)

    print(f"✅ Saved: pavan_{i:02d}.wav")

print("\n All enrollment samples recorded successfully!")