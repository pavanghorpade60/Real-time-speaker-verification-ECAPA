import os
import torch
from speechbrain.inference import EncoderClassifier
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

AUDIO_FOLDER = "data/pavan/processed_audio"

embeddings = []

for file in sorted(os.listdir(AUDIO_FOLDER)):
    if file.endswith(".wav"):
        path = os.path.join(AUDIO_FOLDER, file)

        waveform, sr = torchaudio.load(path)
        waveform = waveform.to(DEVICE)

        with torch.no_grad():
            emb = classifier.encode_batch(waveform)
            emb = torch.nn.functional.normalize(emb, dim=2)

        embeddings.append(emb.squeeze(0).squeeze(0))
        print(f"Processed: {file}")

if len(embeddings) == 0:
    raise ValueError("No audio files found!")

# Stack ALL embeddings (not average)
all_embeddings = torch.stack(embeddings)

torch.save(all_embeddings, "pavan_embeddings.pt")

print("\nEnrollment embeddings saved successfully!")
print("Total samples used:", len(embeddings))