import torch
import torchaudio
import torch.nn.functional as F

TARGET_SR = 16000


# -------------------------------------------------
# Load Audio File
# - Converts to mono
# - Resamples to 16kHz
# - Returns (1, time) tensor
# -------------------------------------------------
def load_audio(path, device="cpu"):
    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    waveform = normalize_audio(waveform)

    return waveform.to(device)


# -------------------------------------------------
# Normalize Audio (Peak normalization)
# -------------------------------------------------
def normalize_audio(audio):
    max_val = torch.max(torch.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


# -------------------------------------------------
# Extract Embedding from Model
# -------------------------------------------------
@torch.no_grad()
def extract_embedding(model, waveform, device="cpu"):
    """
    waveform: (1, time)
    returns: (1, embedding_dim)
    """
    model.eval()
    waveform = waveform.to(device)

    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # (1, 1, time)

    embedding = model(waveform)

    return embedding


# -------------------------------------------------
# Cosine Similarity
# -------------------------------------------------
def cosine_similarity(emb1, emb2):
    """
    emb1, emb2: (1, embedding_dim)
    returns similarity score (float)
    """
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    similarity = torch.sum(emb1 * emb2, dim=1)
    return similarity.item()


# -------------------------------------------------
# Decision Function
# -------------------------------------------------
def identify_speaker(live_embedding, stored_embedding, threshold=0.75):
    """
    Returns:
        "Pavan" or "Unknown"
    """
    score = cosine_similarity(live_embedding, stored_embedding)

    if score >= threshold:
        return "Pavan", score
    else:
        return "Unknown", score