import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# -------------------------------------------------
# Squeeze-and-Excitation Module
# -------------------------------------------------
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(channels, bottleneck)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(bottleneck, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


# -------------------------------------------------
# TDNN Block
# -------------------------------------------------
class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super(TDNNBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# -------------------------------------------------
# ECAPA-TDNN Embedding Model
# -------------------------------------------------
class ECAPA_TDNN(nn.Module):
    def __init__(self, embedding_dim=192):
        super(ECAPA_TDNN, self).__init__()

        self.sample_rate = 16000

        # Log-Mel Feature Extractor
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )

        self.tdnn1 = TDNNBlock(80, 512, kernel_size=5)
        self.tdnn2 = TDNNBlock(512, 512, kernel_size=3, dilation=2)
        self.tdnn3 = TDNNBlock(512, 512, kernel_size=3, dilation=3)
        self.tdnn4 = TDNNBlock(512, 512, kernel_size=1)

        self.se = SEModule(512)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, embedding_dim)

    def preprocess_audio(self, waveform, sr):
        """
        Ensures:
        - Mono audio
        - 16kHz sampling rate
        """

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def forward(self, x, sr=16000):
        """
        x: waveform tensor
        shape: (batch, time) OR (batch, 1, time)
        """

        # If input is (batch, 1, time) → squeeze channel
        if x.dim() == 3:
            x = x.squeeze(1)

        # Extract features
        x = self.mel(x)
        x = torch.log(x + 1e-6)

        # TDNN layers
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)

        # SE attention
        x = self.se(x)

        # Global pooling
        x = self.pool(x).squeeze(-1)

        # Final embedding
        embedding = self.fc(x)

        # L2 normalize (VERY IMPORTANT for cosine similarity)
        return F.normalize(embedding, p=2, dim=1)