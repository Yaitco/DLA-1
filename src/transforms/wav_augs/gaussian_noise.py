import torch
import torchaudio
from torch import nn


class AddGaussianNoise(nn.Module):
    def __init__(
        self, min_snr_db: float = 15.0, max_snr_db: float = 30.0, p: float = 0.5
    ) -> None:
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p
        self._aug = torchaudio.transforms.AddNoise()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.p < torch.rand(1):
            return audio

        noise = torch.randn_like(audio)
        snr = torch.empty(1, device=audio.device, dtype=audio.dtype)
        snr.uniform_(self.min_snr_db, self.max_snr_db)
        audio = self._aug(audio, noise, snr)
        return audio
