import math

import torch
import torchaudio.functional as F
from torch import nn


class Reverb(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        rt60_seconds: tuple[float, float] = (0.1, 0.3),
        wet: float = 0.2,
        p: float = 0.3,
        tail_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.rt60_seconds = rt60_seconds
        self.wet = wet
        self.p = p
        self.tail_strength = tail_strength

    def _generate_rir(self, audio: torch.Tensor) -> torch.Tensor:
        rt60 = torch.empty(()).uniform_(*self.rt60_seconds).item()
        L = round(rt60 * self.sample_rate) + 1

        t = torch.arange(L, device=audio.device) / self.sample_rate
        noise = torch.randn((1, L), device=audio.device, dtype=audio.dtype)

        alpha = 3 * math.log(10) / rt60
        tail = noise * torch.exp(-alpha * t)
        tail[:, 0] = 0

        tail /= torch.norm(tail) + 1e-4
        tail *= self.tail_strength
        tail[0, 0] = 1

        return tail
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.p < torch.rand(1):
            return audio
        rir = self._generate_rir(audio)
        reverberated = F.fftconvolve(audio, rir, mode="full")
        reverberated = reverberated[:, :audio.shape[-1]]
        audio = (1 - self.wet) * audio + self.wet * reverberated
        return audio
