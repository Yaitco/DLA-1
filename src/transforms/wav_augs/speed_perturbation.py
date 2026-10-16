import torch
from torch import nn
import torchaudio


class SpeedPerturbation(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        speeds: tuple[float, ...] = (0.9, 1.1),
        p: float = 0.3,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.speeds = speeds
        self.p = p

        self._aug = torchaudio.transforms.SpeedPerturbation(sample_rate, speeds)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.p < torch.rand(1):
            return audio
        audio, _ = self._aug(audio) 
        return audio
