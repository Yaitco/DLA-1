import torch
import torchaudio
from torch import nn


class TimeMasking(nn.Module):
    def __init__(
        self,
        time_mask_param: int = 1000,
        max_time_ratio: float = 0.05,
        num_masks: int = 10,
    ) -> None:
        super().__init__()
        self.time_mask_param = time_mask_param
        self.max_time_ratio = max_time_ratio
        self.num_masks = num_masks

        self._aug = torchaudio.transforms.TimeMasking(time_mask_param, p=max_time_ratio)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        fill_value = spectrogram.mean()
        for _ in range(self.num_masks):
            spectrogram = self._aug(spectrogram, mask_value=fill_value)
        return spectrogram
