import torch
import torchaudio
from torch import nn


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param: int = 27, num_masks: int = 2) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        fill_value = spectrogram.mean()
        for _ in range(self.num_masks):
            spectrogram = self._aug(spectrogram, mask_value=fill_value)
        return spectrogram
