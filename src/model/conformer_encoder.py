import torch
from torch import nn

from src.model.conformer import ConformerBlock, Conv2dSubsampling


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_feats: int,
        d_model: int = 144,
        num_layers: int = 16,
        num_heads: int = 4,
        expansion_factor: int = 4,
        kernel_size: int = 31,
        conv_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsampling = Conv2dSubsampling(n_feats, d_model, conv_channels)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    num_heads,
                    expansion_factor,
                    kernel_size,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        spectrogram_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, lengths = self.subsampling(spectrogram, spectrogram_length)

        positions = torch.arange(features.shape[1], device=features.device)
        padding_mask = positions.unsqueeze(0) >= lengths.to(features.device).unsqueeze(
            1
        )

        for block in self.blocks:
            features = block(features, padding_mask)

        return features, lengths
