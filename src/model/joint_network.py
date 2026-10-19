import torch
from torch import nn


class JointNetwork(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        encoder_dim: int = 144,
        predictor_dim: int = 320,
        joint_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)

        self.predictor_proj = nn.Linear(predictor_dim, joint_dim)

        self.activation = nn.Tanh()

        self.output_proj = nn.Linear(joint_dim, n_tokens)

    def forward(
        self,
        encoder_features: torch.Tensor,
        predictor_features: torch.Tensor,
    ) -> torch.Tensor:
        encoder_features = self.encoder_proj(encoder_features)
        predictor_features = self.predictor_proj(predictor_features)

        encoder_features.unsqueeze_(2)
        predictor_features.unsqueeze_(1)

        features = self.activation(encoder_features + predictor_features)

        return self.output_proj(features)
