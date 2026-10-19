import torch
from torch import nn


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        embedding_dim: int = 144,
        hidden_size: int = 320,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeded = self.embedding(tokens)
        features, state = self.lstm(embeded, hx=state)
        return features, state
