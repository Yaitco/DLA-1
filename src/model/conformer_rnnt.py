from pathlib import Path

import torch
from torch import nn

from src.model.conformer_encoder import ConformerEncoder
from src.model.joint_network import JointNetwork
from src.model.prediction_network import PredictionNetwork


class ConformerRNNT(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        d_model: int = 144,
        num_layers: int = 16,
        num_heads: int = 4,
        expansion_factor: int = 4,
        kernel_size: int = 31,
        conv_channels: int = 32,
        dropout: float = 0.1,
        embedding_dim: int = 144,
        predictor_hidden_size: int = 320,
        predictor_num_layers: int = 1,
        joint_dim: int = 256,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.blank_id = blank_id

        self.encoder = ConformerEncoder(
            n_feats=n_feats,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            conv_channels=conv_channels,
            dropout=dropout,
        )

        self.predictor = PredictionNetwork(
            n_tokens=n_tokens,
            embedding_dim=embedding_dim,
            hidden_size=predictor_hidden_size,
            num_layers=predictor_num_layers,
        )

        self.joint = JointNetwork(
            n_tokens=n_tokens,
            encoder_dim=d_model,
            predictor_dim=predictor_hidden_size,
            joint_dim=joint_dim,
        )

    def load_encoder_from_ctc(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        allowed_prefix = ["subsampling.", "blocks."]
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not any(key.startswith(prefix) for prefix in allowed_prefix):
                state_dict.pop(key)

        self.encoder.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        spectrogram: torch.Tensor,
        spectrogram_length: torch.Tensor,
        text_encoded: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        encoder_features, lengths = self.encoder(spectrogram, spectrogram_length)

        blank = torch.empty(
            (text_encoded.shape[0], 1),
            device=text_encoded.device,
            dtype=text_encoded.dtype,
        )
        blank.fill_(self.blank_id)

        text_encoded = torch.cat((blank, text_encoded), dim=-1)

        predictor_features, _ = self.predictor(text_encoded)

        logits = self.joint(encoder_features, predictor_features)
        return {
            "logits": logits,
            "logit_lengths": lengths,
            "encoder_features": encoder_features,
        }

    @torch.no_grad()
    def greedy_decode(
        self,
        spectrogram: torch.Tensor,
        spectrogram_length: torch.Tensor,
        max_symbols_per_step: int = 5,
    ) -> list[list[int]]:
        features, lengths = self.encoder(spectrogram, spectrogram_length)
        return self.greedy_decode_features(features, lengths, max_symbols_per_step)

    @torch.no_grad()
    def greedy_decode_features(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        max_symbols_per_step: int = 5,
    ) -> list[list[int]]:
        batch_size, time_steps, _ = features.shape
        lengths = lengths.to(features.device)

        tokens = torch.full(
            (batch_size, 1), self.blank_id, dtype=torch.long, device=features.device
        )

        predictor_features, state = self.predictor(tokens)

        hypotheses = torch.full(
            (batch_size, time_steps, max_symbols_per_step),
            self.blank_id,
            dtype=torch.long,
            device=features.device,
        )

        for t in range(time_steps):
            active = lengths > t

            frame = features[:, t : t + 1]
            for symbol in range(max_symbols_per_step):
                logits = self.joint(frame, predictor_features)

                token_ids = logits[:, 0, 0].argmax(dim=-1)

                active = active & (token_ids != self.blank_id)
                if not active.any():
                    break

                hypotheses[:, t, symbol] = torch.where(active, token_ids, self.blank_id)

                next_features, next_state = self.predictor(token_ids[:, None], state)
                predictor_features = torch.where(
                    active[:, None, None], next_features, predictor_features
                )

                state = tuple(
                    torch.where(active[None, :, None], new, old)
                    for new, old in zip(next_state, state)
                )

        return [
            [token for token in row if token != self.blank_id]
            for row in hypotheses.flatten(1).cpu().tolist()
        ]
