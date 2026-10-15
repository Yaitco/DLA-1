import torch
import torch.nn.functional as F
from torch import nn


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_feats: int, d_model: int, conv_channels: int = 32) -> None:
        super().__init__()

        self.n_feats = n_feats
        self.d_model = d_model
        self.conv_channels = conv_channels

        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2)

        F_out = self.transform_input_lengths(n_feats)
        self.linear = nn.Linear(conv_channels * F_out, d_model)

    @staticmethod
    def transform_input_lengths(
        input_lengths: int | torch.Tensor,
    ) -> int | torch.Tensor:
        single_transform = lambda x: (x - 3) // 2 + 1
        return single_transform(single_transform(input_lengths))

    def forward(
        self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            (spectrogram_length >= 7) & (spectrogram_length <= spectrogram.shape[-1])
        ).all(), "Lengths must be between 7 and T"

        spectrogram = spectrogram.unsqueeze(1)

        spectrogram = F.relu(self.conv1(spectrogram))
        spectrogram = F.relu(self.conv2(spectrogram))
        features = spectrogram.permute(0, 3, 1, 2).flatten(2)
        features = self.linear(features)

        lengths = self.transform_input_lengths(spectrogram_length)
        positions = torch.arange(features.shape[1], device=features.device)
        
        
        padding_mask = positions.unsqueeze(0) >= lengths.to(features.device).unsqueeze(1)
        features = features.masked_fill(padding_mask.unsqueeze(-1), 0)

        return features, lengths


class FeedForwardModule(nn.Module):
    def __init__(
        self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)

        self.swish = nn.SiLU()

        self.dropout_ = nn.Dropout(dropout, inplace=True)

        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        features = self.ln(features)
        
        features = self.swish(self.linear1(features))
        self.dropout_(features)
        
        features = self.linear2(features)
        self.dropout_(features)

        return features


class SelfAttentionModule(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(d_model)

        self.attention = RelativeMultiheadAttention(
            d_model, 
            num_heads=num_heads, 
            dropout=dropout,
        )

        self.dropout_ = nn.Dropout(dropout, inplace=True)
        self.positional = RelativePositionalEncoding(d_model)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        
        normalized = self.ln(features)

        table = self.positional(features)
        out = self.attention(
            normalized,
            table,
            padding_mask, 
        )

        self.dropout_(out)
        out = out.masked_fill(padding_mask.unsqueeze(-1), 0)
        return out


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Input: [B, T, D]. Output: [2 * T - 1, D]. Use even D.
        B, T, D = features.shape
        assert D == self.d_model

        distances = torch.arange(-(T - 1), T, device=features.device)

        frequencies = -torch.arange(0, D, 2, device=features.device) / D
        frequencies = 10000 ** frequencies

        angles = distances.unsqueeze(-1) * frequencies.unsqueeze(0)

        table = torch.zeros((2 * T - 1, D), device=features.device, dtype=features.dtype)

        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)
        return table


class RelativeMultiheadAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()

        assert num_heads > 0 and d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

        self.u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, num_heads, l, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, num_heads * head_dim)

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # features: [B, T, D]; positions: [2T-1, D].
        # padding_mask: bool [B, T], True for padding; output: [B, T, D].
        q = self.q_proj(features)
        k = self.k_proj(features)
        v = self.v_proj(features)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        positions = self.pos_proj(positions)
        positions = self._split_heads(positions.unsqueeze(0)).squeeze(0)

        content_scores = (q + self.u[None, :, None, :]) @ k.transpose(-2, -1) 
        position_scores = (q + self.v[None, :, None, :]) @ positions.transpose(-2, -1)

        T = content_scores.shape[-1]
        arange = torch.arange(T, device=content_scores.device) 
        idx = T - 1 + arange[:, None] - arange[None, :] 
        selected_position_scores = position_scores[:, :, arange[:, None], idx]

        scores = (content_scores + selected_position_scores) / (self.head_dim ** 0.5)

        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = self._merge_heads(scores @ v)
        out = self.out_proj(out)

        out = out.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        return out


class ConvolutionModule(nn.Module):
    def __init__(
        self, d_model: int, kernel_size: int = 31, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1

        self.ln = nn.LayerNorm(d_model)

        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        self.dw_conv = nn.Conv1d(
            d_model, d_model, 
            kernel_size, 
            groups=d_model, 
            padding=kernel_size // 2
        )

        self.bn = nn.BatchNorm1d(d_model)
        self.silu = nn.SiLU()

        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # features: [B, T, D]; padding_mask: bool [B, T], True for padding.
        # Output: [B, T, D].

        features = self.ln(features).transpose(-2, -1)

        features = self.pw_conv1(features)
        features = self.glu(features)

        features = features.masked_fill(padding_mask.unsqueeze(1), 0)

        features = self.dw_conv(features)
        features = self.bn(features)
        features = self.silu(features)
        features = self.pw_conv2(features)

        features = self.dropout(features).transpose(-2, -1)
        features = features.masked_fill(padding_mask.unsqueeze(-1), 0)
        return features


class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        d_model: int = 128,
        num_layers: int = 2,
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
                    d_model, num_heads,
                    expansion_factor,
                    kernel_size, dropout,
                ) 
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(d_model, n_tokens)

    def forward(
        self,
        spectrogram: torch.Tensor,
        spectrogram_length: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        # Input: spectrogram [B, F, T], spectrogram_length [B].
        # Output: {"log_probs": [B, T_out, V], "log_probs_length": [B]}.
        features, lengths = self.subsampling(spectrogram, spectrogram_length)

        positions = torch.arange(features.shape[1], device=features.device)
        padding_mask = positions.unsqueeze(0) >= lengths.to(features.device).unsqueeze(1)

        for block in self.blocks:
            features = block(features, padding_mask)

        out = self.linear(features)
        out = F.log_softmax(out, dim=-1)
        return {"log_probs": out, "log_probs_length": lengths}

class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        expansion_factor: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(
            d_model, 
            expansion_factor=expansion_factor, 
            dropout=dropout
        )
        self.ffn2 = FeedForwardModule(
            d_model, 
            expansion_factor=expansion_factor, 
            dropout=dropout
        )

        self.attention = SelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)

        self.ln = nn.LayerNorm(d_model)

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # Input/output: [B, T, D]; padding_mask: bool [B, T].

        features = features + 0.5 * self.ffn1(features)

        features = features + self.attention(features, padding_mask)

        features = features + self.conv(features, padding_mask)

        features = features + 0.5 * self.ffn2(features)

        features = self.ln(features)

        features = features.masked_fill(padding_mask.unsqueeze(-1), 0)
        return features
