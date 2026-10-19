import torch
from torch import nn
from torchaudio.functional import rnnt_loss


class RNNTLossWrapper(nn.Module):
    def __init__(self, blank_id: int = 0, reduction: str = "mean") -> None:
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        text_encoded: torch.Tensor,
        text_encoded_length: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        targets = text_encoded.to(dtype=torch.int32, device=logits.device)
        logit_lengths = logit_lengths.to(dtype=torch.int32, device=logits.device)
        target_lengths = text_encoded_length.to(dtype=torch.int32, device=logits.device)

        loss = rnnt_loss(
            logits,
            targets,
            logit_lengths,
            target_lengths,
            blank=self.blank_id,
            reduction=self.reduction,
        )
        return {"loss": loss}
