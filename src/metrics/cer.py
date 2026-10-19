from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class RNNTCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, predictions: list[str], text: list[str], **kwargs) -> float:
        cers = []
        for prediction, target in zip(predictions, text):
            cers.append(
                calc_cer(self.text_encoder.normalize_text(target).strip(), prediction)
            )
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size, token_topk, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.token_topk = token_topk

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = []
        for i in range(len(log_probs)):
            prediction = self.text_encoder.ctc_beam_search(
                log_probs[i, : int(log_probs_length[i])],
                beam_size=self.beam_size,
                token_topk=self.token_topk,
            )
            predictions.append(prediction)

        for pred_text, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
