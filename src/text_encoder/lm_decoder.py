from pathlib import Path

from pyctcdecode import build_ctcdecoder


class CTCLMDecoder:
    def __init__(
        self, text_encoder, lm_path=None, unigrams_path=None, alpha=0.5, beta=1.0
    ):
        self.text_encoder = text_encoder
        labels = [text_encoder[i] for i in range(len(text_encoder))]
        labels[0] = ""
        unigrams = None

        if unigrams_path is not None:
            unigrams = Path(unigrams_path).read_text(encoding="utf-8").splitlines()

        self.decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=str(lm_path) if lm_path is not None else None,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )

    def decode(self, log_probs, beam_size=50):
        scores = log_probs.detach().cpu().float().numpy()
        text = self.decoder.decode(scores, beam_width=beam_size)
        return self.text_encoder.normalize_text(text).strip()
