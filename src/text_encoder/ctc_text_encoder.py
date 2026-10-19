import re
from string import ascii_lowercase

import numpy as np
import torch

from src.text_encoder.beam import Beam


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor(
                [self.char2ind[char] for char in text], dtype=torch.long
            ).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result = ""
        pred = None
        for ind in inds:
            c = self.ind2char[int(ind)]
            if c != pred:
                result += c
            pred = c
        return result

    def ctc_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_size: int = 20,
        token_topk: int | None = 10,
    ) -> str:
        # Input: [T, len(vocab)], log probabilities for one unpadded recording.
        # Output: decoded text.

        log_probs = log_probs.detach().cpu()
        k = (
            len(self.vocab) - 1
            if token_topk is None
            else min(token_topk, len(self.vocab) - 1)
        )

        beam = Beam()
        beam.add("", bp=0)

        for t in range(len(log_probs)):
            values, indices = torch.topk(log_probs[t, 1:], k)
            values, indices = values.tolist(), (indices + 1).tolist()
            values.append(log_probs[t, 0].item())
            indices.append(0)

            new_beam = Beam()
            for val, ind in zip(values, indices):
                c = self.ind2char[int(ind)]

                for i in range(len(beam)):
                    s, pb, pnb = beam.get(i)

                    if c == self.EMPTY_TOK:
                        new_beam.add(s=s, bp=np.logaddexp(pb, pnb) + val)
                        continue
                    pred = s[-1] if s else ""
                    if pred != c:
                        new_beam.add(s=s + c, nbp=np.logaddexp(pb, pnb) + val)
                    elif pred == c:
                        new_beam.add(s=s + c, nbp=pb + val)

                        new_beam.add(s=s, nbp=pnb + val)

            new_beam.keep_k(beam_size)
            beam = new_beam

        return beam.get(0)[0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
