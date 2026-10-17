from collections.abc import Sequence
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from src.text_encoder.ctc_text_encoder import Beam, CTCTextEncoder


class BPETextEncoder:
    def __init__(self, model_path: str | Path) -> None:
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
        assert self.tokenizer.pad_id() == 0

    def __len__(self) -> int:
        return self.tokenizer.vocab_size()

    def __getitem__(self, token_id: int) -> str:
        return self.tokenizer.id_to_piece(int(token_id))

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(self.normalize_text(text), out_type=int)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, inds: Sequence[int] | torch.Tensor) -> str:
        ids = [int(ind) for ind in inds if int(ind) != 0]
        return self.tokenizer.decode(ids)

    def ctc_decode(self, inds: Sequence[int] | torch.Tensor) -> str:
        result = []
        pred = None
        for ind in inds:
            ind = int(ind)
            if ind != pred and ind != 0:
                result.append(ind)
            pred = ind
        return self.decode(result)

    def ctc_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_size: int = 20,
        token_topk: int | None = 10,
    ) -> str:
        log_probs = log_probs.detach().cpu()
        k = len(self) - 1 if token_topk is None else min(token_topk, len(self) - 1)

        beam = Beam()
        beam.add((), bp=0)

        for t in range(len(log_probs)):
            values, indices = torch.topk(log_probs[t, 1:], k)
            values, indices = values.tolist(), (indices + 1).tolist()
            values.append(log_probs[t, 0].item())
            indices.append(0)

            new_beam = Beam()
            for val, ind in zip(values, indices):
                for i in range(len(beam)):
                    s, pb, pnb = beam.get(i)

                    if ind == 0:
                        new_beam.add(s=s, bp=np.logaddexp(pb, pnb) + val)
                        continue
                    pred = s[-1] if s else None
                    if pred != ind:
                        new_beam.add(s=s + (ind,), nbp=np.logaddexp(pb, pnb) + val)
                    else:
                        new_beam.add(s=s + (ind,), nbp=pb + val)
                        new_beam.add(s=s, nbp=pnb + val)

            new_beam.keep_k(beam_size)
            beam = new_beam

        return self.decode(beam.get(0)[0])

    @staticmethod
    def normalize_text(text: str) -> str:
        return CTCTextEncoder.normalize_text(text)
