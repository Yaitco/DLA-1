import math
from dataclasses import dataclass, replace

import kenlm
import numpy as np
import torch


@dataclass
class Hypothesis:
    tokens: tuple[int, ...] = ()
    score: float = 0.0
    lm_state: object = None
    word: str = ""
    lm_score: float = 0.0


class RNNTBeamDecoder:
    def __init__(
        self,
        model,
        text_encoder,
        beam_size: int = 4,
        token_topk: int = 4,
        max_symbols_per_step: int = 5,
        lm_path=None,
        alpha: float = 0.1,
        beta: float = 0.5,
    ) -> None:
        if min(beam_size, token_topk, max_symbols_per_step) < 1:
            raise ValueError("Beam size, token top-k and symbol limit must be positive")
        self.model = model
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.token_topk = min(token_topk, len(text_encoder) - 1)
        self.max_symbols_per_step = max_symbols_per_step
        self.lm = kenlm.Model(str(lm_path)) if lm_path is not None else None
        self.alpha = alpha
        self.beta = beta
        self.pieces = [text_encoder[i] for i in range(len(text_encoder))]
        assert model.blank_id == 0

    def _score_word(self, state, word):
        next_state = kenlm.State()
        score = self.lm.BaseScore(state, word, next_state)
        return next_state, self.alpha * math.log(10) * score + self.beta

    def _extend_lm(self, hypothesis, token):
        if self.lm is None:
            return None, "", 0.0
        state, word, score = (
            hypothesis.lm_state,
            hypothesis.word,
            hypothesis.lm_score,
        )
        piece = self.pieces[token]
        if piece == "<unk>":
            piece = "▁<unk>▁"
        for i, fragment in enumerate(piece.replace("▁", " ").split(" ")):
            if i and word:
                state, increment = self._score_word(state, word)
                score += increment
                word = ""
            word += fragment
        return state, word, score

    def _finish_lm(self, hypothesis):
        if self.lm is None:
            return 0.0
        state, score = hypothesis.lm_state, hypothesis.lm_score
        if hypothesis.word:
            state, increment = self._score_word(state, hypothesis.word)
            score += increment
        end_score = self.lm.BaseScore(state, "</s>", kenlm.State())
        return score + self.alpha * math.log(10) * end_score

    @staticmethod
    def _add(pool, hypothesis):
        if hypothesis.tokens in pool:
            previous = pool[hypothesis.tokens]
            previous.score = float(np.logaddexp(previous.score, hypothesis.score))
        else:
            pool[hypothesis.tokens] = hypothesis

    def _prune(self, pool):
        return sorted(pool.values(), key=lambda h: h.score + h.lm_score, reverse=True)[
            : self.beam_size
        ]

    def _predict(self, hypotheses, cache, device):
        missing = [h.tokens for h in hypotheses if h.tokens not in cache]
        if missing:
            tokens = torch.tensor([[prefix[-1]] for prefix in missing], device=device)
            parents = [cache[prefix[:-1]][1] for prefix in missing]
            state = tuple(
                torch.cat([parent[i] for parent in parents], dim=1) for i in range(2)
            )
            prediction, state = self.model.predictor(tokens, state)
            for i, prefix in enumerate(missing):
                cache[prefix] = (
                    prediction[i : i + 1],
                    tuple(value[:, i : i + 1] for value in state),
                )
        return torch.cat([cache[h.tokens][0] for h in hypotheses], dim=0)

    @torch.no_grad()
    def _search(self, features: torch.Tensor) -> list[Hypothesis]:
        # features: [T_valid, encoder_dim], without padding; model must be in eval mode.
        prediction, state = self.model.predictor(
            torch.tensor([[self.model.blank_id]], device=features.device)
        )
        cache = {(): (prediction, state)}
        start = None
        if self.lm is not None:
            start = kenlm.State()
            self.lm.BeginSentenceWrite(start)
        beam = [Hypothesis(lm_state=start)]

        for frame in features:
            active = beam
            completed = {}
            for emitted in range(self.max_symbols_per_step + 1):
                prediction = self._predict(active, cache, features.device)
                scores = self.model.joint(frame[None, None], prediction)[:, 0, 0]
                scores = scores.log_softmax(-1)
                blanks = scores[:, 0].tolist()
                candidates = {}
                if emitted < self.max_symbols_per_step:
                    values, tokens = scores[:, 1:].topk(self.token_topk, dim=-1)
                    values, tokens = values.tolist(), (tokens + 1).tolist()
                for i, hypothesis in enumerate(active):
                    self._add(
                        completed,
                        replace(hypothesis, score=hypothesis.score + blanks[i]),
                    )
                    if emitted == self.max_symbols_per_step:
                        continue
                    for token, value in zip(tokens[i], values[i]):
                        lm_state, word, lm_score = self._extend_lm(hypothesis, token)
                        self._add(
                            candidates,
                            Hypothesis(
                                tokens=hypothesis.tokens + (token,),
                                score=hypothesis.score + value,
                                lm_state=lm_state,
                                word=word,
                                lm_score=lm_score,
                            ),
                        )
                active = self._prune(candidates)
                if not active:
                    break
            beam = self._prune(completed)
            cache = {h.tokens: cache[h.tokens] for h in beam}
        return beam

    def decode(self, features: torch.Tensor) -> str:
        beam = self._search(features)
        best = max(beam, key=lambda h: h.score + self._finish_lm(h))
        return self.text_encoder.normalize_text(
            self.text_encoder.decode(best.tokens)
        ).strip()
