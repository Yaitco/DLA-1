import numpy as np


class Beam:
    def __init__(
        self,
    ):
        self.strings = []
        # log probs
        self.blank_probs = []
        self.non_blank_probs = []

    def add(self, s, bp=float("-inf"), nbp=float("-inf")):
        if s in self.strings:
            i = self.strings.index(s)
            self.blank_probs[i] = np.logaddexp(self.blank_probs[i], bp)
            self.non_blank_probs[i] = np.logaddexp(self.non_blank_probs[i], nbp)
        else:
            self.strings.append(s)
            self.blank_probs.append(bp)
            self.non_blank_probs.append(nbp)

    def get(self, i):
        return (self.strings[i], self.blank_probs[i], self.non_blank_probs[i])

    def keep_k(self, k):
        score = np.logaddexp(self.blank_probs, self.non_blank_probs)
        inds = score.argsort()[::-1][:k]
        self.strings = [self.strings[i] for i in inds]
        self.blank_probs = [self.blank_probs[i] for i in inds]
        self.non_blank_probs = [self.non_blank_probs[i] for i in inds]

    def __len__(self):
        return len(self.strings)
