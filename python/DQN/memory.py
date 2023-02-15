import numpy as np


class ReplyBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.idx = 0
        self.buffer = np.zeros(max_size, dtype=object)

    def store(self, exp):
        self.buffer[self.idx % self.max_size] = exp
        self.idx += 1

    def sample(self, batch_size: int):
        indices = np.random.choice(min(self.idx, self.max_size), batch_size)
        return self.buffer[indices]

    def __len__(self):
        return min(self.idx, self.max_size)
