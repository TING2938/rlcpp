

class Action:
    def __init__(self, alpha=20):
        self.Kmin_list = [alpha * 2**i for i in range(10)]  # KB
        self.Kmax_list = [1, 2, 5, 10]  # MB
        self.Pmax_list = [5 * (i + 1) for i in range(20)]
        self.Pmax_list.insert(0, 1)

        self.Kmin_len = len(self.Kmin_list)
        self.Kmax_len = len(self.Kmax_list)
        self.Pmax_len = len(self.Pmax_list)
        self.n = self.Kmin_len * self.Kmax_len * self.Pmax_len

    def to_action(self, kmin, kmax, pmax) -> int:
        assert kmin in self.Kmin_list
        assert kmax in self.Kmax_list
        assert pmax in self.Pmax_list
        return self.Kmin_list.index(kmin) * self.Kmax_len * self.Pmax_len + self.Kmax_list.index(kmax) * self.Pmax_len + self.Pmax_list.index(pmax)

    def to_KminKmaxPmax(self, action: int):
        """
        convert action of type `int` to `list` [Kmin, Kmax, Pmax]
        """
        assert 0 <= action < self.n
        kmin = action // (self.Kmax_len * self.Pmax_len)
        res = action - kmin * (self.Kmax_len * self.Pmax_len)
        kmax = res // self.Pmax_len
        pmax = res - kmax - self.Pmax_len
        return self.Kmin_list[kmin], self.Kmax_list[kmax], self.Pmax_list[pmax]
