class LoopArea:
    """Streaming estimator of ∮ p_R dV_R with exponential forgetting.
    W_t = λ W_{t-1} + p_{t-1} (V_t - V_{t-1})
    """
    def __init__(self, lam: float = 1.0):
        self.prev_p = None
        self.prev_v = None
        self.W = 0.0
        self.lam = float(lam)

    def update(self, p, v):
        p = float(p); v = float(v)
        if self.prev_p is not None:
            self.W = self.lam * self.W + self.prev_p * (v - self.prev_v)
        self.prev_p, self.prev_v = p, v
        return self.W
