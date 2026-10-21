"""Benchmark helper: the GPU is shared with the window server and other apps, which makes wall-clock timings swing by up to
1.7x.  `Probe` times a fixed GEMM chain; a sample is trusted only if the probe right before and right after it ran at
(nearly) the best speed seen so far."""
import time
import mlx.core as mx


class Probe:
    def __init__(self, tol=1.10):
        self.x = (mx.random.normal((2048, 1024)) * 0.5).astype(mx.bfloat16)
        self.ws = [(mx.random.normal((8224, 1024)) * 0.05).astype(mx.bfloat16) for _ in range(4)]
        mx.eval(self.x, *self.ws)
        self.tol = tol
        self.ref = min(self.once() for _ in range(15))

    def once(self):
        t = time.perf_counter(); mx.eval([self.x @ w.T for w in self.ws]); return time.perf_counter() - t

    def quiet(self):
        t = self.once()
        self.ref = min(self.ref, t)
        return t <= self.ref * self.tol

    def sample(self, fn, tries=40):
        """Run fn() (returns seconds) until it ran between two quiet probes; returns (seconds, accepted)."""
        best = None
        for _ in range(tries):
            q0 = self.quiet(); dt = fn(); q1 = self.quiet()
            best = dt if best is None else min(best, dt)
            if q0 and q1:
                return dt, True
        return best, False
