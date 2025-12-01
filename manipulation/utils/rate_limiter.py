"""Rate limiter utility for controlling simulation frequency."""

from loop_rate_limiters import RateLimiter as LoopRateLimiter


class RateLimiter:
    """Wrapper around loop_rate_limiters.RateLimiter."""

    def __init__(self, frequency: float = 200.0, warn: bool = False):
        self.rate_limiter = LoopRateLimiter(frequency=frequency, warn=warn)
        self.dt = 1.0 / frequency

    def sleep(self):
        self.rate_limiter.sleep()
