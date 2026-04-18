"""N-of-M smoother for anomaly confirmation."""
from __future__ import annotations

from collections import deque
from typing import Literal

EdgeEvent = Literal["none", "rising", "falling"]


class NofMSmoother:
    """Confirms an alarm when K of the last M decisions are True.

    Reports edge transitions via push(): "rising" when first confirmed,
    "falling" when de-confirmed, "none" otherwise.
    """

    def __init__(self, k: int, m: int) -> None:
        if not (1 <= k <= m):
            raise ValueError("require 1 <= k <= m")
        self._k = k
        self._buf: deque[bool] = deque(maxlen=m)
        self._confirmed = False

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def push(self, exceeded: bool) -> EdgeEvent:
        self._buf.append(bool(exceeded))
        new = sum(self._buf) >= self._k
        if new and not self._confirmed:
            self._confirmed = True
            return "rising"
        if self._confirmed and not new:
            self._confirmed = False
            return "falling"
        return "none"
