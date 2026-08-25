from __future__ import annotations

from dataclasses import dataclass


# One generation's timing.
@dataclass(frozen=True)
class Stats:
    ttft_s: float
    total_s: float
    completion_tokens: int
    prompt_tokens: int | None = None

    # Seconds spent generating after the first token arrived.
    @property
    def decode_s(self) -> float:
        return self.total_s - self.ttft_s

    # Decode rate, tokens per second.
    @property
    def tps(self) -> float:
        if self.completion_tokens < 2 or self.decode_s <= 0:
            return 0.0
        return (self.completion_tokens - 1) / self.decode_s
