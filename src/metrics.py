"""Latency measurement contract for the benchmark.

Both numbers are defined here rather than at each call site so the README and
the evaluation cannot drift apart on what they mean.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stats:
    """One generation's timing.

    ttft_s   seconds from sending the request to the first *content* token.
             The clock starts before the request leaves, so prompt processing
             is inside this number — that is what a user waits through.
    total_s  seconds from sending the request to the final token.
    """

    ttft_s: float
    total_s: float
    completion_tokens: int
    prompt_tokens: int | None = None

    @property
    def decode_s(self) -> float:
        return self.total_s - self.ttft_s

    @property
    def tps(self) -> float:
        """Decode rate, tokens per second.

        Prefill is excluded — it is already reported as TTFT, and folding it in
        would let a long prompt masquerade as slow generation. The first token
        is excluded with it, so this is (n - 1) tokens over the decode window.
        """
        if self.completion_tokens < 2 or self.decode_s <= 0:
            return 0.0
        return (self.completion_tokens - 1) / self.decode_s
