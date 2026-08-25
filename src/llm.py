from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Iterator

import requests

from metrics import Stats

BASE_URL = "http://127.0.0.1:8080"


# Iterate the generated text; `stats` is filled once iteration ends.
class Stream:
    # Wrap a streaming response together with the time the request left.
    def __init__(self, response: requests.Response, started_at: float) -> None:
        self._response = response
        self._started_at = started_at
        self.stats: Stats | None = None

    # Yield each content delta, then fill `stats`.
    def __iter__(self) -> Iterator[str]:
        ttft = None
        deltas = 0
        usage: dict = {}

        for raw in self._response.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                break

            event = json.loads(payload)
            usage = event.get("usage") or usage

            choices = event.get("choices") or []
            content = choices[0].get("delta", {}).get("content") if choices else None
            if not content:
                continue

            if ttft is None:
                ttft = perf_counter() - self._started_at
            deltas += 1
            yield content

        elapsed = perf_counter() - self._started_at
        self.stats = Stats(
            ttft_s=ttft if ttft is not None else elapsed,
            total_s=elapsed,
            completion_tokens=usage.get("completion_tokens", deltas),
            prompt_tokens=usage.get("prompt_tokens"),
        )


# HTTP client for a running llama-server.
class LlamaClient:
    # Bind the client to a llama-server base URL.
    def __init__(self, base_url: str = BASE_URL, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # Return whether the server answers its health endpoint.
    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).ok
        except requests.RequestException:
            return False

    # Send a chat completion request and return its unconsumed stream.
    def stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Stream:
        body = {
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started_at = perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return Stream(response, started_at)


# Command line entry point: stream one prompt and report its timing.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = LlamaClient(args.base_url)
    if not client.is_up():
        raise SystemExit(f"llama-server is not answering at {args.base_url}")

    stream = client.stream(
        [{"role": "user", "content": args.prompt}], max_tokens=args.max_tokens
    )
    for text in stream:
        print(text, end="", flush=True)

    stats = stream.stats
    print(
        f"\n\nTTFT {stats.ttft_s * 1000:.0f} ms | {stats.tps:.1f} tok/s | "
        f"{stats.completion_tokens} tokens in {stats.total_s:.2f} s"
    )


if __name__ == "__main__":
    main()
