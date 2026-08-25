from __future__ import annotations

import argparse

from llm import BASE_URL, LlamaClient, Stream
from rag_prompt import build_messages
from retrieve import Config, Hit, Retriever


# The retrieve-then-generate pipeline.
class Rag:
    # Build the retriever and bind a llama-server client.
    def __init__(self, config: Config = Config(), base_url: str = BASE_URL) -> None:
        self.retriever = Retriever(config)
        self.client = LlamaClient(base_url)

    # Retrieve, then hand back the hits and an unconsumed answer stream.
    def answer(self, question: str, *, max_tokens: int = 512) -> tuple[list[Hit], Stream]:
        hits = self.retriever.search(question)
        stream = self.client.stream(build_messages(question, hits), max_tokens=max_tokens)
        return hits, stream


# Answer one question, streaming it to stdout with its timing.
def run(rag: Rag, question: str, *, show_sources: bool, max_tokens: int) -> None:
    hits, stream = rag.answer(question, max_tokens=max_tokens)
    for text in stream:
        print(text, end="", flush=True)

    stats = stream.stats
    print(
        f"\n\nTTFT {stats.ttft_s * 1000:.0f} ms | {stats.tps:.1f} tok/s | "
        f"{stats.completion_tokens} tokens | prompt {stats.prompt_tokens}"
    )
    if show_sources:
        print("sources: " + ", ".join(hit.chunk["id"] for hit in hits))


# Command line entry point: one question, or an interactive loop.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sources", action="store_true")
    args = parser.parse_args()

    rag = Rag(Config(k=args.k), args.base_url)
    if not rag.client.is_up():
        raise SystemExit(f"llama-server is not answering at {args.base_url}")

    if args.question:
        run(rag, args.question, show_sources=args.sources, max_tokens=args.max_tokens)
        return

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if question:
            run(rag, question, show_sources=args.sources, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
