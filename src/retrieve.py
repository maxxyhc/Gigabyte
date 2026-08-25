from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import index


# Every knob the retriever exposes, so ablations are a loop over instances.
@dataclass(frozen=True)
class Config:
    dense: bool = True
    bm25: bool = True
    variant: str = "text"
    k: int = 3
    rrf_k: int = 10
    sku_routing: bool = True
    include_overview: bool = True

    # A short name for this configuration, used in the ablation table.
    @property
    def label(self) -> str:
        parts = [n for n, on in (("dense", self.dense), ("bm25", self.bm25)) if on]
        if self.variant == "text":
            parts.append("alias")
        if not self.sku_routing:
            parts.append("no-sku-routing")
        if not self.include_overview:
            parts.append("no-overview")
        return "+".join(parts)


# One retrieved chunk and its fusion score.
class Hit(NamedTuple):
    chunk: dict
    score: float


# Reciprocal Rank Fusion: sum 1/(rrf_k + rank) across retrievers.
def rrf(rankings: list[list[int]], rrf_k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            scores[position] = scores.get(position, 0.0) + 1 / (rrf_k + rank)
    return scores


# Hybrid retriever: dense and BM25 fused, then filtered by SKU.
class Retriever:
    # Load the chunks and both retrievers for this configuration.
    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.chunks, self.dense, self.bm25 = index.build(config.variant)
        self.model_codes = sorted(
            {code for chunk in self.chunks for code in chunk["model_codes"]}
        )

    # Return the top-k chunks for a question, best first.
    def search(self, query: str) -> list[Hit]:
        rankings = []
        if self.config.dense:
            rankings.append(_voted(self.dense.rank(query)))
        if self.config.bm25:
            rankings.append(_voted(self.bm25.rank(query)))

        scores = rrf(rankings, self.config.rrf_k)
        ordered = sorted(scores, key=lambda p: -scores[p])
        kept = self._select(query, ordered)
        return [Hit(self.chunks[p], scores[p]) for p in kept[: self.config.k]]

    # Route between a field's per-SKU chunks and its comparison chunk.
    def _select(self, query: str, ordered: list[int]) -> list[int]:
        requested = {c for c in self.model_codes if c.lower() in query.lower()}
        kept: list[int] = []

        for position in ordered:
            chunk = self.chunks[position]

            if chunk["kind"] == "derived" and not self.config.include_overview:
                continue

            if self.config.sku_routing:
                if requested and chunk["kind"] == "compare":
                    continue
                if requested and chunk["kind"] == "sku":
                    if not requested.intersection(chunk["model_codes"]):
                        continue
                if not requested and chunk["kind"] == "sku":
                    continue

            kept.append(position)

        return kept


# Positions a retriever actually matched, best first.
def _voted(ranking: list[tuple[int, float]]) -> list[int]:
    return [position for position, score in ranking if score > 0]


# Command line entry point: run one query under the given configuration.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--variant", default="text", choices=["text", "value"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-sku-routing", action="store_true")
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()

    config = Config(
        dense=not args.no_dense,
        bm25=not args.no_bm25,
        variant=args.variant,
        k=args.k,
        rrf_k=args.rrf_k,
        sku_routing=not args.no_sku_routing,
        include_overview=not args.no_overview,
    )

    print(f"[{config.label}]  {args.query!r}")
    for rank, hit in enumerate(Retriever(config).search(args.query), start=1):
        print(f"  {rank}. {hit.score:.4f}  {hit.chunk['id']:<20} {hit.chunk['field']}")


if __name__ == "__main__":
    main()
