"""Fuse the two retrievers into one ranked result list.

Every knob lives on Config, so the ablation table in run_eval.py is a loop
over Config objects rather than five edited copies of this file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import index


@dataclass(frozen=True)
class Config:
    dense: bool = True
    bm25: bool = True
    variant: str = "text"  # "text" carries the alias header, "value" does not
    k: int = 3
    rrf_k: int = 10
    sku_routing: bool = True
    include_overview: bool = True

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


class Hit(NamedTuple):
    chunk: dict
    score: float


def rrf(rankings: list[list[int]], rrf_k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion: sum 1/(rrf_k + rank) across retrievers.

    Fusing on ranks rather than scores because cosine sits in [0, 1] while
    BM25 is unbounded and corpus-dependent — any weighted sum of the two
    would need a normalisation that is unstable over 20 documents.

    rrf_k is 60 in the original paper, which was tuned on rankings thousands
    of documents deep. Over 20 chunks that constant flattens rank 1 and rank
    10 to nearly the same weight, so it is a parameter here, not a constant.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            scores[position] = scores.get(position, 0.0) + 1 / (rrf_k + rank)
    return scores


class Retriever:
    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.chunks, self.dense, self.bm25 = index.build(config.variant)
        self.model_codes = sorted(
            {code for chunk in self.chunks for code in chunk["model_codes"]}
        )

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

    def _select(self, query: str, ordered: list[int]) -> list[int]:
        """Route between a field's per-SKU chunks and its comparison chunk.

        The two are mutually exclusive, never competing. Naming a SKU asks
        about that machine, so the other SKUs and the comparison are dropped.
        Naming none asks about the product, so the comparison chunk answers in
        one slot — the per-SKU chunks score within 0.004 of each other, and
        letting them compete both fills k=3 with near-duplicates and reports an
        arbitrary winner as though it were the answer.
        """
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


def _voted(ranking: list[tuple[int, float]]) -> list[int]:
    """Positions a retriever actually matched, best first.

    Zero-scored documents are dropped rather than ranked. BM25 scores every
    chunk 0 for a query like 「插頭多大顆」, whose terms are absent from the
    corpus; ranking those ties would let an arbitrary tie-break order outvote
    a dense retriever that did find the answer.
    """
    return [position for position, score in ranking if score > 0]


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
