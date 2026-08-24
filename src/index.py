"""The two retrievers: dense vectors and a hand-written BM25.

Both expose the same `rank(query)`, returning *every* chunk as
(position, score) sorted best-first. Full rankings rather than a top-n
because retrieve.py fuses with RRF, which needs each retriever's rank for a
document. With 20 chunks the full sort is free.

Positions index into the chunk list, so both retrievers must be built from
the same chunks.jsonl in the same order — build() asserts that.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from embed import CHUNKS_PATH, INDEX_DIR, encode, load_chunks, load_index

CJK = "一-鿿㐀-䶿"

# Latin runs keep internal dots so version-like strings survive whole
# ("usb3.2", "802.11be"); CJK runs are captured whole and split into bigrams.
TOKEN_RE = re.compile(rf"[a-z0-9]+(?:\.[a-z0-9]+)*|[{CJK}]+")

# Robertson's original IDF goes negative once a term appears in more than about
# half the documents, so a document containing it is *penalised*. On 20 docs
# indexed by character bigrams that is not an edge case: 12 terms go negative
# here, including am6h, byh, 機型 and 產品. Lucene's variant adds 1 inside the
# log and stays positive. The broken formula is kept so the README can show it.
IDF_FORMULAS = {
    "lucene": lambda n, df: math.log(1 + (n - df + 0.5) / (df + 0.5)),
    "robertson": lambda n, df: math.log((n - df + 0.5) / (df + 0.5)),
}


def tokenize(text: str) -> list[str]:
    """Lowercased Latin words plus CJK character bigrams.

    Bigrams instead of a word segmenter (jieba): one dependency fewer, and a
    segmenter mangles the model strings this corpus is full of — AM6H, WQXGA,
    Gen4x4 — which are exactly the terms a spec question hinges on.
    """
    tokens: list[str] = []

    for match in TOKEN_RE.findall(text.lower().replace("×", "x")):
        if not re.fullmatch(rf"[{CJK}]+", match):
            tokens.append(match)
        elif len(match) == 1:
            tokens.append(match)
        else:
            tokens.extend(match[i : i + 2] for i in range(len(match) - 1))

    return tokens


class BM25Index:
    """Okapi BM25 over the chunk corpus. Pure Python, no dependencies."""

    def __init__(
        self,
        docs: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_formula: str = "lucene",
    ) -> None:
        self.k1 = k1
        self.b = b

        doc_tokens = [tokenize(doc) for doc in docs]
        self.doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean())
        self.term_freqs = [Counter(tokens) for tokens in doc_tokens]

        df = Counter(term for tokens in doc_tokens for term in set(tokens))
        formula = IDF_FORMULAS[idf_formula]
        self.idf = {term: formula(len(docs), count) for term, count in df.items()}

    def rank(self, query: str) -> list[tuple[int, float]]:
        scores = np.zeros(len(self.term_freqs), dtype=np.float32)

        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for position, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[position] / self.avgdl
                scores[position] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        return _sorted(scores)


class DenseIndex:
    """Cosine similarity over the pre-built embedding matrix.

    Twenty 1024-dim vectors: `matrix @ query` is the entire search.
    """

    def __init__(self, variant: str = "text", index_dir: Path = INDEX_DIR) -> None:
        self.vectors, self.meta = load_index(variant, index_dir)

    def rank(self, query: str) -> list[tuple[int, float]]:
        # Encoded with meta['model'], never a default: an index built by one
        # model and queried by another returns plausible nonsense silently.
        query_vector = encode([query], self.meta["model"], is_query=True)[0]
        return _sorted(self.vectors @ query_vector)


def _sorted(scores: np.ndarray) -> list[tuple[int, float]]:
    return [(int(i), float(scores[i])) for i in np.argsort(-scores)]


def build(
    variant: str = "text",
    chunks_path: Path = CHUNKS_PATH,
    index_dir: Path = INDEX_DIR,
    **bm25_kwargs,
) -> tuple[list[dict], DenseIndex, BM25Index]:
    """Load the chunks and both retrievers, aligned on position."""
    chunks = load_chunks(chunks_path)
    dense = DenseIndex(variant, index_dir)

    if [chunk["id"] for chunk in chunks] != dense.meta["ids"]:
        raise ValueError("chunks.jsonl and the embedding index disagree — rerun embed.py")

    return chunks, dense, BM25Index([chunk[variant] for chunk in chunks], **bm25_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--tokenize", dest="show_tokens")
    parser.add_argument("--variant", default="text", choices=["text", "value"])
    parser.add_argument("--method", default="both", choices=["dense", "bm25", "both"])
    parser.add_argument("--idf-formula", default="lucene", choices=list(IDF_FORMULAS))
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if args.show_tokens:
        print(tokenize(args.show_tokens))
        return

    chunks, dense, bm25 = build(args.variant, idf_formula=args.idf_formula)
    negative = [term for term, value in bm25.idf.items() if value < 0]
    print(
        f"{len(chunks)} chunks, variant={args.variant} | "
        f"bm25: {len(bm25.idf)} terms, avgdl={bm25.avgdl:.1f}, "
        f"idf={args.idf_formula}, {len(negative)} negative {negative[:6]}"
    )

    if not args.query:
        return

    for name, retriever in (("dense", dense), ("bm25", bm25)):
        if args.method in (name, "both"):
            print(f"\n{name}  {args.query!r}")
            for rank, (position, score) in enumerate(retriever.rank(args.query)[: args.top], 1):
                print(f"  {rank}. {score:8.4f}  {chunks[position]['id']}")


if __name__ == "__main__":
    main()
