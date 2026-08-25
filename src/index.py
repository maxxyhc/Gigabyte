from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from embed import CHUNKS_PATH, INDEX_DIR, encode, load_chunks, load_index

CJK = "一-鿿㐀-䶿"

TOKEN_RE = re.compile(rf"[a-z0-9]+(?:\.[a-z0-9]+)*|[{CJK}]+")

IDF_FORMULAS = {
    "lucene": lambda n, df: math.log(1 + (n - df + 0.5) / (df + 0.5)),
    "robertson": lambda n, df: math.log((n - df + 0.5) / (df + 0.5)),
}


# Lowercased Latin words plus CJK character bigrams.
def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for match in TOKEN_RE.findall(text.lower().replace("×", "x")):
        if not re.fullmatch(rf"[{CJK}]+", match):
            tokens.append(match)
        elif len(match) == 1:
            tokens.append(match)
        else:
            tokens.extend(match[i : i + 2] for i in range(len(match) - 1))

    return tokens


# Okapi BM25 over the chunk corpus. Pure Python, no dependencies.
class BM25Index:
    # Precompute term frequencies, document lengths and IDF for the corpus.
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

    # Score every chunk against the query, best first.
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


# Cosine similarity over the pre-built embedding matrix.
class DenseIndex:
    # Load the embedding matrix and its meta for one index variant.
    def __init__(self, variant: str = "text", index_dir: Path = INDEX_DIR) -> None:
        self.vectors, self.meta = load_index(variant, index_dir)

    # Score every chunk by cosine similarity to the query, best first.
    def rank(self, query: str) -> list[tuple[int, float]]:
        query_vector = encode([query], self.meta["model"], is_query=True)[0]
        return _sorted(self.vectors @ query_vector)


# Return (position, score) pairs ordered best first.
def _sorted(scores: np.ndarray) -> list[tuple[int, float]]:
    return [(int(i), float(scores[i])) for i in np.argsort(-scores)]


# Load the chunks and both retrievers, aligned on position.
def build(
    variant: str = "text",
    chunks_path: Path = CHUNKS_PATH,
    index_dir: Path = INDEX_DIR,
    **bm25_kwargs,
) -> tuple[list[dict], DenseIndex, BM25Index]:
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
