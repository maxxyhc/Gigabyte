from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"

DEFAULT_MODEL = "BAAI/bge-m3"

VARIANTS = ("text", "value")

PREFIXES: dict[str, tuple[str, str]] = {
    "bge-m3": ("", ""),
    "e5": ("query: ", "passage: "),
    "bge-small-zh": ("為這個句子生成表示以用於檢索相關文章：", ""),
}


# Return the (query, passage) prefix pair the given model expects.
def prefixes_for(model_id: str) -> tuple[str, str]:
    for key, pair in PREFIXES.items():
        if key in model_id:
            return pair
    raise KeyError(
        f"no prefix convention registered for {model_id!r} — look up the model card "
        f"and add it to PREFIXES, do not assume the empty prefix"
    )


# Read chunks.jsonl into a list of dicts.
def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# Load the sentence-transformers encoder, cached across calls.
@lru_cache(maxsize=2)
def get_encoder(model_id: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device="cpu")


# Return L2-normalised float32 embeddings, one row per text.
def encode(texts: list[str], model_id: str = DEFAULT_MODEL, *, is_query: bool = False) -> np.ndarray:
    query_prefix, passage_prefix = prefixes_for(model_id)
    prefix = query_prefix if is_query else passage_prefix

    vectors = get_encoder(model_id).encode(
        [prefix + text for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


# Encode every chunk into both index variants and write them with a meta file.
def build(chunks: list[dict], model_id: str = DEFAULT_MODEL, out_dir: Path = INDEX_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        vectors = encode([chunk[variant] for chunk in chunks], model_id)
        np.save(out_dir / f"emb_{variant}.npy", vectors)
        print(f"  emb_{variant}.npy  {vectors.shape}")

    meta = {
        "model": model_id,
        "dim": int(vectors.shape[1]),
        "count": len(chunks),
        "ids": [chunk["id"] for chunk in chunks],
        "variants": list(VARIANTS),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


# Return (embedding matrix, meta), rows aligned to meta['ids'].
def load_index(variant: str = "text", out_dir: Path = INDEX_DIR) -> tuple[np.ndarray, dict]:
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    if variant not in meta["variants"]:
        raise KeyError(f"variant {variant!r} not in index; have {meta['variants']}")

    vectors = np.load(out_dir / f"emb_{variant}.npy")
    if vectors.shape[0] != len(meta["ids"]):
        raise ValueError("index and meta are out of sync — rebuild with embed.py")

    return vectors, meta


# Smoke test: print the nearest chunks to a query.
def probe(query: str, variant: str, top: int, out_dir: Path = INDEX_DIR) -> None:
    vectors, meta = load_index(variant, out_dir)
    scores = vectors @ encode([query], meta["model"], is_query=True)[0]

    print(f"query: {query!r}   variant: {variant}   model: {meta['model']}")
    for rank, position in enumerate(np.argsort(-scores)[:top], start=1):
        print(f"  {rank}. {scores[position]:.4f}  {meta['ids'][position]}")


# Command line entry point: build the index, or probe it with --probe.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--probe", help="skip building; print nearest chunks to this query")
    parser.add_argument("--variant", default="text", choices=VARIANTS)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if args.probe:
        probe(args.probe, args.variant, args.top, args.out)
        return

    chunks = load_chunks(args.chunks)
    print(f"encoding {len(chunks)} chunks with {args.model} on CPU")
    meta = build(chunks, args.model, args.out)
    print(f"wrote {args.out}/meta.json — dim {meta['dim']}, {meta['count']} chunks")


if __name__ == "__main__":
    main()
