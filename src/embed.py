"""Build the vector index from chunks.jsonl. CPU-only, offline, run once.

Embeddings deliberately never touch the GPU. The index is 20 vectors built
once into a .npy file, and at query time only the question is encoded, which
is imperceptible on CPU. That reserves the entire 4GB VRAM budget for the LLM.

Two variants are always built, so the alias ablation is a config switch in
retrieve.py rather than an index rebuild:

    emb_text.npy    chunk['text']   — includes the [產品]/[機型]/[欄位] header
                                      carrying the bilingual aliases
    emb_value.npy   chunk['value']  — the raw spec value alone
"""

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

# Each variant embeds the chunk field of the same name.
VARIANTS = ("text", "value")

# Query/passage prefixes are model-family specific, and getting them wrong
# degrades retrieval silently — no error, just worse ranking. E5 requires
# them, BGE-m3 requires none, the Chinese BGE v1.5 models want an instruction
# on the query only. Keyed by a substring of the model id; an unlisted model
# raises rather than defaulting to "no prefix".
PREFIXES: dict[str, tuple[str, str]] = {
    # model id substring: (query prefix, passage prefix)
    "bge-m3": ("", ""),
    "e5": ("query: ", "passage: "),
    "bge-small-zh": ("為這個句子生成表示以用於檢索相關文章：", ""),
}


def prefixes_for(model_id: str) -> tuple[str, str]:
    for key, pair in PREFIXES.items():
        if key in model_id:
            return pair
    raise KeyError(
        f"no prefix convention registered for {model_id!r} — look up the model card "
        f"and add it to PREFIXES, do not assume the empty prefix"
    )


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=2)
def get_encoder(model_id: str = DEFAULT_MODEL):
    # Imported lazily so that `--help` and the pure-numpy consumers of this
    # module do not pay the multi-second torch import.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device="cpu")


def encode(texts: list[str], model_id: str = DEFAULT_MODEL, *, is_query: bool = False) -> np.ndarray:
    """Return L2-normalised float32 embeddings, one row per text.
    """
    query_prefix, passage_prefix = prefixes_for(model_id)
    prefix = query_prefix if is_query else passage_prefix

    vectors = get_encoder(model_id).encode(
        [prefix + text for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


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


def load_index(variant: str = "text", out_dir: Path = INDEX_DIR) -> tuple[np.ndarray, dict]:
    """Return (embedding matrix, meta), rows aligned to meta['ids'].

    Callers must encode queries with meta['model']. Mixing an index built by
    one model with queries encoded by another produces plausible-looking
    nonsense and no error, so index.py asserts on this.
    """
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    if variant not in meta["variants"]:
        raise KeyError(f"variant {variant!r} not in index; have {meta['variants']}")

    vectors = np.load(out_dir / f"emb_{variant}.npy")
    if vectors.shape[0] != len(meta["ids"]):
        raise ValueError("index and meta are out of sync — rebuild with embed.py")

    return vectors, meta


def probe(query: str, variant: str, top: int, out_dir: Path = INDEX_DIR) -> None:
    """Smoke test: print the nearest chunks to a query.

    Run this before writing any retrieval code. If '顯卡多強' does not put a
    spec.gpu.* chunk first, the problem is the model or the prefix convention,
    not the ranking logic — and debugging it later through the full pipeline
    is far more expensive.
    """
    vectors, meta = load_index(variant, out_dir)
    scores = vectors @ encode([query], meta["model"], is_query=True)[0]

    print(f"query: {query!r}   variant: {variant}   model: {meta['model']}")
    for rank, position in enumerate(np.argsort(-scores)[:top], start=1):
        print(f"  {rank}. {scores[position]:.4f}  {meta['ids'][position]}")


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
