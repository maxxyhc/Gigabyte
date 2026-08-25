from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "raw" / "spec_zh.html"
ALIAS_PATH = ROOT / "data" / "aliases.json"
OUT_PATH = ROOT / "data" / "chunks.jsonl"

PRODUCT = "GIGABYTE AORUS MASTER 16 AM6H"
SOURCE_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"

EXPECTED_FIELDS = 17
TRADEMARK_CHARS = "®™©"


FOOTNOTE_SELECTOR = "p[style*='font-size:80%']"

BLOCK_TAGS = ("p", "div")


SPEC_ROOT = "div.desktop-spec-content"
FIELD_SELECTOR = "div.spec-column div.multiple-title"
SLIDE_SELECTOR = "div.swiper-wrapper > div.swiper-slide"
ROW_SELECTOR = "div.spec-item-list[data-spec-row]"
MODEL_NAME_SELECTOR = ".model-base-info-subtitle"


# Read the field alias table.
def load_aliases(path: Path = ALIAS_PATH) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# 'AORUS MASTER 16 BZH' -> 'BZH'.
def model_code(name: str) -> str:
    return name.split()[-1]


# Model names in the same order as the comparison table's columns.
def extract_models(soup: BeautifulSoup) -> list[str]:
    subtitle = soup.select_one(MODEL_NAME_SELECTOR)
    if subtitle is None:
        return []
    return [part.strip() for part in subtitle.get_text(strip=True).split("/") if part.strip()]


# Return (model names, ordered [(field, {model: value_node})]).
def extract(
    html: str, keep_footnotes: bool = True
) -> tuple[list[str], list[tuple[str, dict[str, Tag]]]]:
    soup = BeautifulSoup(html, "html.parser")

    root = soup.select_one(SPEC_ROOT)
    if root is None:
        raise ValueError(f"{SPEC_ROOT} not found — the page layout has changed")

    fields = [node.get_text(strip=True) for node in root.select(FIELD_SELECTOR)]
    slides = root.select(SLIDE_SELECTOR)

    models = extract_models(soup)
    if len(models) != len(slides):
        raise ValueError(
            f"found {len(slides)} spec columns but {len(models)} model names "
            f"{models} — cannot map values to models"
        )

    by_row: dict[int, dict[str, Tag]] = {}
    for model, slide in zip(models, slides):
        for row in slide.select(ROW_SELECTOR):
            index = int(row["data-spec-row"])
            if index >= len(fields):
                continue
            if not keep_footnotes:
                for footnote in row.select(FOOTNOTE_SELECTOR):
                    footnote.decompose()
            by_row.setdefault(index, {})[model] = row

    return models, [(fields[i], by_row[i]) for i in sorted(by_row)]


# Flatten a value node to clean multi-line plain text.
def normalize(node: Tag) -> str:
    for br in node.find_all("br"):
        br.replace_with("\n")

    for block in node.find_all(BLOCK_TAGS):
        block.insert_before("\n")

    text = node.get_text()
    text = text.translate({ord(c): None for c in TRADEMARK_CHARS})
    text = unicodedata.normalize("NFKC", text)

    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


# Collapse models that share a value, preserving first-seen order.
def merge_identical(per_model: dict[str, str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for model, value in per_model.items():
        groups.setdefault(value, []).append(model)
    return list(groups.items())


# Build one retrieval chunk.
def to_chunk(field: str, value: str, models: list[str], all_models: list[str], meta: dict) -> dict:
    codes = [model_code(m) for m in models]
    shared = set(models) == set(all_models)

    if shared:
        chunk_id = f"spec.{meta['id']}"
        model_line = f"[機型] 全機型 ({' / '.join(model_code(m) for m in all_models)})"
    else:
        chunk_id = f"spec.{meta['id']}." + "-".join(code.lower() for code in codes)
        model_line = f"[機型] {' / '.join(models)}"

    labels = " / ".join([meta["en"], *meta["aliases"]])
    text = "\n".join(
        [
            f"[產品] {PRODUCT}",
            model_line,
            f"[欄位] {field} ({labels})",
            value,
        ]
    )

    return {
        "id": chunk_id,
        "kind": "spec" if shared else "sku",
        "field": field,
        "field_en": meta["en"],
        "models": models,
        "model_codes": codes,
        "shared_across_models": shared,
        "value": value,
        "text": text,
        "source": SOURCE_URL,
    }


# One chunk holding every SKU's value for a field that varies.
def to_compare_chunk(field: str, values: dict[str, str], all_models: list[str], meta: dict) -> dict:
    body = "\n".join(f"[{model_code(m)}]\n{values[m]}" for m in all_models)
    labels = " / ".join([meta["en"], *meta["aliases"]])
    codes = [model_code(m) for m in all_models]

    return {
        "id": f"spec.{meta['id']}.compare",
        "kind": "compare",
        "field": field,
        "field_en": meta["en"],
        "models": all_models,
        "model_codes": codes,
        "shared_across_models": False,
        "value": body,
        "text": "\n".join(
            [
                f"[產品] {PRODUCT}",
                f"[機型] 三機型比較 ({' / '.join(codes)})",
                f"[欄位] {field} ({labels})",
                body,
            ]
        ),
        "source": SOURCE_URL,
    }


# A digest chunk: every field with the head of its value.
def build_overview(merged: list[tuple], all_models: list[str]) -> dict:
    lines: list[str] = []
    for field, meta, _values, groups in merged:
        if len(groups) == 1:
            lines.append(f"{field} ({meta['en']}): {groups[0][0].splitlines()[0]}")
        else:
            variants = " | ".join(
                f"{'/'.join(model_code(m) for m in models)}: {value.splitlines()[0]}"
                for value, models in groups
            )
            lines.append(f"{field} ({meta['en']}): {variants}")

    header = [
        f"[產品] {PRODUCT}",
        f"[機型] 全機型 ({' / '.join(model_code(m) for m in all_models)})",
        "[欄位] 規格總覽 (Spec overview / summary / 全部規格 / 機型比較)",
    ]

    return {
        "id": "derived.overview",
        "kind": "derived",
        "field": "規格總覽",
        "field_en": "Spec overview",
        "models": all_models,
        "model_codes": [model_code(m) for m in all_models],
        "shared_across_models": True,
        "value": "\n".join(lines),
        "text": "\n".join([*header, *lines]),
        "source": SOURCE_URL,
    }


# Turn the raw HTML into the full list of retrieval chunks.
def build_chunks(html: str, aliases: dict[str, dict], keep_footnotes: bool = True) -> list[dict]:
    models, pairs = extract(html, keep_footnotes=keep_footnotes)

    if len(pairs) != EXPECTED_FIELDS:
        print(
            f"warning: extracted {len(pairs)} fields, expected {EXPECTED_FIELDS} "
            f"— the page layout may have changed",
            file=sys.stderr,
        )

    missing = [field for field, _ in pairs if field not in aliases]
    if missing:
        raise KeyError(f"no alias entry for: {missing} — add them to {ALIAS_PATH.name}")

    merged = []
    for field, nodes in pairs:
        values = {model: normalize(node) for model, node in nodes.items()}
        merged.append((field, aliases[field], values, merge_identical(values)))

    chunks = []
    for field, meta, values, groups in merged:
        for value, group_models in groups:
            chunks.append(to_chunk(field, value, group_models, models, meta))
        if len(groups) > 1:
            chunks.append(to_compare_chunk(field, values, models, meta))

    chunks.append(build_overview(merged, models))
    return chunks


# Command line entry point: parse the snapshot into chunks.jsonl.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=IN_PATH)
    parser.add_argument("--out", dest="out_path", type=Path, default=OUT_PATH)
    parser.add_argument("--aliases", type=Path, default=ALIAS_PATH)
    parser.add_argument(
        "--drop-footnotes",
        action="store_true",
        help="strip small-type footnotes from spec values",
    )
    args = parser.parse_args()

    html = args.in_path.read_text(encoding="utf-8")
    chunks = build_chunks(
        html,
        load_aliases(args.aliases),
        keep_footnotes=not args.drop_footnotes,
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    varying = sum(1 for c in chunks if not c["shared_across_models"])
    print(f"wrote {args.out_path} — {len(chunks)} chunks ({varying} SKU-specific)")
    for chunk in chunks:
        scope = "all" if chunk["shared_across_models"] else "/".join(chunk["model_codes"])
        print(f"  {chunk['id']:<26} {chunk['kind']:<8} {chunk['field']:<8} {scope:<12} {len(chunk['value']):>4} chars")


if __name__ == "__main__":
    main()
