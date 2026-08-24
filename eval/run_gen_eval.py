"""Generation evaluation: run the golden set through the full pipeline.

Retrieval is already scored by run_eval.py. This scores what the model does
with what it retrieved — whether the facts survive into the answer, whether
unanswerable questions are declined, and whether anything was invented.

The no-RAG row is the point of the table: it shows what the bare 4B model says
about a laptop that postdates its training data.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm import BASE_URL, LlamaClient  # noqa: E402
from rag_prompt import build_messages, build_plain_messages  # noqa: E402
from retrieve import Config, Retriever  # noqa: E402

GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"
OUT_PATH = ROOT / "eval" / "results" / "gen_eval.json"

# Matching contract from AGENTS.md: compare after casefolding and removing
# whitespace, hyphens and the multiplication sign, so "USB 3.2 Gen 2" matches
# the page's "USB3.2 Gen2" and a correct answer is not failed on spacing.
def normalise(text: str) -> str:
    return "".join(text.split()).replace("×", "x").replace("-", "").casefold()


# A refusal has no fixed wording, so this is a heuristic over the phrasings the
# prompt actually produces plus common variants. It is reported as a rate, not
# used as a pass/fail gate on any single answer.
REFUSAL_MARKERS = [
    "未提供", "沒有提供", "未列出", "沒有列出", "未提及", "無法確認", "無法判斷",
    "does not provide", "not provide", "no information", "not specified",
    "does not state", "not available", "cannot determine",
]

# Simplified-only forms that show up in this domain. A targeted probe, not a
# general converter: it answers whether simplified leakage is frequent enough
# to justify adding a conversion dependency.
SIMPLIFIED_CHARS = set("显内网线频处规键认证储装备简传输电视无独声测题设计运转换级别")


def satisfied(answer: str, group: list[str]) -> bool:
    normalised = normalise(answer)
    return any(normalise(form) in normalised for form in group)


def grade(question: dict, answer: str) -> dict:
    groups = question["answer_contains"]
    hits = [satisfied(answer, group) for group in groups]
    forbidden = question.get("must_not_contain", [])

    return {
        "facts_all": bool(groups) and all(hits),
        "facts_partial": statistics.mean(hits) if hits else None,
        "refused": any(m in answer.lower() or m in answer for m in REFUSAL_MARKERS),
        "fabricated": any(satisfied(answer, group) for group in forbidden),
        "simplified": sorted(SIMPLIFIED_CHARS.intersection(answer)),
    }


def collect(text_stream) -> str:
    return "".join(text_stream)


def run_config(label: str, questions: list[dict], retriever: Retriever | None, client: LlamaClient) -> list[dict]:
    records = []
    for number, question in enumerate(questions, start=1):
        if retriever is None:
            messages = build_plain_messages(question["question"])
            sources = []
        else:
            hits = retriever.search(question["question"])
            messages = build_messages(question["question"], hits)
            sources = [hit.chunk["id"] for hit in hits]

        # Greedy decoding, stated explicitly rather than inherited from the
        # client default: a benchmark whose numbers move between runs cannot
        # tell a real change from sampling noise.
        stream = client.stream(messages, temperature=0.0, max_tokens=320)
        answer = collect(stream)
        record = {
            "config": label,
            "id": question["id"],
            "type": question["type"],
            "answer": answer,
            "sources": sources,
            "ttft_s": stream.stats.ttft_s,
            "tps": stream.stats.tps,
            "prompt_tokens": stream.stats.prompt_tokens,
            **grade(question, answer),
        }
        records.append(record)
        print(f"  [{label}] {number}/{len(questions)} {question['id']}", end="\r", flush=True)

    print(" " * 60, end="\r")
    return records


def summarise(records: list[dict], questions: dict[str, dict]) -> dict:
    answerable = [r for r in records if questions[r["id"]]["gold"]]
    unanswerable = [r for r in records if not questions[r["id"]]["gold"]]
    ttft = sorted(r["ttft_s"] for r in records)

    return {
        "facts_all": statistics.mean(r["facts_all"] for r in answerable),
        "facts_partial": statistics.mean(r["facts_partial"] for r in answerable),
        "refuse_ok": statistics.mean(r["refused"] for r in unanswerable),
        "refuse_wrong": statistics.mean(r["refused"] for r in answerable),
        "fabricated": sum(r["fabricated"] for r in records),
        "simplified": sum(1 for r in records if r["simplified"]),
        "ttft_p50": ttft[len(ttft) // 2],
        "ttft_p95": ttft[int(len(ttft) * 0.95)],
        "tps": statistics.mean(r["tps"] for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    with args.golden.open(encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle if line.strip()]
    by_id = {q["id"]: q for q in questions}

    client = LlamaClient(args.base_url)
    if not client.is_up():
        raise SystemExit(f"llama-server is not answering at {args.base_url}")

    configs = [
        ("no RAG", None),
        ("RAG k=1", Retriever(Config(k=1))),
        ("RAG k=3", Retriever(Config(k=3))),
    ]

    all_records = []
    rows = {}
    for label, retriever in configs:
        records = run_config(label, questions, retriever, client)
        all_records.extend(records)
        rows[label] = summarise(records, by_id)

    header = (
        f"{'config':<10}{'facts':>7}{'part':>7}{'refuse':>8}{'falseR':>8}"
        f"{'fab':>5}{'簡':>4}{'TTFT50':>8}{'TTFT95':>8}{'TPS':>7}"
    )
    print(header)
    print("-" * len(header))
    for label, row in rows.items():
        print(
            f"{label:<10}{row['facts_all']:>7.3f}{row['facts_partial']:>7.3f}"
            f"{row['refuse_ok']:>8.3f}{row['refuse_wrong']:>8.3f}"
            f"{row['fabricated']:>5}{row['simplified']:>4}"
            f"{row['ttft_p50']:>7.2f}s{row['ttft_p95']:>7.2f}s{row['tps']:>7.1f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": rows, "records": all_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
