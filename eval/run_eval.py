"""Retrieval-only evaluation: score every Config against the golden set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from embed import get_encoder
from retrieve import Config, Retriever

GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"

BEST = Config()

ABLATIONS = [
    ("dense only, alias", Config(bm25=False)),
    ("bm25 only, alias", Config(dense=False)),
    ("hybrid, no alias", Config(variant="value")),
    ("hybrid + alias", BEST),
    ("  ...no SKU routing", Config(sku_routing=False)),
    ("  ...no overview chunk", Config(include_overview=False)),
]

TYPES = ("single_field", "cross_field", "model_diff")


# Read the golden set.
def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# The highest R@1 this question set can reach.
def r1_ceiling(questions: list[dict]) -> float:
    return statistics.mean(1 / len(q["gold"]) for q in questions)


# Retrieval metrics over a set of answerable questions.
def score(retriever: Retriever, questions: list[dict]) -> dict:
    recall1, recall3, hit3, reciprocal = [], [], [], []

    for question in questions:
        gold = set(question["gold"])
        found = [hit.chunk["id"] for hit in retriever.search(question["question"])]

        recall1.append(len(gold & set(found[:1])) / len(gold))
        recall3.append(len(gold & set(found)) / len(gold))
        hit3.append(1.0 if gold & set(found) else 0.0)
        ranks = [i for i, cid in enumerate(found, start=1) if cid in gold]
        reciprocal.append(1 / ranks[0] if ranks else 0.0)

    return {
        "R@1": statistics.mean(recall1),
        "R@3": statistics.mean(recall3),
        "Hit@3": statistics.mean(hit3),
        "MRR": statistics.mean(reciprocal),
        "n": len(questions),
    }


# Mean top-1 fusion score over a set of questions.
def mean_top_score(retriever: Retriever, questions: list[dict]) -> float:
    results = [retriever.search(q["question"]) for q in questions]
    return statistics.mean(hits[0].score for hits in results if hits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    questions = load_golden(args.golden)
    answerable = [q for q in questions if q["gold"]]
    unanswerable = [q for q in questions if not q["gold"]]
    alias_subset = [q for q in answerable if q["relies_on_alias"]]

    print(
        f"{len(questions)} questions: {len(answerable)} answerable, "
        f"{len(unanswerable)} unanswerable (excluded from retrieval metrics), "
        f"{len(alias_subset)} alias-dependent\n"
    )

    get_encoder(Retriever(BEST).dense.meta["model"])

    header = f"{'configuration':<24}{'R@1':>7}{'R@3':>7}{'Hit@3':>7}{'MRR':>7}   {'alias R@1':>9}"
    print(header)
    print("-" * len(header))

    best = None
    for label, config in ABLATIONS:
        retriever = Retriever(config)
        if config is BEST:
            best = retriever
        overall = score(retriever, answerable)
        alias = score(retriever, alias_subset)
        print(
            f"{label:<24}{overall['R@1']:>7.3f}{overall['R@3']:>7.3f}"
            f"{overall['Hit@3']:>7.3f}{overall['MRR']:>7.3f}   {alias['R@1']:>9.3f}"
        )

    print(
        f"{'(R@1 ceiling)':<24}{r1_ceiling(answerable):>7.3f}"
        f"{'-':>7}{'-':>7}{'-':>7}   {r1_ceiling(alias_subset):>9.3f}"
    )

    print("\nby question type — hybrid + alias        R@1 (ceiling)     R@3     MRR")
    for kind in TYPES:
        subset = [q for q in answerable if q["type"] == kind]
        result = score(best, subset)
        print(
            f"  {kind:<16} n={result['n']:<3} "
            f"{result['R@1']:>10.3f} ({r1_ceiling(subset):.3f}) "
            f"{result['R@3']:>7.3f} {result['MRR']:>7.3f}"
        )

    print(
        "\nabstain signal (mean top-1 fusion score): "
        f"answerable {mean_top_score(best, answerable):.4f}  vs  "
        f"unanswerable {mean_top_score(best, unanswerable):.4f}"
    )


if __name__ == "__main__":
    main()
