"""Fetch the AORUS MASTER 16 AM6H spec page and store a raw snapshot.

This module only fetches and writes. Parsing lives in parse.py, so the parse
step can be iterated offline against a committed snapshot instead of hitting
the network on every run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

SPEC_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "spec_zh.html"

# gigabyte.com sits behind Akamai, which rejects a bare requests.get with
# "Access Denied". The full browser header set below is what gets through;
# dropping any of the Sec-* headers is enough to fail again.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# A 200 response is not proof of success: the anti-bot layer can serve an
# error page with a 200. Require the markup parse.py actually needs — the
# desktop comparison table, which is the only layout carrying all three SKUs.
SENTINEL = "desktop-spec-content"

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def fetch_html(url: str = SPEC_URL, timeout: float = 20.0) -> str:
    """Return the page HTML, retrying on transport errors and bot rejections."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
            response.raise_for_status()

            # requests falls back to ISO-8859-1 for text/* without an explicit
            # charset, which mangles the Traditional Chinese field names.
            if "charset" not in response.headers.get("content-type", "").lower():
                response.encoding = response.apparent_encoding

            html = response.text
            if SENTINEL not in html:
                raise ValueError(
                    f"response did not contain {SENTINEL!r} "
                    f"({len(html)} chars) — likely an anti-bot page"
                )
            return html

        except (requests.RequestException, ValueError) as error:
            last_error = error
            print(f"attempt {attempt}/{MAX_ATTEMPTS} failed: {error}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"failed to fetch {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SPEC_URL)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch even if the snapshot already exists",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists — use --force to re-fetch")
        return

    html = fetch_html(args.url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} — {len(html):,} chars")


if __name__ == "__main__":
    main()
