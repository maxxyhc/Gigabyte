from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

SPEC_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "spec_zh.html"

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

SENTINEL = "desktop-spec-content"

HTTP2 = True

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


# Return the page HTML, retrying on transport errors and bot rejections.
def fetch_html(url: str = SPEC_URL, timeout: float = 20.0) -> str:
    last_error: Exception | None = None

    with httpx.Client(
        http2=HTTP2, headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.get(url)
                response.raise_for_status()

                html = response.text
                if SENTINEL not in html:
                    raise ValueError(
                        f"response did not contain {SENTINEL!r} "
                        f"({len(html)} chars) — likely an anti-bot page"
                    )
                return html

            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                print(f"attempt {attempt}/{MAX_ATTEMPTS} failed: {error}", file=sys.stderr)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"failed to fetch {url}") from last_error


# Command line entry point: fetch the spec page unless a snapshot exists.
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
