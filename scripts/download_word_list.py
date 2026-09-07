from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a newline-delimited word list from a user-supplied URL. "
            "Only use sources whose license allows your intended use."
        )
    )
    parser.add_argument("--url", required=True, help="Direct URL to a plain-text word list.")
    parser.add_argument(
        "--output",
        default="data/raw/english_words.txt",
        help="Destination under data/raw/ by convention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s", args.url)
    with urllib.request.urlopen(args.url, timeout=60) as response:
        content = response.read()
    output.write_bytes(content)
    logging.info("Wrote %s bytes to %s", len(content), output)


if __name__ == "__main__":
    main()
