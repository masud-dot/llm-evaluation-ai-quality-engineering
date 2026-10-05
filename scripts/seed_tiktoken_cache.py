"""Pre-seed tiktoken's cache so Inspect AI's mock model runs
offline in CI. Run once, with network, in the adapters job.

tiktoken looks up TIKTOKEN_CACHE_DIR/<sha1 of the URL> and
checks the file's SHA-256; both are reproduced here.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

URL = ("https://openaipublic.blob.core.windows.net/encodings/"
       "o200k_base.tiktoken")
SHA256 = ("446a9538cb6c348e3516120d7c08b09f"
          "57c36495e2acfffe59a5bf8b0cfb1a2d")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        data: bytes = r.read()
        return data


def seed(cache_dir: Path, url: str = URL,
         expected: str = SHA256,
         get: Callable[[str], bytes] = fetch) -> Path:
    data = get(url)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("downloaded encoding failed hash check")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / hashlib.sha1(url.encode()).hexdigest()
    path.write_bytes(data)
    return path


if __name__ == "__main__":
    print(seed(Path(os.environ["TIKTOKEN_CACHE_DIR"])))
    sys.exit(0)
