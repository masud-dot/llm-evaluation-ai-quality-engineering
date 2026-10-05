import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
from seed_tiktoken_cache import seed  # noqa: E402


def test_seed_writes_where_tiktoken_looks(tmp_path: Path) -> None:
    data = b"fake encoding"
    want = hashlib.sha256(data).hexdigest()
    p = seed(tmp_path, url="https://x/enc", expected=want,
             get=lambda u: data)
    assert p.name == hashlib.sha1(b"https://x/enc").hexdigest()
    with pytest.raises(ValueError):
        seed(tmp_path, url="https://x/enc", expected="0" * 64,
             get=lambda u: data)
