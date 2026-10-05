from pathlib import Path

import pytest

from aiqe.providers import (Completion, RecordingProvider,
                            ReplayProvider)


class Fake:
    def complete(self, model: str, prompt: str) -> Completion:
        return Completion(text="ok", model=model,
                          input_tokens=1, output_tokens=1)


def test_record_then_replay(tmp_path: Path) -> None:
    rec = RecordingProvider(Fake(), tmp_path)
    a = rec.complete("m", "hello")
    b = ReplayProvider(tmp_path).complete("m", "hello")
    assert a == b


def test_replay_miss_is_loud(tmp_path: Path) -> None:
    with pytest.raises(LookupError):
        ReplayProvider(tmp_path).complete("m", "unseen")
