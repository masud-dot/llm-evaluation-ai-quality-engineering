from pathlib import Path

import pytest

from aiqe.datasets import Split, load_jsonl
from aiqe.evaluators.code import ClaimDeadline
from aiqe.stats import pass_hat_k
from aiqe.trials import flaky, run_trials, trial_providers

DS = load_jsonl(Path("datasets/compass/credit-claims-v1.jsonl"),
                "credit-claims", "1.0")


def test_missing_trial_recordings_raise(tmp_path: Path) -> None:
    (tmp_path / "trial-0").mkdir()
    with pytest.raises(FileNotFoundError):
        trial_providers(tmp_path, 2)
    assert len(trial_providers(tmp_path, 1)) == 1


def test_run_trials_and_flaky() -> None:
    good = "Report damage within 14 days of delivery."
    bad = "Report damage within 7 days."
    systems = [lambda q: good, lambda q: bad, lambda q: good]
    trials, deferred = run_trials(DS, Split.HOLDOUT, systems,
                                  ClaimDeadline())
    assert trials == {"cc-003": [True, False, True]}
    assert deferred == 6          # two non-claim cases x 3
    assert flaky(trials) == ["cc-003"]
    assert pass_hat_k(3, 2, 2) == pytest.approx(1 / 3)
