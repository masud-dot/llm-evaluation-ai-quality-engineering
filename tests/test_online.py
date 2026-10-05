from datetime import datetime, timezone

from aiqe.evaluators.code import (CommitmentPreFilter,
                                  ContactAllowList)
from aiqe.online import evaluate_sampled

NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
DIRECTORY = ContactAllowList(
    emails={"claims@tidewater-logistics.example"},
    phones={"020 7946 0321"}, urls=set())


def test_online_scores_are_labelled() -> None:
    out = evaluate_sampled(
        "conv-88", "Late parcel, what now?",
        "You'll receive a credit. Call 020 7946 0999.",
        ["OBJ-CREDIT-01", "OBJ-CONTACT-01"],
        [CommitmentPreFilter(), DIRECTORY], NOW)
    assert [o.setting for o in out] == ["online", "online"]
    assert [o.score.passed for o in out] == [False, False]
    assert out[0].score.case_id == "conv-88"
