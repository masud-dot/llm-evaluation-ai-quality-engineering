"""Comparison only. Runs in the isolated Ragas environment."""
import asyncio

from ragas import SingleTurnSample
from ragas.metrics import (_IDBasedContextPrecision,
                           _IDBasedContextRecall)

sample = SingleTurnSample(
    user_input="How long do I have to claim for a broken item?",
    retrieved_context_ids=["POL-CLM-2", "POL-CLM-1",
                           "POL-FAQ-9"],
    reference_context_ids=["POL-CLM-1"])

precision = asyncio.run(
    _IDBasedContextPrecision().single_turn_ascore(sample))
recall = asyncio.run(
    _IDBasedContextRecall().single_turn_ascore(sample))
print(f"precision={precision:.4f} recall={recall:.4f}")
