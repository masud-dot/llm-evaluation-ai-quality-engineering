# llm-evaluation-ai-quality-engineering

Companion repository for the book **LLM Evaluation & AI Quality Engineering:
Measure, Regression-Test, Release, and Monitor LLM, RAG, and Agent Systems
with Python** by Masud Mondal.

It contains `aiqe`, the reader-owned evaluation toolkit built across the
book's five projects, and Compass, the fictional system under test it
evaluates.

**Measurement is evidence, not truth.** Evaluations here replay recorded
model responses by default, so every published result is reproducible, costs
nothing to re-run, and needs no API key.

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest -q
```

Provider adapters are only needed when recording new responses:

```bash
uv pip install -e ".[providers]"
```

## Layout

| Path | What it holds |
|---|---|
| `src/aiqe/` | The toolkit: plans, datasets, evaluators, judges, statistics, experiments, change routing, policies, gates, tracing, drift, triage, governance |
| `src/aiqe/adapters/` | Provider adapters (OpenAI, Anthropic) |
| `systems/compass/` | The system under test: prompt, policy corpus, single-turn, RAG, conversation, and agent wrappers, and the sandbox |
| `plans/`, `gates/`, `rubrics/`, `prompts/`, `judges/`, `suites/` | Evaluation plan v1.4, release policy v1.2, pull-request policy v1.0, rubrics, judge templates and configuration, suite routing |
| `datasets/` | Seed evaluation datasets with content hashes |
| `projects/` | Project 4 suite runner, Project 5 worker and capstone walkthrough |
| `scripts/` | Recording, labelling, figure regeneration, tokenizer-cache seeding |
| `figures/` | The book's figures and the script that draws them |
| `tests/` | 115 tests covering every chapter's code |
| `docs/VERSION_LOCK.md` | Every pinned version and what has and has not been verified |
| `.github/workflows/` | The quality gate and the adapter comparison jobs |

## How the pieces fit

```text
change -> aiqe.ci manifest -> aiqe.ci plan (impact) -> run_suites (replay)
       -> evidence -> aiqe.ci gate (policy + overrides) -> exit code
production -> traces -> sampling -> online evaluation -> triage
           -> promotion -> regression set -> the gate above
```

## Status

Verified in the author's environment: 115 tests pass on Python 3.12 and
3.13; `mypy --strict` is clean; the pipeline blocks a deliberately degraded
change, passes a neutral one, and fails closed when a recording is missing;
OpenTelemetry spans reach an OTLP backend; provider adapters work against a
local stub server.

Not yet done, and not claimed anywhere: live provider calls, Compass
baseline recordings, the measured versions of Figures 9, 10 and 16, a run on
a GitHub-hosted runner, and a Docker build. `docs/VERSION_LOCK.md` records
the detail.

## Recording your own fixtures

```bash
python scripts/record_fixtures.py --provider stub --dry-run      # plan only
python scripts/record_fixtures.py --provider openai --trials 5   # needs a key
python scripts/label_batch.py export --run build/run-critical.json
python scripts/make_figures.py --out figures/                    # measured F9/F10/F16
```

`make_figures.py` refuses to draw a figure whose inputs are missing rather
than inventing values.

## Licence

MIT, see `LICENSE`. The book text is not covered by this licence.
