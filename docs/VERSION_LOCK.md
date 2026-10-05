# VERSION_LOCK.md
## LLM Evaluation & AI Quality Engineering: technical version freeze (gate C4)

**Frozen:** 18 September 2026
**Blueprint:** Revision 1.0 (locked)
**Result:** C4 — PASS, with two recorded corrections (§4) and three open items (§7)

## 1. Status legend

| Status | Meaning |
|---|---|
| VERIFIED | Installed in a clean environment, conflict-checked with `uv pip check`, and exercised by an executed smoke test |
| FAILED | Did not install, resolve, or run as proposed |
| NOT TESTED | Not exercised in this gate; the reason is given |
| OPTIONAL/COMPARISON ONLY | Installed and import-checked only; the book compares it but ships no code that depends on it |

## 2. Test environment

| Item | Value |
|---|---|
| Primary interpreter | CPython 3.12.3 (Ubuntu 24.04) |
| Secondary interpreter | CPython 3.13.13 (via `uv python install 3.13`) |
| Installer / resolver | `uv` (clean virtual environment per profile) |
| Network | PyPI reachable; model provider APIs **not** reachable; no API keys used |
| Docker | Not available in the gate environment |

Five isolated environments were built from nothing: `core` (3.12), `core313` (3.13), `adapters` (3.12), `ragasenv` (3.12), and `cmp` (3.12).

## 3. Final exact version lock

### 3.1 Core (`aiqe` runtime, vendor-neutral)

| Package | Version | Status |
|---|---|---|
| Python | ≥3.12 (tested 3.12.3 and 3.13.13) | VERIFIED |
| pydantic | 2.13.5 | VERIFIED |
| scipy | 1.18.1 | VERIFIED (requires Python ≥3.12; confirms the floor) |
| numpy | 2.5.3 (transitive) | VERIFIED |
| statsmodels | 0.15.0 | VERIFIED |
| krippendorff | 0.8.2 | VERIFIED |
| opentelemetry-api | 1.44.0 | VERIFIED |
| opentelemetry-sdk | 1.44.0 | VERIFIED |

No model-provider SDK is a dependency of the core. Core statistics and agreement code imports nothing vendor-specific.

### 3.2 Development

| Package | Version | Status |
|---|---|---|
| pytest | 9.1.1 | VERIFIED |
| mypy | 2.3.1 | VERIFIED (`--strict` clean on the `aiqe` provider prototype, 3.12 and 3.13) |

### 3.3 Provider adapters (extra: `providers`)

| Package | Version | Status |
|---|---|---|
| openai | 3.15.0 | VERIFIED offline (Responses API against a local stub server) |
| anthropic | 1.6.0 | VERIFIED offline (Messages API against a local stub server) |

Live calls to real provider endpoints: **NOT TESTED** (no network access to providers, no keys). See §7.

### 3.4 Platform (extra: `platform`)

| Package | Version | Status |
|---|---|---|
| opentelemetry-exporter-otlp-proto-http | 1.44.0 | VERIFIED. **Added** (see §4.2) |

### 3.5 Tool adapters (extra: `adapters`, one shared environment)

| Package | Version | Status |
|---|---|---|
| deepeval | 4.2.3 | VERIFIED offline |
| arize-phoenix | 20.14.0 | VERIFIED locally (server + client round trip) |
| openinference-instrumentation | 0.1.65 | VERIFIED (with Phoenix) |
| inspect-ai | 0.3.265 | VERIFIED offline with one environment caveat (§5.6) |

These four packages co-install with the full core and provider lock; `uv pip check` reports no conflicts (216 packages). Core and provider smoke tests were re-run inside this environment and pass.

### 3.6 Ragas adapter (separate environment, not in the `adapters` extra)

| Package | Version | Status |
|---|---|---|
| ragas | 0.4.3 | VERIFIED in isolation, after corrections (§4.1) |
| langchain-community | 0.4.1 | **Added pin** (correction) |
| openai (resolved inside this env only) | 3.3.0 | Constraint forced by Ragas's dependency chain; **not** the book's provider pin |
| instructor | 1.17.0 | Transitive |
| jiter | 0.14.0 | Transitive |

### 3.7 Comparison only

| Package | Version | Status |
|---|---|---|
| mlflow | 3.16.1 | OPTIONAL/COMPARISON ONLY. Installs and imports alongside the core lock; `mlflow.genai` present |
| langfuse | 4.15.4 | OPTIONAL/COMPARISON ONLY. Installs and imports alongside the core lock |
| langsmith | 0.12.6 | OPTIONAL/COMPARISON ONLY. Installs and imports alongside the core lock |
| promptfoo | (Node.js CLI) | OPTIONAL/COMPARISON ONLY. NOT TESTED; not a Python dependency |
| OpenAI hosted evaluation features | platform service | OPTIONAL/COMPARISON ONLY. NOT TESTED |

### 3.8 Avoided

| Package | Reason |
|---|---|
| evals (legacy `openai/evals`) | No release since May 2024 |

## 4. Corrections to the blueprint candidates

No proposed version was changed. Two issues required corrections, both confined to optional material.

### 4.1 Ragas 0.4.3 cannot share an environment with openai 3.15.0 — FAILED as proposed, corrected

**Finding 1: dependency conflict.**
- Ragas 0.4.3 depends on `instructor`.
- The instructor releases that accept openai 3.x cap `jiter` below 0.15.
- openai 3.4.0 and later require `jiter>=0.16`.
- Resolution fails against openai 3.15.0. It also fails against 3.14.0, 3.12.0, 3.10.0, 3.8.0, 3.6.0 and 3.4.0; all of these were tested.
- The resolver only succeeds by selecting openai 3.3.0.

**Finding 2: import failure.** Even in isolation, `import ragas` fails with the default resolution. `ragas.llms.base` imports `langchain_community.chat_models.vertexai`, which is absent from langchain-community 0.4.2. Pinning `langchain-community==0.4.1` restores the import; 0.4 also works.

**Smallest compatible correction:**
1. Keep ragas 0.4.3.
2. Add `langchain-community==0.4.1`.
3. Move Ragas out of the shared `adapters` extra into its own lock file (`requirements/lock-adapter-ragas.txt`), with its own CI job.
4. The book's core and provider pins are unchanged; openai stays at 3.15.0.

This is consistent with the locked blueprint: Ragas was already "compare / adapter demo in Ch 12 only" and adapters must be isolated and removable. A separate environment is stricter isolation, not a structural change.

**Consequence for Chapter 12:** the Ragas demonstration uses the offline, non-LLM metrics that were verified here:
- `_IDBasedContextPrecision` and `_IDBasedContextRecall`;
- `ExactMatch` and `StringPresence` from `ragas.metrics.collections`.

Ragas's LLM-judged metrics are NOT TESTED (§7).

**Note:** in 0.4.3, the legacy metric classes are underscore-prefixed in `ragas.metrics`. The newer API lives in `ragas.metrics.collections`, but not every legacy metric has a collections equivalent (the ID-based ones do not).

### 4.2 OTLP/HTTP exporter added to the `platform` extra

The core tracing path must be able to export to any OTLP backend without Phoenix or OpenInference installed. `opentelemetry-exporter-otlp-proto-http==1.44.0` (same release train as the SDK) was added and verified: plain OTel SDK → OTLP/HTTP → a local backend, with the span read back. This is an addition required for vendor neutrality, not a version change.

## 5. Smoke-test evidence

All tests below were executed; the files are in `c4-smoke/`.

### 5.1 Core (`test_core.py`) — 7 passed on 3.12.3 and 3.13.13

- pydantic `EvalCase` schema round-trip and JSON schema.
- `scipy.stats.binomtest(...).proportion_ci(method="wilson")`.
- `scipy.stats.bootstrap` (percentile) and `wilcoxon`.
- statsmodels `mcnemar` (exact), `proportion_confint` (Wilson), `inter_rater.cohens_kappa`.
- `krippendorff.alpha` (nominal, with missing values).
- OTel SDK nested spans captured by `InMemorySpanExporter`.

### 5.2 Providers (`test_providers.py`) — 2 passed on 3.12.3 and 3.13.13

- OpenAI `client.responses.create(...)`: request routed to `/v1/responses`, `output_text` and `usage` parsed.
- Anthropic `client.messages.create(...)`: request routed to `/v1/messages`, text blocks and `usage` parsed.
- Both run against a local stub HTTP server.

### 5.3 Vendor-neutral provider protocol and replay (`aiqe_prototype/`)

- `Provider` protocol, `ReplayProvider`, `RecordingProvider`, `OpenAIProvider`, and `AnthropicProvider`.
- `mypy --strict`: no issues in 5 files (3.12 and 3.13).
- Tests (2 passed): record-then-replay equality, and a replay miss raises loudly.
- This confirms the offline-replay-by-default design is implementable with the locked stack.

### 5.4 DeepEval (`test_deepeval_adapter.py`) — 2 passed, offline

- An `aiqe`-style code evaluator wrapped as a DeepEval `BaseMetric` passes through `assert_test(..., run_async=False)`.
- `GEval` runs with a custom `DeepEvalBaseLLM` returning recorded judge output. This shows DeepEval judges can be driven by the book's replay path.
- API notes for the manuscript:
  - `evaluation_params` uses `SingleTurnParams` (not the older `LLMTestCaseParams`).
  - Set `DEEPEVAL_TELEMETRY_OPT_OUT=YES`.
  - DeepEval registers a pytest plugin when installed. Keep it out of the core test environment; core tests still pass inside the adapters environment.

### 5.5 Arize Phoenix (`test_phoenix_adapter.py`, `test_otlp_neutral.py`) — 2 passed

- `phoenix serve` started locally, with no external services, and was healthy in about 12 seconds.
- Spans exported via `phoenix.otel.register(...)` and, separately, via plain OTel SDK + OTLP/HTTP.
- Both were read back with `phoenix.client.Client().spans.get_spans_dataframe(project_identifier=...)`.

### 5.6 Inspect AI (`test_inspect_adapter.py`) — 1 passed, with caveat

- A task with a `@tool` (`track_shipment`), `use_tools` + `generate`, and the `includes()` scorer, run against `mockllm/model` with scripted `custom_outputs`: one tool call, then a final answer.
- Log status `success`; accuracy 1.0; trajectory `["track_shipment"]` extracted from the sample messages.
- **Caveat:** Inspect's mockllm counts tokens with `tiktoken.get_encoding("o200k_base")`. This downloads the BPE file once from `openaipublic.blob.core.windows.net`, which the gate environment blocks. The test therefore patches the token estimator.
  - For readers with internet access, the one-time download is automatic.
  - For fully offline CI, pre-seed `TIKTOKEN_CACHE_DIR` in the workflow. The pre-seeding step itself is NOT TESTED here.

### 5.7 Ragas (`test_ragas_adapter.py`) — 2 passed in the isolated environment

- ID-based context precision = 1/3 and recall = 1.0 on a three-document retrieval with one relevant ID.
- `ExactMatch` and `StringPresence` = 1.0.

### 5.8 Dependency conflict checks

| Environment | Packages | `uv pip check` |
|---|---|---|
| core (3.12) | 41 | All compatible |
| core313 (3.13) | 41 | All compatible |
| adapters (3.12) | 216 | All compatible |
| ragasenv (3.12) | 111 | All compatible (after §4.1) |
| cmp (3.12) | 121 | All compatible |

The adapters set also resolves for Python 3.13 (resolution only; not installed on 3.13).

## 6. Rules carried into Phase 2

1. The core `aiqe` package depends only on §3.1. Providers, platform exporters, and tool adapters are extras.
2. Ragas is never installed in the same environment as the provider extra. It has its own lock file and CI job.
3. All numbers published in the book come from the offline replay path. Live runs are optional and never the source of published figures.
4. Exact pins (`==`) everywhere. Transitive versions are fixed by the lock files in `requirements/`.
5. Any change after this freeze is a recorded baseline revision in this file, with the reason and re-run evidence.
6. CI matrix: Python 3.12 and 3.13.
7. `DEEPEVAL_TELEMETRY_OPT_OUT=YES` in all adapter jobs; `TIKTOKEN_CACHE_DIR` pre-seeded in the Inspect job.

## 7. Open items (not blocking C4)

| Item | Status | When it closes |
|---|---|---|
| Live calls to OpenAI and Anthropic endpoints | NOT TESTED (no provider access or keys) | First recording session in Phase 2, before any replay fixture is committed |
| Judge-model and system-model snapshot IDs | NOT DETERMINED (must be chosen from live model lists) | Same recording session; recorded in the judge configuration record (T11) |
| Ragas LLM-judged metrics offline | NOT TESTED | Only if Chapter 12 needs them; not required by the blueprint |
| `TIKTOKEN_CACHE_DIR` pre-seeding for Inspect in offline CI | NOT TESTED | Phase 3 CI build |
| Docker image build (P5) | NOT TESTED (no Docker in gate environment) | Phase 5 gate item, per blueprint |
| GitHub Actions on a real runner | NOT TESTED | Phase 5 gate item, per blueprint |

## 8. Reproducing this freeze

```bash
uv venv core --python 3.12
uv pip install --python core/bin/python \
  -r requirements/lock-core-providers-dev.txt
uv pip check --python core/bin/python
core/bin/python -m pytest c4-smoke/test_core.py \
  c4-smoke/test_providers.py -q

uv venv adapters --python 3.12
uv pip install --python adapters/bin/python \
  -r requirements/lock-adapters.txt

uv venv ragasenv --python 3.12
uv pip install --python ragasenv/bin/python \
  -r requirements/lock-adapter-ragas.txt
```

Phoenix tests require `phoenix serve` running on port 6006 in the same session.

## 9. Gate decision

- Every proposed package version installs, resolves, and passes its smoke test.
- The one incompatibility (Ragas with openai 3.x) is contained by isolating an adapter that was already optional.
- The vendor-neutral core and the offline replay default are both demonstrated in executed code.

C4 — PASS

## 10. Phase 3 additions (recorded, not silent)

No package pin in sections 3.1 to 3.8 changed. Added:

| Item | Version | Purpose | Status |
|---|---|---|---|
| uv | 0.11.7 | Installer used for the C4 freeze; pinned in CI | VERIFIED locally |
| hatchling | 1.32.3 | Build backend for `pip install -e .` (build-only) | VERIFIED |
| actions/checkout | v7.0.1 (3d3c42e5aac5ba805825da76410c181273ba90b1) | CI | Tag and SHA verified; not run on a real runner |
| astral-sh/setup-uv | v10.1.0 (bec219d24cd3e171d82865faccec33120bb574f4) | CI | Tag and SHA verified; inputs checked against action.yml |
| actions/upload-artifact | v7.0.1 (043fb46d1a93c77aae656e7c1c64a875d1fc6a0a) | CI | Tag and SHA verified |
| actions/download-artifact | v8.0.1 | CI | Tag verified; inputs checked |
| uv.lock | generated with uv 0.11.7 | `uv run` resolution | Shared transitive versions identical to lock-core-providers-dev.txt |

Type-checking exceptions: mypy `ignore_missing_imports` for
`krippendorff`, `scipy.*`, `statsmodels.*` (no type information
shipped). Adapter comparison scripts under `adapters/` are not
held to `mypy --strict` because DeepEval's own type information
is incomplete.

API finding: in anthropic 1.6.0, `Messages.create` exposes no
`temperature`, `top_p`, or `top_k` parameters; the Anthropic
adapter therefore sets no sampling parameters. The OpenAI
adapter passes `temperature` to `Responses.create` (verified
against a local stub server). Recordings are stored under
`replay_root(base, sampling)`, so a sampling change causes a
loud replay miss.

Inspect AI offline CI: `scripts/seed_tiktoken_cache.py`
reproduces tiktoken's cache layout (TIKTOKEN_CACHE_DIR, SHA-1
of the URL as file name, SHA-256 check of the contents, taken
from tiktoken's source). The seeding step itself needs network
access and has NOT been executed.

## 11. Phase 5 addition (book production only)

| Item | Version | Purpose | Status |
|---|---|---|---|
| matplotlib | 3.11.2 | `scripts/make_figures.py` and `figures/make_diagrams.py` redraw the book's figures; not a dependency of `aiqe` | VERIFIED locally (figures produced; script tests pass) |

Fonts used for the figures and the interior: EB Garamond, Lato, and
JetBrains Mono. They are not installed by this repository.
