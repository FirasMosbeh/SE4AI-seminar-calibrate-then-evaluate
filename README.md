# Calibrate-Then-Evaluate

Reference implementation of the **Calibrate-Then-Evaluate** protocol proposed in:

> *Prompt Sensitivity in LLM Evaluation Frameworks: A Comparative Literature
> Review on Reliability and Reproducibility* — SE4AI Seminar, TUM, Summer 2026.

The paper's central claim, made executable: **averaging over many prompt
variants reduces variance but not systematic judge bias** — formally, the
observed score decomposes as `s_i = q + b + ε_i`, and multi-prompt averaging
estimates `q + b`, never `q`. This repo therefore measures `b` *first*
(Phase 1) and only then evaluates with variance made visible (Phase 2).

```
Phase 1 — Calibration (once per judge / task type)
  ~50 equal-quality pairs × 3–5 prompt variants × both orders
  → measure position, verbosity, self-preference bias
  → any bias > 10 pp?  →  REJECT JUDGE (exit 1)

Phase 2 — Evaluation (per model version)
  k = 3 prompt variants (3× cost, not 15×)
  → report min / median / max win-rate
  → range > 15 pp?  →  UNSTABLE, no promotion decision (exit 2)
  → otherwise: score usable as CI regression gate (exit 0/1)
```

## Quickstart (zero API cost)

```bash
pip install -e ".[dev]"
python scripts/generate_sample_data.py   # regenerate the demo datasets
pytest                                   # 8 tests, incl. bias-injection checks

# Full protocol with the free deterministic MockJudge:
cte pipeline --judge mock

# Watch Phase 1 catch an injected verbosity bias:
cte pipeline --judge mock --mock-verbosity-bias 0.3

# The paper's key result, empirically (bias survives averaging over k):
python examples/demo_biased_judge.py
```

## Using a real judge

```bash
pip install -e ".[anthropic]"       # or .[openai]
export ANTHROPIC_API_KEY=sk-ant-...
cte pipeline --judge anthropic
```

Model names are configurable via `CTE_ANTHROPIC_MODEL` / `CTE_OPENAI_MODEL`;

Cost estimate for the defaults: Phase 1 = 50 pairs × 3 variants × 2 orders
= **300 judge calls (once per judge/task type)**; Phase 2 = 40 items × 3
variants × 2 orders = **240 calls per model version** — the 3× (not 15×)
regime argued for in the paper.


## Repository layout

```
src/cte/judge.py         # judge abstraction + 5 versioned prompt variants
src/cte/calibration.py   # Phase 1: joint bias measurement
src/cte/evaluation.py    # Phase 2: k=3 evaluation + stability gate
src/cte/cli.py           # `cte calibrate | evaluate | pipeline`
data/                    # sample datasets (JSONL)
scripts/                 # reproducible data generation
tests/                   # incl. bias-injection validation tests
examples/                # demo: bias survives averaging
.github/workflows/       # the CI regression gate
```

## Limitations:

* Sample data is synthetic; a real study needs human-labelled equal-quality
  pairs and genuine model outputs.
* The self-preference probe uses a `[mock]` tag; with a real judge you would
  insert actual outputs from the judge's own model family.
* Thresholds (10 pp / 15 pp) are the paper's heuristics — validating them is
  exactly the future work the paper proposes.
