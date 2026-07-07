# Calibrate-Then-Evaluate

Reference implementation of the **Calibrate-Then-Evaluate** protocol from:

> *Prompt Sensitivity in LLM Evaluation Frameworks: Why More Stable Scores Do Not Necessarily Mean
> Better Measurements — SE4AI Seminar, TUM, Summer 2026.

The paper argues that averaging scores over many prompt variants reduces
random variance, but not a judge's systematic bias. Formally, the observed
score decomposes as `s_i = q + b + ε_i`, so multi-prompt averaging only ever
gives you `q + b`, never the true quality `q`. This repo implements that
idea directly: it measures the bias term `b` first (Phase 1), then runs the
evaluation with the remaining variance reported explicitly instead of hidden
behind an average (Phase 2).

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

# Run the full protocol with the free deterministic MockJudge:
cte pipeline --judge mock

# Inject a verbosity bias and see Phase 1 flag it:
cte pipeline --judge mock --mock-verbosity-bias 0.3

# Reproduce the paper's main result: bias survives averaging over k:
python examples/demo_biased_judge.py
```

## Using a real judge

```bash
pip install -e ".[anthropic]"       # or .[openai]
export ANTHROPIC_API_KEY=sk-ant-...
cte pipeline --judge anthropic
```

Model names are configurable via `CTE_ANTHROPIC_MODEL` / `CTE_OPENAI_MODEL`.

Cost estimate with the defaults: Phase 1 uses 50 pairs × 3 variants × 2
orders = **300 judge calls** (once per judge/task type). Phase 2 uses 40
items × 3 variants × 2 orders = **240 calls per model version**. That's the
3× cost the paper argues for, instead of the usual 15×.

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

## Limitations

- Sample data is synthetic; a real study would need human-labelled
  equal-quality pairs and real model outputs.
- The self-preference probe uses a `[mock]` tag; with a real judge you'd
  insert actual outputs from that judge's own model family instead.
- The 10pp / 15pp thresholds are heuristics from the paper. Whether they
  actually hold up on real data is not tested here — that's future work.
