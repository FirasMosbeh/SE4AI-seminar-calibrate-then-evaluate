"""Demo: the central claim of the paper, executable.

We inject a verbosity bias into the judge and show that
(a) Phase 1 catches it, and
(b) averaging over MORE prompt variants does NOT make it go away —
    the confidence interval tightens while the bias b stays put
    (equation (1): E[s_bar] = q + b for any k).

Run:  python examples/demo_biased_judge.py
"""
from cte import MockJudge, calibrate, load_pairs

pairs = load_pairs("data/calibration_pairs.jsonl")
judge = MockJudge(verbosity_bias=0.3, noise=0.05)

print("=== Judge with hidden verbosity bias ===\n")
for variants in (["v1_plain", "v2_rubric", "v3_terse"],
                 ["v1_plain", "v2_rubric", "v3_terse", "v4_cot", "v5_role"]):
    report = calibrate(judge, pairs, variants)
    print(f"k = {len(variants)} prompt variants "
          f"-> measured verbosity bias: {report.verbosity_bias_pp:+.1f} pp "
          f"({'REJECTED' if not report.passed else 'passed'})")

print("\nMore variants != less bias. Averaging estimates q + b, never q.")
