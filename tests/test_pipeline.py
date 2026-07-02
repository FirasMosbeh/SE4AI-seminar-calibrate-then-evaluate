"""Tests for the Calibrate-Then-Evaluate pipeline.

The key scientific test: if we INJECT a known bias into the MockJudge,
Phase 1 must DETECT it. This validates the measurement logic itself —
the same trick a controlled study (paper, Sec. V-C) would use.
"""

from cte.calibration import calibrate, load_pairs
from cte.evaluation import evaluate, load_items
from cte.judge import MockJudge

PAIRS = load_pairs("data/calibration_pairs.jsonl")
ITEMS = load_items("data/eval_items.jsonl")
VARIANTS = ["v1_plain", "v2_rubric", "v3_terse"]


def test_unbiased_judge_passes_calibration():
    judge = MockJudge(noise=0.1)
    report = calibrate(judge, PAIRS, VARIANTS, threshold_pp=10.0)
    assert report.passed, report.summary()
    assert abs(report.position_bias_pp) <= 10
    assert abs(report.verbosity_bias_pp) <= 10
    assert abs(report.self_preference_pp) <= 10


def test_position_bias_is_detected():
    judge = MockJudge(position_bias=0.25, noise=0.05)
    report = calibrate(judge, PAIRS, VARIANTS, threshold_pp=10.0)
    assert not report.passed
    assert report.position_bias_pp > 10, report.summary()


def test_verbosity_bias_is_detected():
    judge = MockJudge(verbosity_bias=0.3, noise=0.05)
    report = calibrate(judge, PAIRS, VARIANTS, threshold_pp=10.0)
    assert not report.passed
    assert report.verbosity_bias_pp > 10, report.summary()


def test_self_preference_is_detected():
    judge = MockJudge(self_preference=0.3, noise=0.05)
    report = calibrate(judge, PAIRS, VARIANTS, threshold_pp=10.0)
    assert not report.passed
    assert report.self_preference_pp > 10, report.summary()


def test_bias_survives_averaging():
    """Equation (1) of the paper, empirically: adding more prompt variants
    does NOT shrink a systematic verbosity bias."""
    judge = MockJudge(verbosity_bias=0.3, noise=0.05)
    r3 = calibrate(judge, PAIRS, ["v1_plain", "v2_rubric", "v3_terse"])
    r5 = calibrate(judge, PAIRS, ["v1_plain", "v2_rubric", "v3_terse",
                                  "v4_cot", "v5_role"])
    # bias estimate stays large regardless of k — averaging estimates q+b
    assert r3.verbosity_bias_pp > 15
    assert r5.verbosity_bias_pp > 15


def test_evaluation_reports_and_gates():
    judge = MockJudge(noise=0.1)
    report = evaluate(judge, ITEMS, VARIANTS, range_threshold_pp=15.0)
    assert report.n_items == len(ITEMS)
    assert len(report.win_rate_per_variant) == 3
    assert report.range_pp == report.max_pp - report.min_pp
    # gate consistency
    if report.range_pp > 15.0:
        assert not report.stable and not report.promotion_allowed


def test_unstable_result_blocks_promotion():
    # huge noise → win-rates swing across variants → gate must trip
    judge = MockJudge(noise=0.5)
    report = evaluate(judge, ITEMS, VARIANTS, range_threshold_pp=1.0)
    assert not report.stable
    assert not report.promotion_allowed


def test_determinism():
    """Same inputs → identical reports (reproducibility requirement)."""
    j1, j2 = MockJudge(noise=0.1), MockJudge(noise=0.1)
    r1 = calibrate(j1, PAIRS, VARIANTS)
    r2 = calibrate(j2, PAIRS, VARIANTS)
    assert r1 == r2
