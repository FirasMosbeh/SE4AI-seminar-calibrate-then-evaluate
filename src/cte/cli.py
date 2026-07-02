"""Command-line interface for the Calibrate-Then-Evaluate pipeline.

Exit codes are designed for CI:

* ``cte calibrate``  → exit 0 if the judge passes calibration, exit 1 if it
  must be rejected (Fig. 3, Phase 1 gate).
* ``cte evaluate``   → exit 0 if the result is stable AND the candidate may
  be promoted; exit 1 if stable but candidate loses; exit 2 if UNSTABLE
  (range > threshold → no promotion decision allowed).
* ``cte pipeline``   → runs both phases; refuses to evaluate with a judge
  that failed calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibration import calibrate, load_pairs
from .evaluation import evaluate, load_items
from .judge import PAIRWISE_PROMPT_VARIANTS, get_judge


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--judge",
        default="mock",
        choices=["mock", "anthropic", "openai"],
        help="Judge backend (default: mock — free, deterministic, offline).",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(PAIRWISE_PROMPT_VARIANTS)[:3],
        choices=list(PAIRWISE_PROMPT_VARIANTS),
        help="Prompt variants to use (default: first 3, i.e. k=3).",
    )
    # Mock-judge bias injection, for demos and validation:
    p.add_argument("--mock-position-bias", type=float, default=0.0)
    p.add_argument("--mock-verbosity-bias", type=float, default=0.0)
    p.add_argument("--mock-self-preference", type=float, default=0.0)
    p.add_argument("--mock-noise", type=float, default=0.1)


def _make_judge(args):
    if args.judge == "mock":
        return get_judge(
            "mock",
            position_bias=args.mock_position_bias,
            verbosity_bias=args.mock_verbosity_bias,
            self_preference=args.mock_self_preference,
            noise=args.mock_noise,
        )
    return get_judge(args.judge)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cte", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Phase 1: measure judge biases")
    _add_judge_args(p_cal)
    p_cal.add_argument("--pairs", default="data/calibration_pairs.jsonl")
    p_cal.add_argument("--threshold-pp", type=float, default=10.0)
    p_cal.add_argument("--out", default="reports/calibration.json")

    p_ev = sub.add_parser("evaluate", help="Phase 2: evaluate candidate vs baseline")
    _add_judge_args(p_ev)
    p_ev.add_argument("--items", default="data/eval_items.jsonl")
    p_ev.add_argument("--range-threshold-pp", type=float, default=15.0)
    p_ev.add_argument("--promotion-threshold-pp", type=float, default=50.0)
    p_ev.add_argument("--out", default="reports/evaluation.json")

    p_pipe = sub.add_parser("pipeline", help="Phase 1 then Phase 2 (full protocol)")
    _add_judge_args(p_pipe)
    p_pipe.add_argument("--pairs", default="data/calibration_pairs.jsonl")
    p_pipe.add_argument("--items", default="data/eval_items.jsonl")
    p_pipe.add_argument("--threshold-pp", type=float, default=10.0)
    p_pipe.add_argument("--range-threshold-pp", type=float, default=15.0)
    p_pipe.add_argument("--promotion-threshold-pp", type=float, default=50.0)
    p_pipe.add_argument("--out-dir", default="reports")

    args = parser.parse_args(argv)
    judge = _make_judge(args)

    if args.cmd == "calibrate":
        report = calibrate(
            judge, load_pairs(args.pairs), args.variants, args.threshold_pp
        )
        print(report.summary())
        _save(report, args.out)
        return 0 if report.passed else 1

    if args.cmd == "evaluate":
        report = evaluate(
            judge,
            load_items(args.items),
            args.variants,
            args.range_threshold_pp,
            args.promotion_threshold_pp,
        )
        print(report.summary())
        _save(report, args.out)
        if not report.stable:
            return 2
        return 0 if report.promotion_allowed else 1

    if args.cmd == "pipeline":
        out_dir = Path(args.out_dir)
        cal = calibrate(
            judge, load_pairs(args.pairs), args.variants, args.threshold_pp
        )
        print(cal.summary())
        _save(cal, out_dir / "calibration.json")
        if not cal.passed:
            print("\nJudge rejected in Phase 1 — evaluation will not run.")
            return 1

        print()
        ev = evaluate(
            judge,
            load_items(args.items),
            args.variants,
            args.range_threshold_pp,
            args.promotion_threshold_pp,
        )
        print(ev.summary())
        _save(ev, out_dir / "evaluation.json")
        if not ev.stable:
            return 2
        return 0 if ev.promotion_allowed else 1

    return 0


def _save(report, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(path)
    print(f"  → report written to {path}")


if __name__ == "__main__":
    sys.exit(main())
