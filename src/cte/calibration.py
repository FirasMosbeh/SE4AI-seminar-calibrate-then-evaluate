"""Phase 1 -- Calibration (run once per judge model and task type).

Implements Section V-B of the paper: on a held-out set of ~50 pairs of
*known equal quality*, measure three systematic judge biases jointly:

* **Position bias** (Wang et al., 2024): does the judge prefer whichever
  response appears first? Measured by running every pair in both orders
  (A,B) and (B,A). An unbiased judge should pick slot A ~50% of the time.
* **Verbosity bias** (Dubois et al., 2024): do longer responses win more
  often than 50% on equal-quality pairs?
* **Self-preference** (Panickssery et al., 2024): do responses tagged as
  coming from the judge's own model family win more often than 50%?

Each bias is reported in percentage points of deviation from the 50%
no-bias baseline. If any bias exceeds ``threshold_pp`` (default 10 pp,
the heuristic from the paper), the judge is flagged as unreliable for
this task type BEFORE any evaluation starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .judge import Judge, PAIRWISE_PROMPT_VARIANTS


@dataclass
class CalibrationPair:
    """Two responses of (approximately) known equal quality."""

    instruction: str
    response_1: str
    response_2: str
    # metadata used by the bias probes:
    longer: int  # 1 or 2 -- which response is materially longer (0 = neither)
    own_family: int  # 1 or 2 -- which response is from the judge's family (0 = neither)

    @staticmethod
    def from_dict(d: dict) -> "CalibrationPair":
        return CalibrationPair(
            instruction=d["instruction"],
            response_1=d["response_1"],
            response_2=d["response_2"],
            longer=d.get("longer", 0),
            own_family=d.get("own_family", 0),
        )


@dataclass
class CalibrationReport:
    judge_family: str
    n_pairs: int
    n_variants: int
    n_judgements: int
    position_bias_pp: float
    verbosity_bias_pp: float
    self_preference_pp: float
    threshold_pp: float
    passed: bool
    per_variant_position_pp: dict[str, float]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def summary(self) -> str:
        flag = "PASS ✅" if self.passed else "REJECT JUDGE ❌"
        lines = [
            f"Calibration report — judge family: {self.judge_family}",
            f"  pairs: {self.n_pairs}  variants: {self.n_variants}  "
            f"judgements: {self.n_judgements}",
            f"  position bias:    {self.position_bias_pp:+6.1f} pp",
            f"  verbosity bias:   {self.verbosity_bias_pp:+6.1f} pp",
            f"  self-preference:  {self.self_preference_pp:+6.1f} pp",
            f"  threshold: ±{self.threshold_pp:.0f} pp  →  {flag}",
        ]
        return "\n".join(lines)


def load_pairs(path: str | Path) -> list[CalibrationPair]:
    """Load calibration pairs from a JSONL file."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            pairs.append(CalibrationPair.from_dict(json.loads(line)))
    return pairs


def calibrate(
    judge: Judge,
    pairs: list[CalibrationPair],
    prompt_variants: list[str] | None = None,
    threshold_pp: float = 10.0,
) -> CalibrationReport:
    """Run Phase 1 and return a calibration report.

    Every pair is judged in BOTH orders under every prompt variant, which is
    what allows position bias to be separated from genuine preference:
    on order-swapped equal-quality pairs, any consistent preference for slot
    A can only come from position.
    """
    variants = prompt_variants or list(PAIRWISE_PROMPT_VARIANTS)[:3]

    slot_a_wins = 0                    # judge picked whatever sat in slot A
    per_variant_slot_a: dict[str, list[int]] = {v: [] for v in variants}
    longer_wins, longer_total = 0, 0
    own_wins, own_total = 0, 0
    n_judgements = 0

    for pair in pairs:
        for variant in variants:
            for order in ("12", "21"):
                if order == "12":
                    a, b = pair.response_1, pair.response_2
                    slot_of = {1: "A", 2: "B"}
                else:
                    a, b = pair.response_2, pair.response_1
                    slot_of = {1: "B", 2: "A"}

                verdict = judge.compare(pair.instruction, a, b, variant)
                n_judgements += 1

                picked_slot_a = verdict.winner == "A"
                slot_a_wins += picked_slot_a
                per_variant_slot_a[variant].append(int(picked_slot_a))

                # which underlying response (1 or 2) won?
                winner_resp = 1 if slot_of[1] == verdict.winner else 2

                if pair.longer in (1, 2):
                    longer_total += 1
                    longer_wins += winner_resp == pair.longer
                if pair.own_family in (1, 2):
                    own_total += 1
                    own_wins += winner_resp == pair.own_family

    def pp(wins: int, total: int) -> float:
        return 0.0 if total == 0 else (wins / total - 0.5) * 100

    position_pp = pp(slot_a_wins, n_judgements)
    verbosity_pp = pp(longer_wins, longer_total)
    self_pref_pp = pp(own_wins, own_total)

    per_variant_pp = {
        v: pp(sum(xs), len(xs)) for v, xs in per_variant_slot_a.items()
    }

    passed = all(
        abs(x) <= threshold_pp for x in (position_pp, verbosity_pp, self_pref_pp)
    )

    return CalibrationReport(
        judge_family=judge.family,
        n_pairs=len(pairs),
        n_variants=len(variants),
        n_judgements=n_judgements,
        position_bias_pp=round(position_pp, 2),
        verbosity_bias_pp=round(verbosity_pp, 2),
        self_preference_pp=round(self_pref_pp, 2),
        threshold_pp=threshold_pp,
        passed=passed,
        per_variant_position_pp={k: round(v, 2) for k, v in per_variant_pp.items()},
    )
