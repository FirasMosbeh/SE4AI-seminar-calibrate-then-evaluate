"""Phase 2 -- Evaluation (run per model version).

Implements the second half of Section V-B: evaluate the candidate model
against a baseline with k=3 prompt variants (3x cost instead of 15x),
report min / median / max win-rate across variants, and apply the
stability gate: if the range exceeds ``range_threshold_pp`` (default
15 pp), the result is marked UNSTABLE and must not be used for a
promotion decision.

Position bias is neutralised mechanically here via balanced ordering
(each item is judged once as (candidate, baseline) and once swapped),
following Wang et al. (2024). That removes the one bias that CAN be
removed cheaply; verbosity and self-preference are the reason Phase 1
must run first.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from .judge import Judge, PAIRWISE_PROMPT_VARIANTS


@dataclass
class EvalItem:
    """One benchmark item: instruction + candidate & baseline responses."""

    instruction: str
    candidate: str
    baseline: str

    @staticmethod
    def from_dict(d: dict) -> "EvalItem":
        return EvalItem(d["instruction"], d["candidate"], d["baseline"])


@dataclass
class EvaluationReport:
    n_items: int
    variants: list[str]
    win_rate_per_variant: dict[str, float]  # candidate win-rate, percent
    min_pp: float
    median_pp: float
    max_pp: float
    range_pp: float
    range_threshold_pp: float
    stable: bool
    promotion_allowed: bool
    promotion_threshold_pp: float

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def summary(self) -> str:
        status = "STABLE ✅" if self.stable else "UNSTABLE — no promotion decision ❌"
        promo = "PROMOTE ✅" if self.promotion_allowed else "DO NOT PROMOTE ⛔"
        lines = [
            f"Evaluation report — {self.n_items} items × {len(self.variants)} variants",
            *(
                f"  {v}: candidate win-rate {wr:5.1f}%"
                for v, wr in self.win_rate_per_variant.items()
            ),
            f"  min / median / max: {self.min_pp:.1f} / {self.median_pp:.1f} / "
            f"{self.max_pp:.1f} %",
            f"  range: {self.range_pp:.1f} pp (threshold {self.range_threshold_pp:.0f} pp)"
            f" → {status}",
        ]
        if self.stable:
            lines.append(
                f"  median vs. promotion threshold "
                f"({self.promotion_threshold_pp:.0f}%): {promo}"
            )
        return "\n".join(lines)


def load_items(path: str | Path) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            items.append(EvalItem.from_dict(json.loads(line)))
    return items


def evaluate(
    judge: Judge,
    items: list[EvalItem],
    prompt_variants: list[str] | None = None,
    range_threshold_pp: float = 15.0,
    promotion_threshold_pp: float = 50.0,
) -> EvaluationReport:
    """Run Phase 2 and return the evaluation report.

    ``promotion_threshold_pp`` is the median win-rate the candidate must
    reach for the CI gate to pass (50% = candidate must at least tie the
    baseline).
    """
    variants = prompt_variants or list(PAIRWISE_PROMPT_VARIANTS)[:3]

    win_rates: dict[str, float] = {}
    for variant in variants:
        candidate_wins = 0.0
        for item in items:
            # balanced ordering: judge both (cand, base) and (base, cand)
            v1 = judge.compare(item.instruction, item.candidate, item.baseline, variant)
            v2 = judge.compare(item.instruction, item.baseline, item.candidate, variant)
            w1 = 1.0 if v1.winner == "A" else 0.0  # candidate sat in slot A
            w2 = 1.0 if v2.winner == "B" else 0.0  # candidate sat in slot B
            candidate_wins += (w1 + w2) / 2.0
        win_rates[variant] = round(100.0 * candidate_wins / len(items), 2)

    values = list(win_rates.values())
    mn, md, mx = min(values), statistics.median(values), max(values)
    rng = mx - mn
    stable = rng <= range_threshold_pp
    promotion_allowed = stable and md >= promotion_threshold_pp

    return EvaluationReport(
        n_items=len(items),
        variants=variants,
        win_rate_per_variant=win_rates,
        min_pp=round(mn, 2),
        median_pp=round(md, 2),
        max_pp=round(mx, 2),
        range_pp=round(rng, 2),
        range_threshold_pp=range_threshold_pp,
        stable=stable,
        promotion_allowed=promotion_allowed,
        promotion_threshold_pp=promotion_threshold_pp,
    )
