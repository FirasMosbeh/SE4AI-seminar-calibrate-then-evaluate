"""Calibrate-Then-Evaluate: a two-phase LLM evaluation protocol.

Reference implementation of the protocol proposed in:
"Prompt Sensitivity in LLM Evaluation Frameworks: A Comparative
Literature Review on Reliability and Reproducibility" (SE4AI Seminar,
TUM, Summer 2026).
"""

from .calibration import calibrate, load_pairs, CalibrationPair, CalibrationReport
from .evaluation import evaluate, load_items, EvalItem, EvaluationReport
from .judge import get_judge, MockJudge, PAIRWISE_PROMPT_VARIANTS

__all__ = [
    "calibrate", "load_pairs", "CalibrationPair", "CalibrationReport",
    "evaluate", "load_items", "EvalItem", "EvaluationReport",
    "get_judge", "MockJudge", "PAIRWISE_PROMPT_VARIANTS",
]
__version__ = "0.1.0"
