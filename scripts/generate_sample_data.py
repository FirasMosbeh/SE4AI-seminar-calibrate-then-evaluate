"""Generate sample datasets for the demo pipeline.

* ``data/calibration_pairs.jsonl`` — 50 pairs of equal-quality responses
  (Phase 1). Half the pairs have one materially longer response (verbosity
  probe); half tag one response as same-family via '[mock]' (self-preference
  probe for the MockJudge; with a real judge you would insert actual outputs
  of the judge's own model family here).
* ``data/eval_items.jsonl`` — 40 items comparing a 'candidate' model output
  against a 'baseline' output (Phase 2).

The texts are synthetic but structurally realistic. For a real study,
replace these with human-labelled equal-quality pairs (Phase 1) and real
model outputs (Phase 2).
"""

import json
import random
from pathlib import Path

random.seed(42)

STYLES = ["", " in simple terms", " for a beginner", " concisely",
          " with an example", " step by step", " for an expert audience"]

TOPICS = [
    "explain what a hash map is",
    "summarise the causes of the French Revolution",
    "describe how photosynthesis works",
    "explain the difference between TCP and UDP",
    "give tips for writing a good CV",
    "explain what overfitting means in machine learning",
    "describe the water cycle",
    "explain what a REST API is",
    "summarise the plot structure of a three-act story",
    "explain why the sky is blue",
]

SHORT = (
    "{topic_cap}: it is best understood through its core mechanism. "
    "The key idea is straightforward and covers the main cases."
)
LONG = (
    "{topic_cap}: to answer this thoroughly, consider the underlying "
    "mechanism, the typical use cases, and the common misconceptions. "
    "First, the core principle operates consistently across contexts. "
    "Second, practical applications illustrate the idea concretely. "
    "Third, edge cases are worth noting because they reveal the limits "
    "of the simple explanation. Taken together these points give a "
    "complete picture of the concept in question."
)
NEUTRAL_A = (
    "{topic_cap}: the concept rests on a small number of principles that "
    "interact in predictable ways, and a short example makes them concrete."
)
NEUTRAL_B = (
    "{topic_cap}: one can capture the essentials with a compact definition "
    "followed by a brief illustration that shows the idea in action."
)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # ---- Phase 1: 50 equal-quality calibration pairs -------------------
    pairs = []
    for i in range(50):
        topic = TOPICS[i % len(TOPICS)]
        cap = topic[0].upper() + topic[1:]
        instr = f"Please {topic}{STYLES[i % len(STYLES)]}."
        kind = i % 4
        if kind == 0:  # verbosity probe: response 1 longer
            pairs.append(dict(instruction=instr,
                              response_1=LONG.format(topic_cap=cap),
                              response_2=SHORT.format(topic_cap=cap),
                              longer=1, own_family=0))
        elif kind == 1:  # verbosity probe: response 2 longer
            pairs.append(dict(instruction=instr,
                              response_1=SHORT.format(topic_cap=cap),
                              response_2=LONG.format(topic_cap=cap),
                              longer=2, own_family=0))
        elif kind == 2:  # self-preference probe: response 1 same-family
            pairs.append(dict(instruction=instr,
                              response_1="[mock] " + NEUTRAL_A.format(topic_cap=cap),
                              response_2=NEUTRAL_B.format(topic_cap=cap),
                              longer=0, own_family=1))
        else:            # self-preference probe: response 2 same-family
            pairs.append(dict(instruction=instr,
                              response_1=NEUTRAL_A.format(topic_cap=cap),
                              response_2="[mock] " + NEUTRAL_B.format(topic_cap=cap),
                              longer=0, own_family=2))

    with open(data_dir / "calibration_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # ---- Phase 2: 40 candidate-vs-baseline items ------------------------
    items = []
    for i in range(40):
        topic = TOPICS[i % len(TOPICS)]
        cap = topic[0].upper() + topic[1:]
        instr = f"Please {topic}{STYLES[i % len(STYLES)]}."
        items.append(dict(
            instruction=instr,
            candidate=NEUTRAL_A.format(topic_cap=cap) + " (candidate v2 output)",
            baseline=NEUTRAL_B.format(topic_cap=cap) + " (baseline v1 output)",
        ))

    with open(data_dir / "eval_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"Wrote {len(pairs)} calibration pairs and {len(items)} eval items to {data_dir}")


if __name__ == "__main__":
    main()
