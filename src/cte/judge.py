"""Judge abstractions for the Calibrate-Then-Evaluate protocol.

A Judge performs a pairwise comparison: given a task instruction and two
candidate responses (A and B), it returns which response it prefers.

Three implementations are provided:

* ``AnthropicJudge`` – uses the Anthropic Messages API.
* ``OpenAIJudge``    – uses the OpenAI Chat Completions API.
* ``MockJudge``      – deterministic, offline judge with *injectable biases*
  (position / verbosity / self-preference). This lets you demo and unit-test
  the whole pipeline without spending a single API call, and lets you verify
  that Phase 1 actually detects the biases you inject.
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


PAIRWISE_PROMPT_VARIANTS: dict[str, str] = {
    # Prompt variants are versioned, immutable test assets (paper, Sec. V-C).
    # Never edit an existing variant in place -- add a new key instead.
    "v1_plain": (
        "You are comparing two responses to the same instruction.\n\n"
        "Instruction:\n{instruction}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        "Which response is better? Answer with exactly one letter: A or B."
    ),
    "v2_rubric": (
        "Evaluate the two candidate responses below against the instruction.\n"
        "Criteria: correctness, completeness, clarity. Ignore length unless "
        "extra length adds substance.\n\n"
        "### Instruction\n{instruction}\n\n"
        "### Candidate A\n{response_a}\n\n"
        "### Candidate B\n{response_b}\n\n"
        "Reply with a single character, A or B, naming the better candidate."
    ),
    "v3_terse": (
        "Task: {instruction}\n"
        "A: {response_a}\n"
        "B: {response_b}\n"
        "Better answer (A/B)?"
    ),
    "v4_cot": (
        "You will judge two responses to an instruction. First think briefly "
        "about the strengths and weaknesses of each, then give a verdict.\n\n"
        "Instruction: {instruction}\n\n"
        "Response A: {response_a}\n\n"
        "Response B: {response_b}\n\n"
        "End your reply with the line 'VERDICT: A' or 'VERDICT: B'."
    ),
    "v5_role": (
        "Act as a strict senior reviewer. Two colleagues answered the same "
        "request; you must pick the stronger answer.\n\n"
        "Request: {instruction}\n\n"
        "Colleague A wrote:\n{response_a}\n\n"
        "Colleague B wrote:\n{response_b}\n\n"
        "Final decision, one letter only: A or B."
    ),
}


@dataclass
class Verdict:
    """Result of a single pairwise judgement."""

    winner: str  # "A" or "B"
    raw_output: str
    prompt_variant: str


class Judge(ABC):
    """Abstract pairwise judge."""

    #: Model-family tag used by the self-preference check
    #: (e.g. "claude", "gpt", "mock").
    family: str = "unknown"

    @abstractmethod
    def compare(
        self,
        instruction: str,
        response_a: str,
        response_b: str,
        prompt_variant: str = "v1_plain",
    ) -> Verdict:
        """Return which response the judge prefers."""

    # ------------------------------------------------------------------
    def _render(self, variant: str, instruction: str, a: str, b: str) -> str:
        template = PAIRWISE_PROMPT_VARIANTS[variant]
        return template.format(instruction=instruction, response_a=a, response_b=b)

    @staticmethod
    def _parse_verdict(text: str) -> str:
        """Extract 'A' or 'B' from a judge reply, tolerating chatter."""
        m = re.search(r"VERDICT:\s*([AB])", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # last standalone A/B token wins (handles CoT-style replies)
        tokens = re.findall(r"\b([AB])\b", text.upper())
        if tokens:
            return tokens[-1]
        raise ValueError(f"Could not parse verdict from judge output: {text!r}")


# ----------------------------------------------------------------------
# Real judges
# ----------------------------------------------------------------------
class AnthropicJudge(Judge):
    """Judge backed by the Anthropic Messages API.

    Requires ``pip install anthropic`` and the ``ANTHROPIC_API_KEY``
    environment variable. Model is configurable via ``CTE_ANTHROPIC_MODEL``;
    check https://docs.claude.com/en/api/overview for current model names.
    """

    family = "claude"

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        import anthropic  # lazy import so MockJudge users need no SDK

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get(
            "CTE_ANTHROPIC_MODEL", "claude-sonnet-4-5"
        )
        self.temperature = temperature

    def compare(self, instruction, response_a, response_b, prompt_variant="v1_plain"):
        prompt = self._render(prompt_variant, instruction, response_a, response_b)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Verdict(self._parse_verdict(text), text, prompt_variant)


class OpenAIJudge(Judge):
    """Judge backed by the OpenAI Chat Completions API.

    Requires ``pip install openai`` and ``OPENAI_API_KEY``. Model is
    configurable via ``CTE_OPENAI_MODEL``.
    """

    family = "gpt"

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.environ.get("CTE_OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def compare(self, instruction, response_a, response_b, prompt_variant="v1_plain"):
        prompt = self._render(prompt_variant, instruction, response_a, response_b)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return Verdict(self._parse_verdict(text), text, prompt_variant)


# ----------------------------------------------------------------------
# Mock judge with injectable biases
# ----------------------------------------------------------------------
@dataclass
class MockJudge(Judge):
    """Deterministic offline judge with configurable systematic biases.

    This is the ``b`` term of equation (1) in the paper, made explicit:

    * ``position_bias``     – probability mass shifted toward response A.
    * ``verbosity_bias``    – probability mass shifted toward the longer response.
    * ``self_preference``   – probability mass shifted toward responses tagged
      with ``[mock]`` (simulating same-family outputs).
    * ``noise``             – prompt-variant-dependent random flip rate,
      i.e. the epsilon_i term.

    Determinism: verdicts are derived from a SHA-256 hash of the inputs, so
    the same call always returns the same verdict (reproducible tests), while
    different items/variants decorrelate like pseudo-random draws.
    """

    position_bias: float = 0.0
    verbosity_bias: float = 0.0
    self_preference: float = 0.0
    noise: float = 0.1
    family: str = "mock"

    def _u(self, *parts: str) -> float:
        """Deterministic uniform(0,1) from input strings."""
        h = hashlib.sha256("||".join(parts).encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def compare(self, instruction, response_a, response_b, prompt_variant="v1_plain"):
        # Base preference: whichever answer is "objectively" better.
        # For calibration pairs the two answers are of equal quality, so the
        # base preference is a coin flip -- exactly what Phase 1 assumes.
        score_a = 0.5

        score_a += self.position_bias                     # A is shown first
        # verbosity only triggers on a *material* length difference,
        # so tiny artefacts (e.g. the '[mock]' family tag) don't bleed
        # into the verbosity measurement
        if len(response_a) - len(response_b) > 30:
            score_a += self.verbosity_bias
        elif len(response_b) - len(response_a) > 30:
            score_a -= self.verbosity_bias
        if "[mock]" in response_a:
            score_a += self.self_preference
        if "[mock]" in response_b:
            score_a -= self.self_preference

        # prompt-specific noise (epsilon_i): deterministic per (item, variant)
        u = self._u(instruction, response_a, response_b, prompt_variant)
        jitter = (u - 0.5) * 2 * self.noise
        score_a += jitter

        winner = "A" if score_a >= 0.5 else "B"
        return Verdict(winner, f"[mock score_a={score_a:.3f}]", prompt_variant)


def get_judge(name: str, **kwargs) -> Judge:
    """Factory: ``mock`` | ``anthropic`` | ``openai``."""
    name = name.lower()
    if name == "mock":
        return MockJudge(**kwargs)
    if name == "anthropic":
        return AnthropicJudge(**kwargs)
    if name == "openai":
        return OpenAIJudge(**kwargs)
    raise ValueError(f"Unknown judge: {name!r}")
