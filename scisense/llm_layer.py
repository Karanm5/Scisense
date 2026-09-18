"""Gated LLM layer.

The gate runs first. If a GP result fails the gate, no API call is made.
If it passes, the prompt contains only numbers computed by the GP engine
(plus optional user-supplied variable descriptions), and the wording the
model is told to use is tied to a confidence tier. After generation, every
number in the reply is checked against the numbers in the prompt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .gp_engine import GPEngine, GPResult

DEFAULT_MODEL = "llama-3.3-70b-versatile"

TIERS = [(0.95, "definitive"), (0.85, "strong"), (0.75, "indicative")]

TIER_INSTRUCTIONS = {
    "definitive": "Use firm wording such as 'clearly shows' or 'the fit confirms'.",
    "strong": "Use wording such as 'strongly suggests' and acknowledge a small residual uncertainty.",
    "indicative": "Use hedged wording such as 'is consistent with' and state the remaining uncertainty explicitly.",
}

SYSTEM_PROMPT = """You summarise Gaussian Process regression results for a scientist.

Rules:
1. Match the certainty of your wording to the tier you are given. Never sound more certain than the tier allows.
2. Only use numbers that appear in the prompt. Do not invent values, constants or citations.
3. The analysis is a one-feature regression. Describe associations, not causes. Only name a physical mechanism if it appears in the variable descriptions, and present it as context, not as something this fit proves.
4. Write 3 to 4 sentences. The last sentence is a testable follow-up question."""


def get_certainty_level(confidence: float, threshold: float = 0.75) -> str:
    if confidence < threshold:
        return "insufficient"
    for cutoff, tier in TIERS:
        if confidence >= cutoff:
            return tier
    return "indicative"


@dataclass
class SciInsight:
    predictor: str
    confidence: float
    gate_passed: bool
    certainty_level: str
    insight: str
    llm_called: bool = False
    unsupported_numbers: list = field(default_factory=list)


def build_prompt(
    result: GPResult,
    certainty: str,
    dataset_name: str = "Dataset",
    n_rows: Optional[int] = None,
    variable_notes: Optional[dict] = None,
) -> str:
    notes = variable_notes or {}
    lines = [
        f"Dataset: {dataset_name}" + (f" ({n_rows} rows)" if n_rows else ""),
        f"Relationship: {result.predictor} -> {result.target}",
    ]
    for col in (result.predictor, result.target):
        if col in notes:
            lines.append(f"Description of {col} (user supplied): {notes[col]}")
    lines += [
        "",
        "GP results (held-out test split):",
        f"  Kernel: {result.kernel_name}",
        f"  R2 test: {result.r2_test:.4f}",
        f"  RMSE test: {result.rmse_test:.6f}",
        f"  95% interval coverage on test points: {result.coverage_95_test:.3f}",
        f"  Mean predictive standard deviation: {result.mean_uncertainty:.6f}",
        f"  Confidence score: {result.confidence:.4f}",
        "",
        f"Certainty tier: {certainty.upper()}",
        f"Wording rule: {TIER_INSTRUCTIONS[certainty]}",
        "",
        "Write the summary now.",
    ]
    return "\n".join(lines)


_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")


def find_unsupported_numbers(text: str, prompt: str) -> list[str]:
    """Numbers in the reply that do not appear in the prompt.

    Small integers (0 to 10) are ignored because they are usually counts
    or ordinal words. This is a cheap hallucination check, not a proof.
    """
    allowed = set()
    for m in _NUM.findall(prompt):
        allowed.add(m)
        try:
            v = float(m)
            for d in range(0, 7):
                allowed.add(f"{v:.{d}f}")
            allowed.add(f"{v * 100:.1f}")
            allowed.add(f"{v * 100:.0f}")
        except ValueError:
            pass
    flagged = []
    for m in _NUM.findall(text):
        try:
            v = float(m)
        except ValueError:
            continue
        if v.is_integer() and 0 <= v <= 10:
            continue
        if m not in allowed and m.rstrip("0").rstrip(".") not in allowed:
            flagged.append(m)
    return sorted(set(flagged))


def groq_completion(model: str = DEFAULT_MODEL) -> Optional[Callable[[str, str], str]]:
    """Return a completion function backed by Groq, or None if no key is set."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    from groq import Groq

    client = Groq(api_key=key)

    def complete(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    return complete


class LLMReasoningLayer:
    """Calls the LLM only for GP results that pass the gate."""

    def __init__(
        self,
        engine: GPEngine,
        complete_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.engine = engine
        self.complete_fn = complete_fn

    def generate_insight(
        self,
        result: GPResult,
        dataset_name: str = "Dataset",
        n_rows: Optional[int] = None,
        variable_notes: Optional[dict] = None,
    ) -> SciInsight:
        threshold = self.engine.confidence_threshold
        if not self.engine.gate_passes(result):
            return SciInsight(
                predictor=result.predictor,
                confidence=result.confidence,
                gate_passed=False,
                certainty_level="insufficient",
                insight=(
                    f"Blocked by the uncertainty gate: confidence "
                    f"{result.confidence:.3f} is below {threshold}. "
                    "The LLM was not called."
                ),
            )

        certainty = get_certainty_level(result.confidence, threshold)
        if self.complete_fn is None:
            return SciInsight(
                predictor=result.predictor,
                confidence=result.confidence,
                gate_passed=True,
                certainty_level=certainty,
                insight="Gate passed. No LLM configured (set GROQ_API_KEY to enable).",
            )

        prompt = build_prompt(result, certainty, dataset_name, n_rows, variable_notes)
        try:
            text = self.complete_fn(SYSTEM_PROMPT, prompt)
        except Exception as exc:  # network or API errors
            return SciInsight(
                predictor=result.predictor,
                confidence=result.confidence,
                gate_passed=True,
                certainty_level=certainty,
                insight=f"LLM call failed: {exc}",
                llm_called=True,
            )
        return SciInsight(
            predictor=result.predictor,
            confidence=result.confidence,
            gate_passed=True,
            certainty_level=certainty,
            insight=text,
            llm_called=True,
            unsupported_numbers=find_unsupported_numbers(text, prompt),
        )
