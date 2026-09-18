"""Tests for the GP engine, the gate and the gated LLM layer.

No network access or API key is needed: the LLM is replaced by a stub.
"""

import numpy as np
import pandas as pd
import pytest

from scisense import GPEngine, LLMReasoningLayer, get_certainty_level, run_scisense
from scisense.llm_layer import find_unsupported_numbers
from scisense.pipeline import NOISE_CONTROL


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 300
    x_det = rng.uniform(0.07, 0.5, n)
    y = 2.26 * np.exp(-1.0 / x_det)  # deterministic BCS-like curve
    x_noisy = x_det + rng.normal(0, 0.08, n)  # informative but noisy
    x_null = rng.normal(size=n)  # unrelated to y
    return pd.DataFrame({"x_det": x_det, "x_noisy": x_noisy, "x_null": x_null, "y": y})


def test_deterministic_relation_passes_gate(data):
    eng = GPEngine()
    r = eng.fit_and_analyse(data.x_det.values, data.y.values, "x_det", "y")
    assert r.r2_test > 0.99
    assert eng.gate_passes(r)


def test_unrelated_feature_is_blocked(data):
    eng = GPEngine()
    r = eng.fit_and_analyse(data.x_null.values, data.y.values, "x_null", "y")
    assert r.r2_test < 0.2
    assert not eng.gate_passes(r)


def test_confidence_bounded_and_coverage_reasonable(data):
    eng = GPEngine()
    r = eng.fit_and_analyse(data.x_noisy.values, data.y.values, "x_noisy", "y")
    assert 0.0 <= r.confidence <= 1.0
    assert 0.8 <= r.coverage_95_test <= 1.0


def test_blocked_result_never_calls_llm(data):
    calls = []

    def stub(system, user):
        calls.append(user)
        return "stub"

    eng = GPEngine()
    r = eng.fit_and_analyse(data.x_null.values, data.y.values, "x_null", "y")
    ins = LLMReasoningLayer(eng, complete_fn=stub).generate_insight(r)
    assert not ins.gate_passed and not ins.llm_called and calls == []


def test_certainty_tiers():
    assert get_certainty_level(0.99) == "definitive"
    assert get_certainty_level(0.90) == "strong"
    assert get_certainty_level(0.80) == "indicative"
    assert get_certainty_level(0.50) == "insufficient"


def test_unsupported_number_check():
    prompt = "R2 test: 0.9172\nConfidence score: 0.8730"
    assert find_unsupported_numbers("R2 of 0.917 and confidence 0.873", prompt) == []
    assert find_unsupported_numbers("a 93x gap and 26.1 percent", prompt) == ["26.1", "93"]


def test_pipeline_with_noise_control(data):
    out = run_scisense(data, target_col="y", add_noise_control=True, complete_fn=lambda s, u: "ok")
    assert NOISE_CONTROL in out.gp_results
    assert not out.engine.gate_passes(out.gp_results[NOISE_CONTROL])
    assert out.insights["x_det"].llm_called


def test_bad_target_raises(data):
    with pytest.raises(ValueError):
        run_scisense(data, target_col="not_a_column")
