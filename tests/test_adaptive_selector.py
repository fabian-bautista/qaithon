"""Tests for :class:`qaithon.ir.adaptive_selector.AdaptiveBackendSelector`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qaithon.ir.adaptive_selector import AdaptiveBackendSelector, ObservedStats
from qaithon.ir.analyzer import analyze_model


def _toy_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 32, bias=True),
        nn.ReLU(),
        nn.Linear(32, 8, bias=False),
    )


# ---------------------------------------------------------------------------
# ObservedStats — running mean correctness
# ---------------------------------------------------------------------------
def test_observed_stats_starts_at_zero():
    s = ObservedStats()
    assert s.n_samples == 0
    assert s.mean_latency_us == 0.0
    assert s.mean_energy_pj == 0.0


def test_observed_stats_single_observation():
    s = ObservedStats()
    s.record(latency_us=100.0, energy_pj=5.0)
    assert s.n_samples == 1
    assert s.mean_latency_us == pytest.approx(100.0)
    assert s.mean_energy_pj == pytest.approx(5.0)


def test_observed_stats_running_mean_is_correct():
    s = ObservedStats()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in values:
        s.record(latency_us=v, energy_pj=v / 2)
    assert s.n_samples == 5
    assert s.mean_latency_us == pytest.approx(30.0)
    assert s.mean_energy_pj == pytest.approx(15.0)


def test_observed_stats_no_drift_on_large_sequence():
    """Welford-style update should not accumulate float drift."""
    s = ObservedStats()
    for _ in range(10_000):
        s.record(latency_us=42.0, energy_pj=7.0)
    # Mean of a constant sequence must equal the constant.
    assert s.mean_latency_us == pytest.approx(42.0, abs=1e-9)
    assert s.mean_energy_pj == pytest.approx(7.0, abs=1e-9)


# ---------------------------------------------------------------------------
# AdaptiveBackendSelector — construction and validation
# ---------------------------------------------------------------------------
def test_construct_default_works():
    sel = AdaptiveBackendSelector()
    assert sel._alpha == 0.3
    assert sel._warmup == 5


def test_construct_invalid_alpha_raises():
    with pytest.raises(ValueError):
        AdaptiveBackendSelector(alpha=1.5)
    with pytest.raises(ValueError):
        AdaptiveBackendSelector(alpha=-0.1)


def test_construct_invalid_warmup_raises():
    with pytest.raises(ValueError):
        AdaptiveBackendSelector(warmup_samples=0)
    with pytest.raises(ValueError):
        AdaptiveBackendSelector(warmup_samples=-3)


# ---------------------------------------------------------------------------
# AdaptiveBackendSelector — recording and lookup
# ---------------------------------------------------------------------------
def test_record_creates_entry_per_shape_and_backend():
    sel = AdaptiveBackendSelector()
    sel.record((16, 32), "mock", latency_us=10.0, energy_pj=1.0)
    sel.record((16, 32), "mock", latency_us=20.0, energy_pj=2.0)
    sel.record((32, 8), "mock", latency_us=99.0, energy_pj=9.0)
    sel.record((16, 32), "pennylane.sim", latency_us=500.0, energy_pj=50.0)

    s1 = sel._stats[((16, 32), "mock")]
    assert s1.n_samples == 2
    assert s1.mean_latency_us == pytest.approx(15.0)

    s2 = sel._stats[((32, 8), "mock")]
    assert s2.n_samples == 1
    assert s2.mean_energy_pj == pytest.approx(9.0)

    s3 = sel._stats[((16, 32), "pennylane.sim")]
    assert s3.n_samples == 1
    assert s3.mean_latency_us == pytest.approx(500.0)


def test_has_data_false_until_warmup():
    sel = AdaptiveBackendSelector(warmup_samples=3)
    assert sel.has_data() is False
    sel.record((4, 4), "mock", 1.0, 0.1)
    sel.record((4, 4), "mock", 1.0, 0.1)
    assert sel.has_data() is False
    sel.record((4, 4), "mock", 1.0, 0.1)
    assert sel.has_data() is True


def test_blend_uses_declared_during_warmup():
    sel = AdaptiveBackendSelector(alpha=0.3, warmup_samples=5)
    declared, observed = 100.0, 200.0
    # Below warmup → return declared, observation ignored.
    assert sel._blend(declared, observed, n_samples=4) == declared
    # At/above warmup → blended.
    blended = sel._blend(declared, observed, n_samples=5)
    assert blended == pytest.approx(0.3 * 100.0 + 0.7 * 200.0)


def test_blend_full_trust_when_alpha_zero():
    sel = AdaptiveBackendSelector(alpha=0.0, warmup_samples=1)
    blended = sel._blend(declared=100.0, observed=42.0, n_samples=5)
    assert blended == pytest.approx(42.0)


def test_blend_full_distrust_when_alpha_one():
    sel = AdaptiveBackendSelector(alpha=1.0, warmup_samples=1)
    blended = sel._blend(declared=100.0, observed=42.0, n_samples=5)
    assert blended == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# AdaptiveBackendSelector — selection integrates with planner
# ---------------------------------------------------------------------------
def test_select_returns_a_result_for_a_toy_model():
    sel = AdaptiveBackendSelector(candidate_names=("mock",))
    plan = analyze_model(_toy_model())
    result = sel.select(plan, objective="speed")
    # Every replaceable layer must have a chosen backend.
    assert set(result.per_layer.keys()) == {m.name for m in plan.matches}
    assert all(b.profile.name == "mock" for b in result.per_layer.values())


def test_select_then_record_then_select_again():
    """After observations, selection must still succeed (no crash)."""
    sel = AdaptiveBackendSelector(
        candidate_names=("mock",), warmup_samples=2, alpha=0.5
    )
    plan = analyze_model(_toy_model())

    first = sel.select(plan, objective="balanced")
    assert len(first.decisions) > 0

    # Feed back observations for one shape.
    sample_shape = (first.decisions[0].in_features, first.decisions[0].out_features)
    sel.record(sample_shape, "mock", latency_us=1.0, energy_pj=0.1)
    sel.record(sample_shape, "mock", latency_us=1.0, energy_pj=0.1)
    assert sel.has_data() is True

    second = sel.select(plan, objective="balanced")
    # The shape that has data should pick mock just like before — but the
    # important point is that the call is stable post-recording.
    assert second.decisions[0].backend == first.decisions[0].backend
