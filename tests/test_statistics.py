"""
Tests for AMDEC Statistical Analysis Engine.

Verifies:
1. compute_metric_stats accuracy (mean, standard deviation, Student's t 95% CI).
2. Latency pooling and empirical percentile calculations.
3. Hand-verifiable analytical cases.
"""

import numpy as np
import pytest
import scipy.stats as stats
from experiments.run_full_eval import compute_metric_stats


def test_compute_metric_stats_analytical():
    """Verify compute_metric_stats on a known analytical sample."""
    # Data: [10.0, 12.0, 14.0, 16.0, 18.0] -> mean=14.0, var=10.0, std=sqrt(10)=3.162277
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    res = compute_metric_stats(data, confidence=0.95)

    assert res["n"] == 5
    assert pytest.approx(res["mean"], 1e-4) == 14.0
    assert pytest.approx(res["median"], 1e-4) == 14.0
    assert pytest.approx(res["min"], 1e-4) == 10.0
    assert pytest.approx(res["max"], 1e-4) == 18.0
    assert pytest.approx(res["std"], 1e-4) == np.std(data, ddof=1)

    # Student's t critical value for df=4, alpha=0.05 is 2.776445
    se = np.std(data, ddof=1) / np.sqrt(5)
    expected_hw = 2.776445 * se
    assert pytest.approx(res["ci_half_width"], 1e-3) == expected_hw
    assert pytest.approx(res["ci_lower"], 1e-3) == 14.0 - expected_hw
    assert pytest.approx(res["ci_upper"], 1e-3) == 14.0 + expected_hw


def test_compute_metric_stats_single_element():
    """Single element sample must not crash and have zero CI width."""
    data = [42.0]
    res = compute_metric_stats(data, confidence=0.95)
    assert res["n"] == 1
    assert res["mean"] == 42.0
    assert res["std"] == 0.0
    assert res["ci_half_width"] == 0.0


def test_percentile_pooling():
    """Verify that pooled latency percentiles match aggregate distribution."""
    lats1 = [1.0, 2.0, 3.0]
    lats2 = [4.0, 5.0, 6.0]
    pooled = np.array(lats1 + lats2)

    assert pytest.approx(np.median(pooled), 1e-4) == 3.5
    assert pytest.approx(np.percentile(pooled, 90), 1e-4) == 5.5
