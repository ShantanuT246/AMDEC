"""
Unit tests for Module 4: Computational Acceleration Benchmark (src/modules/m4_acceleration.py).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import Page, SyntheticWorkloadGenerator, PAGE_SIZE
from src.modules.m4_acceleration import AccelerationBenchmark


def test_scalar_vs_vectorized_diff_fidelity():
    """Verify scalar Python and NumPy vectorized diff produce identical byte-for-byte outputs."""
    generator = SyntheticWorkloadGenerator(seed=42)
    benchmark = AccelerationBenchmark()

    base = generator._create_template_page_content("heap_struct")
    target = generator._mutate_page_content(base, mutation_rate=0.01)

    diff_scalar = benchmark.diff_scalar_python(target, base)
    diff_vector = benchmark.diff_vectorized_numpy(target, base)

    assert len(diff_scalar) == PAGE_SIZE
    assert len(diff_vector) == PAGE_SIZE
    assert diff_scalar == diff_vector


def test_acceleration_benchmark_execution():
    """Verify benchmark runs and measures significant speedup for vectorized uint64 diff."""
    generator = SyntheticWorkloadGenerator(seed=42)
    benchmark = AccelerationBenchmark()

    pages = generator.generate_workload(num_pages=50)
    results = benchmark.run_benchmark(pages, num_iterations=2)

    assert "scalar" in results
    assert "vectorized" in results
    assert results["speedup_factor"] >= 1.5, "Vectorized diff should achieve >=1.5x speedup"
    assert results["vectorized"].throughput_mb_s > results["scalar"].throughput_mb_s


if __name__ == "__main__":
    pytest.main(["-v", __file__])
