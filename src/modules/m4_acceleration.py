"""
Module 4 — Computational Acceleration Benchmark.

Compares pure Python scalar byte loops against NumPy-vectorized uint64 differencing
to measure throughput (MB/s), latency per page (µs), and speedup factors.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.workload.page_generator import Page, PAGE_SIZE


@dataclass
class BenchmarkResult:
    """Stores benchmarking metrics for a diff implementation."""
    name: str
    num_pages: int
    total_time_ms: float
    latency_per_page_us: float
    throughput_mb_s: float
    speedup: float = 1.0


class AccelerationBenchmark:
    """
    Computational Acceleration Benchmark (Module 4).
    """

    @staticmethod
    def diff_scalar_python(page_content: bytearray, base_content: bytearray) -> bytearray:
        """Pure Python scalar loop byte-by-byte XOR."""
        n = len(page_content)
        result = bytearray(n)
        for i in range(n):
            result[i] = page_content[i] ^ base_content[i]
        return result

    @staticmethod
    def diff_vectorized_numpy(page_content: bytearray, base_content: bytearray) -> bytearray:
        """NumPy uint64-vectorized XOR differencing (processes 8 bytes per SIMD operation)."""
        arr_p = np.frombuffer(page_content, dtype=np.uint64)
        arr_b = np.frombuffer(base_content, dtype=np.uint64)
        xor_arr = np.bitwise_xor(arr_p, arr_b)
        return bytearray(xor_arr.tobytes())

    def run_benchmark(
        self,
        pages: List[Page],
        num_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs scalar vs. vectorized diff benchmarks across page pairs.
        Returns:
            Dictionary with timing, throughput, and speedup metrics.
        """
        n_pages = len(pages)
        if n_pages < 2:
            raise ValueError("Need at least 2 pages for acceleration benchmark")

        base_content = pages[0].content
        target_contents = [p.content for p in pages[1:]]
        num_diffs = len(target_contents) * num_iterations

        total_bytes = num_diffs * PAGE_SIZE
        total_mb = total_bytes / (1024.0 * 1024.0)

        # 1. Warm-up & Correctness Check
        diff_scalar_check = self.diff_scalar_python(target_contents[0], base_content)
        diff_vector_check = self.diff_vectorized_numpy(target_contents[0], base_content)
        if diff_scalar_check != diff_vector_check:
            raise RuntimeError("Scalar and Vectorized diff outputs do not match byte-for-byte!")

        # 2. Benchmark Scalar Python Loop
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            for target in target_contents:
                self.diff_scalar_python(target, base_content)
        t_scalar_ms = (time.perf_counter() - t0) * 1000.0

        lat_scalar_us = (t_scalar_ms * 1000.0) / num_diffs
        tp_scalar_mb_s = total_mb / (t_scalar_ms / 1000.0)

        # 3. Benchmark NumPy Vectorized uint64
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            for target in target_contents:
                self.diff_vectorized_numpy(target, base_content)
        t_vector_ms = (time.perf_counter() - t0) * 1000.0

        lat_vector_us = (t_vector_ms * 1000.0) / num_diffs
        tp_vector_mb_s = total_mb / (t_vector_ms / 1000.0)

        speedup = t_scalar_ms / t_vector_ms if t_vector_ms > 0 else 1.0

        scalar_res = BenchmarkResult(
            name="Scalar Python Loop",
            num_pages=num_diffs,
            total_time_ms=t_scalar_ms,
            latency_per_page_us=lat_scalar_us,
            throughput_mb_s=tp_scalar_mb_s,
            speedup=1.0,
        )

        vector_res = BenchmarkResult(
            name="NumPy Vectorized uint64",
            num_pages=num_diffs,
            total_time_ms=t_vector_ms,
            latency_per_page_us=lat_vector_us,
            throughput_mb_s=tp_vector_mb_s,
            speedup=speedup,
        )

        return {
            "scalar": scalar_res,
            "vectorized": vector_res,
            "speedup_factor": speedup,
            "latency_reduction_pct": ((lat_scalar_us - lat_vector_us) / lat_scalar_us) * 100.0,
        }
