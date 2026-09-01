"""
AMDEC — Isolated M1 Sampling Benchmark (Fix #2).

Measures the computational cost and filtering efficacy of Module 1 independently
from the full pipeline, using high-resolution monotonic timing with warm-up iterations.

Reports:
- Number and percentage of HOT pages filtered
- Isolated M1 filtering throughput (pages/sec)
- Equivalent uniform-scan baseline throughput
- Mean, median, SD, and 95% CI of per-iteration timing
"""

import argparse
import csv
import gc
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine


def _uniform_scan_baseline(pages, current_time: float):
    """
    Simulates a baseline uniform scan that processes ALL pages without filtering.
    Performs the same classification work M1 does but does not skip HOT pages.
    This ensures the timing comparison is fair (same computational work per page).
    """
    engine = RegionSamplingEngine()
    # Classify every page (same as M1 internally) but never skip
    results = []
    for page in pages:
        tier, score = engine.classify_page(page, current_time)
        results.append((page.page_id, tier, score))
    return results


def run_m1_benchmark(
    num_pages: int = 10000,
    num_trials: int = 5,
    initial_seed: int = 42,
    warmup_iterations: int = 3,
    measurement_iterations: int = 20,
):
    print("=" * 95)
    print(f"  AMDEC ISOLATED M1 SAMPLING BENCHMARK")
    print(f"  N = {num_pages:,} pages | {num_trials} trials × {measurement_iterations} iterations "
          f"(+ {warmup_iterations} warmup) per trial")
    print("=" * 95)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(raw_dir, exist_ok=True)

    trial_records: List[Dict[str, Any]] = []

    for t_idx in range(num_trials):
        seed = initial_seed + t_idx
        gen = SyntheticWorkloadGenerator(seed=seed, zipf_param=1.3)
        pages = gen.generate_workload(num_pages=num_pages)
        sim_end_time = gen.simulate_write_traffic(
            pages, num_ticks=40, writes_per_tick=250, current_time=100.0
        )

        # --- Warm-up ---
        for _ in range(warmup_iterations):
            m1_warmup = RegionSamplingEngine()
            _ = m1_warmup.filter_and_prioritize(pages, current_time=sim_end_time)
            _ = _uniform_scan_baseline(pages, sim_end_time)

        # --- Measurement: M1 filtering ---
        m1_times_ns: List[int] = []
        m1_skipped = 0
        m1_total = 0
        m1_stats_last = {}

        gc.disable()
        try:
            for _ in range(measurement_iterations):
                m1_eng = RegionSamplingEngine()
                t0 = time.perf_counter_ns()
                candidates, m1_st = m1_eng.filter_and_prioritize(pages, current_time=sim_end_time)
                t1 = time.perf_counter_ns()
                m1_times_ns.append(t1 - t0)
                m1_stats_last = m1_st
        finally:
            gc.enable()

        m1_skipped = m1_stats_last["hot_pages_skipped"]
        m1_total = m1_stats_last["total_pages"]
        m1_passed = m1_total - m1_skipped

        # --- Measurement: Uniform scan baseline ---
        baseline_times_ns: List[int] = []

        gc.disable()
        try:
            for _ in range(measurement_iterations):
                t0 = time.perf_counter_ns()
                _ = _uniform_scan_baseline(pages, sim_end_time)
                t1 = time.perf_counter_ns()
                baseline_times_ns.append(t1 - t0)
        finally:
            gc.enable()

        # Convert to microseconds
        m1_us = np.array(m1_times_ns) / 1000.0
        bl_us = np.array(baseline_times_ns) / 1000.0

        m1_mean = float(np.mean(m1_us))
        m1_median = float(np.median(m1_us))
        m1_std = float(np.std(m1_us, ddof=1))
        m1_se = float(stats.sem(m1_us))
        m1_ci_hw = float(stats.t.ppf(0.975, len(m1_us) - 1) * m1_se) if len(m1_us) > 1 else 0.0

        bl_mean = float(np.mean(bl_us))
        bl_median = float(np.median(bl_us))
        bl_std = float(np.std(bl_us, ddof=1))
        bl_se = float(stats.sem(bl_us))
        bl_ci_hw = float(stats.t.ppf(0.975, len(bl_us) - 1) * bl_se) if len(bl_us) > 1 else 0.0

        # Throughput
        m1_pps = num_pages / (m1_mean / 1e6) if m1_mean > 0 else 0.0
        bl_pps = num_pages / (bl_mean / 1e6) if bl_mean > 0 else 0.0

        rec = {
            "trial_id": t_idx + 1,
            "seed": seed,
            "num_pages": num_pages,
            "hot_pages_skipped": m1_skipped,
            "pages_passed_to_m2": m1_passed,
            "skip_pct": (m1_skipped / m1_total) * 100.0,
            "m1_mean_us": m1_mean,
            "m1_median_us": m1_median,
            "m1_std_us": m1_std,
            "m1_ci_lower_us": m1_mean - m1_ci_hw,
            "m1_ci_upper_us": m1_mean + m1_ci_hw,
            "m1_throughput_pps": m1_pps,
            "baseline_mean_us": bl_mean,
            "baseline_median_us": bl_median,
            "baseline_std_us": bl_std,
            "baseline_ci_lower_us": bl_mean - bl_ci_hw,
            "baseline_ci_upper_us": bl_mean + bl_ci_hw,
            "baseline_throughput_pps": bl_pps,
            "speedup_mean": bl_mean / m1_mean if m1_mean > 0 else 0.0,
        }
        trial_records.append(rec)

        print(f"  [Trial {t_idx+1}/{num_trials}] Seed {seed} | Skipped: {m1_skipped:4d} ({rec['skip_pct']:4.1f}%) | "
              f"M1: {m1_mean/1000:.2f} ms ± {m1_std/1000:.2f} ms | "
              f"Baseline: {bl_mean/1000:.2f} ms ± {bl_std/1000:.2f} ms | "
              f"Speedup: {rec['speedup_mean']:.3f}x")

    # Export
    csv_path = os.path.join(raw_dir, "benchmark_m1_isolated.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_records[0].keys()))
        writer.writeheader()
        writer.writerows(trial_records)

    # Aggregate summary
    skip_pcts = [r["skip_pct"] for r in trial_records]
    speedups = [r["speedup_mean"] for r in trial_records]
    m1_means = [r["m1_mean_us"] for r in trial_records]
    bl_means = [r["baseline_mean_us"] for r in trial_records]

    print("\n" + "=" * 95)
    print("  ISOLATED M1 BENCHMARK SUMMARY")
    print("-" * 95)
    print(f"  Pages Filtered (HOT skipped): {np.mean(skip_pcts):.1f}% ± {np.std(skip_pcts, ddof=1):.1f}%")
    print(f"  M1 Filtering Time:  {np.mean(m1_means)/1000:.2f} ± {np.std(m1_means, ddof=1)/1000:.2f} ms "
          f"({np.mean([r['m1_throughput_pps'] for r in trial_records]):,.0f} pages/sec)")
    print(f"  Baseline Scan Time: {np.mean(bl_means)/1000:.2f} ± {np.std(bl_means, ddof=1)/1000:.2f} ms "
          f"({np.mean([r['baseline_throughput_pps'] for r in trial_records]):,.0f} pages/sec)")
    print(f"  Isolated Speedup:   {np.mean(speedups):.3f}x ± {np.std(speedups, ddof=1):.3f}x")
    print(f"\n  NOTE: M1's primary value is FILTERING (reducing input to M2), not raw scan speed.")
    print(f"        {np.mean(skip_pcts):.1f}% of pages are classified HOT and excluded from O(N²)-sensitive M2.")
    print(f"  [Raw CSV Exported]: {csv_path}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Isolated M1 Sampling Benchmark")
    parser.add_argument("--pages", type=int, default=10000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run_m1_benchmark(
        num_pages=args.pages,
        num_trials=args.trials,
        initial_seed=args.seed,
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
    )
