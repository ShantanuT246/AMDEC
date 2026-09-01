"""
AMDEC — Multi-Trial Statistical Confidence Analysis.

Runs multiple independent randomized trials (across different seeds) to compute:
- Mean Logical Memory Savings (%) and 95% Confidence Intervals (CI)
- Mean LSH Candidate Pruning Reduction (%)
- Mean Reconstruction Latency (µs) and P95/P99 latency bounds
- Mean Throughput (pages/sec)
"""

import sys
import os
import time
from typing import Dict, List
import numpy as np
import scipy.stats as stats

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager
from src.modules.m5_cow import SimulatedCOWHandler


def compute_ci(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Computes mean, standard error, and confidence interval bounds."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1) if n > 1 else 0.0
    return {
        "mean": float(m),
        "std": float(np.std(a, ddof=1)) if n > 1 else 0.0,
        "sem": float(se),
        "ci_lower": float(m - h),
        "ci_upper": float(m + h),
        "ci_half_width": float(h),
    }


def run_statistical_analysis(num_trials: int = 10, num_pages: int = 5000):
    print("=" * 80)
    print(f"  AMDEC STATISTICAL CONFIDENCE ANALYSIS ({num_trials} TRIALS, N = {num_pages:,} PAGES)")
    print("=" * 80)

    savings_list = []
    compression_ratios = []
    lsh_reductions = []
    latencies_mean = []
    latencies_p95 = []
    latencies_p99 = []
    throughputs_pps = []

    m1 = RegionSamplingEngine()
    m2 = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
    m3 = BaseDeltaStorageManager(max_delta_ratio=0.30)
    cow = SimulatedCOWHandler(delta_manager=m3, sampling_engine=m1)

    for trial in range(num_trials):
        seed = 42 + trial * 7
        t0 = time.perf_counter()

        gen = SyntheticWorkloadGenerator(seed=seed)
        pages = gen.generate_workload(num_pages=num_pages)
        sim_time = gen.simulate_write_traffic(pages, num_ticks=30, writes_per_tick=150, current_time=100.0)

        cands, _ = m1.filter_and_prioritize(pages, current_time=sim_time)
        clusters, lsh_stats = m2.cluster_pages([c.page for c in cands])
        compressed_clusters, m3_stats = m3.compress_workload(clusters, pages)

        cow.populate_from_compressed_clusters(compressed_clusters, pages)
        for p in pages[:500]:
            if p.page_id in cow.page_table:
                _ = cow.read_page(p.page_id)

        lat_stats = cow.get_latency_metrics()
        total_time_s = time.perf_counter() - t0

        savings_list.append(m3_stats["logical_memory_savings_pct"])
        compression_ratios.append(m3_stats["compression_ratio"])
        lsh_reductions.append(lsh_stats["candidate_reduction_pct"])
        latencies_mean.append(lat_stats["mean_us"])
        latencies_p95.append(lat_stats["p95_us"])
        latencies_p99.append(lat_stats["p99_us"])
        throughputs_pps.append(num_pages / total_time_s)

        print(f"  Trial {trial+1:2d}/{num_trials} [Seed={seed:3d}] | Savings: {m3_stats['logical_memory_savings_pct']:5.2f}% | Latency (Mean/P95): {lat_stats['mean_us']:.2f}/{lat_stats['p95_us']:.2f} µs")

    ci_savings = compute_ci(savings_list)
    ci_comp = compute_ci(compression_ratios)
    ci_lsh = compute_ci(lsh_reductions)
    ci_lat_mean = compute_ci(latencies_mean)
    ci_lat_p95 = compute_ci(latencies_p95)
    ci_lat_p99 = compute_ci(latencies_p99)
    ci_tp = compute_ci(throughputs_pps)

    print("\n" + "=" * 80)
    print("  AMDEC 95% STATISTICAL CONFIDENCE INTERVAL SUMMARY")
    print("=" * 80)
    print(f"  Metric                             | Mean ± Std.Dev        | 95% Confidence Interval")
    print("-" * 80)
    print(f"  Logical Memory Savings (%)         | {ci_savings['mean']:5.2f}% ± {ci_savings['std']:4.2f}%    | [{ci_savings['ci_lower']:5.2f}%, {ci_savings['ci_upper']:5.2f}%]")
    print(f"  Effective Compression Ratio        | {ci_comp['mean']:5.2f}x ± {ci_comp['std']:4.2f}x     | [{ci_comp['ci_lower']:5.2f}x, {ci_comp['ci_upper']:5.2f}x]")
    print(f"  LSH Pair Pruning Reduction (%)     | {ci_lsh['mean']:5.3f}% ± {ci_lsh['std']:4.3f}%   | [{ci_lsh['ci_lower']:5.3f}%, {ci_lsh['ci_upper']:5.3f}%]")
    print(f"  Reconstruction Latency (Mean, µs)  | {ci_lat_mean['mean']:5.2f} µs ± {ci_lat_mean['std']:4.2f} µs   | [{ci_lat_mean['ci_lower']:5.2f} µs, {ci_lat_mean['ci_upper']:5.2f} µs]")
    print(f"  Reconstruction Latency (P95, µs)   | {ci_lat_p95['mean']:5.2f} µs ± {ci_lat_p95['std']:4.2f} µs   | [{ci_lat_p95['ci_lower']:5.2f} µs, {ci_lat_p95['ci_upper']:5.2f} µs]")
    print(f"  Reconstruction Latency (P99, µs)   | {ci_lat_p99['mean']:5.2f} µs ± {ci_lat_p99['std']:4.2f} µs   | [{ci_lat_p99['ci_lower']:5.2f} µs, {ci_lat_p99['ci_upper']:5.2f} µs]")
    print(f"  Pipeline Throughput (pages/sec)    | {ci_tp['mean']:5.0f} ± {ci_tp['std']:4.0f} pps      | [{ci_tp['ci_lower']:5.0f}, {ci_tp['ci_upper']:5.0f}] pps")
    print("=" * 80)


if __name__ == "__main__":
    run_statistical_analysis()
