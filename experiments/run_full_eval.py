"""
AMDEC — Multi-Trial Research-Grade Evaluation Pipeline.

Executes independent multi-trial evaluations across seeds, collects raw per-trial records,
pools high-resolution latency measurements, computes Student's t 95% confidence intervals,
and exports structured raw CSV logs to results/raw/.
"""

import argparse
import csv
from datetime import datetime
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import AMDECConfig
from src.workload.page_generator import SyntheticWorkloadGenerator, PAGE_SIZE
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager
from src.modules.m4_acceleration import AccelerationBenchmark
from src.modules.m5_cow import SimulatedCOWHandler, PageState


def compute_metric_stats(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Calculates mean, std, median, min, max, and Student's t 95% CI."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(stats.sem(arr)) if n > 1 else 0.0
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, n - 1)) if n > 1 else 0.0
    ci_hw = float(t_crit * se) if n > 1 else 0.0

    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "ci_lower": mean_val - ci_hw,
        "ci_upper": mean_val + ci_hw,
        "ci_half_width": ci_hw,
    }


def execute_single_trial(
    trial_id: int,
    seed: int,
    config: AMDECConfig,
    num_pages: int,
) -> Tuple[Dict[str, Any], List[float]]:
    """Executes a single fully-deterministic trial of the 5-module AMDEC pipeline."""
    # 1. Workload Generation
    generator = SyntheticWorkloadGenerator(
        seed=seed,
        near_dup_ratio=config.workload.near_dup_ratio,
        exact_dup_ratio=config.workload.exact_dup_ratio,
        unique_ratio=config.workload.unique_ratio,
        num_regions=config.workload.num_regions,
        zipf_param=config.workload.zipf_param,
    )
    t0 = time.perf_counter()
    pages = generator.generate_workload(num_pages=num_pages, sim_start_time=config.workload.sim_start_time)
    t_gen_ms = (time.perf_counter() - t0) * 1000.0
    sim_end_time = generator.simulate_write_traffic(
        pages,
        num_ticks=config.workload.num_ticks,
        writes_per_tick=config.workload.writes_per_tick,
        current_time=config.workload.sim_start_time,
    )

    # 2. Module 1: Region-Aware Sampling
    m1_engine = RegionSamplingEngine(
        hot_write_rate_threshold=config.m1_sampling.hot_write_rate_threshold,
        cold_write_rate_threshold=config.m1_sampling.cold_write_rate_threshold,
        min_age_for_cold=config.m1_sampling.min_age_for_cold,
        cow_penalty_weight=config.m1_sampling.cow_penalty_weight,
    )
    t0 = time.perf_counter()
    candidates, m1_stats = m1_engine.filter_and_prioritize(pages, current_time=sim_end_time)
    t_m1_ms = (time.perf_counter() - t0) * 1000.0
    candidate_pages = [c.page for c in candidates]
    cand_reduction_pct = (m1_stats["hot_pages_skipped"] / num_pages) * 100.0

    # 3. Module 2: MinHash-LSH Clustering
    m2_engine = MinHashLSHClusteringEngine(
        block_size=config.m2_clustering.block_size,
        num_hashes=config.m2_clustering.num_hashes,
        num_bands=config.m2_clustering.num_bands,
        similarity_threshold=config.m2_clustering.similarity_threshold,
        seed=seed,
    )
    t0 = time.perf_counter()
    clusters, m2_stats = m2_engine.cluster_pages(candidate_pages, verify_similarity=True)
    t_m2_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Module 3: Base-Delta Storage Manager
    m3_manager = BaseDeltaStorageManager(
        max_delta_ratio=config.m3_delta.max_delta_ratio,
        base_selection_strategy=config.m3_delta.base_selection_strategy,
        seed=seed,
    )
    t0 = time.perf_counter()
    compressed_clusters, m3_metrics = m3_manager.compress_workload(clusters, pages)
    t_m3_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Module 4: SIMD Acceleration Benchmark
    accel_bench = AccelerationBenchmark()
    accel_res = accel_bench.run_benchmark(pages[:200], num_iterations=3)

    # 6. Module 5: Simulated COW Reconstruction
    cow_handler = SimulatedCOWHandler(delta_manager=m3_manager, sampling_engine=m1_engine)
    cow_handler.populate_from_compressed_clusters(compressed_clusters, pages)

    delta_page_ids = [pid for pid, entry in cow_handler.page_table.items() if entry.state == PageState.DELTA_PAGE]
    sample_reads = delta_page_ids[:min(500, len(delta_page_ids))]

    # Disable GC during latency benchmarking to prevent interpreter scheduler jitter
    gc.disable()
    try:
        for pid in sample_reads:
            _ = cow_handler.read_page(pid)
    finally:
        gc.enable()

    sample_writes = delta_page_ids[:min(200, len(delta_page_ids))]
    curr_t = sim_end_time + 10.0
    for pid in sample_writes:
        curr_t += 0.1
        _ = cow_handler.handle_write_fault(pid, offset=16, data=b"WRITE_EVENT", current_time=curr_t)

    raw_latencies = cow_handler.get_raw_latency_samples()
    lat_stats = cow_handler.get_latency_metrics()
    total_pipeline_time_ms = t_gen_ms + t_m1_ms + t_m2_ms + t_m3_ms

    trial_record = {
        "trial_id": trial_id,
        "seed": seed,
        "num_pages": num_pages,
        "baseline_memory_mb": m3_metrics["baseline_storage_bytes"] / (1024.0 * 1024.0),
        "amdec_memory_mb": m3_metrics["amdec_storage_bytes"] / (1024.0 * 1024.0),
        "memory_savings_percent": m3_metrics["logical_memory_savings_pct"],
        "compression_ratio": m3_metrics["compression_ratio"],
        "hot_pages_skipped": m1_stats["hot_pages_skipped"],
        "m1_reduction_pct": cand_reduction_pct,
        "candidate_pairs": m2_stats["candidate_pairs_generated"],
        "pair_reduction_percent": m2_stats["candidate_reduction_pct"],
        "num_clusters": len(clusters),
        "clustered_pages": m2_stats["clustered_pages_count"],
        "base_pages": m3_metrics["total_base_pages"],
        "delta_pages": m3_metrics["total_delta_pages"],
        "unmerged_pages": m3_metrics["total_unmerged_cluster_pages"],
        "mean_delta_bytes": m3_metrics["avg_delta_bytes"],
        "p95_delta_bytes": m3_metrics["p95_delta_bytes"],
        "mean_reconstruction_us": lat_stats["mean_us"],
        "median_reconstruction_us": lat_stats["median_us"],
        "p95_reconstruction_us": lat_stats["p95_us"],
        "p99_reconstruction_us": lat_stats["p99_us"],
        "simd_speedup": accel_res["speedup_factor"],
        "simd_throughput_mb_s": accel_res["vectorized"].throughput_mb_s,
        "pipeline_time_ms": total_pipeline_time_ms,
        "pipeline_throughput_pages_s": num_pages / (total_pipeline_time_ms / 1000.0) if total_pipeline_time_ms > 0 else 0.0,
    }

    return trial_record, raw_latencies


def run_full_evaluation(
    num_pages: int = 10000,
    num_trials: int = 10,
    initial_seed: int = 42,
    config_path: Optional[str] = None,
    output_csv: bool = True,
):
    print("=" * 85)
    print(f"  AMDEC RESEARCH-GRADE MULTI-TRIAL EVALUATION ({num_trials} Trials, N = {num_pages:,} Pages)")
    print("=" * 85)

    if config_path and os.path.exists(config_path):
        config = AMDECConfig.load_json(config_path)
        print(f"  Loaded Configuration: {config_path}")
    else:
        config = AMDECConfig()

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    trial_records: List[Dict[str, Any]] = []
    all_raw_latencies: List[float] = []

    for trial_idx in range(num_trials):
        seed = initial_seed + trial_idx
        t_start = time.perf_counter()
        record, raw_lats = execute_single_trial(
            trial_id=trial_idx + 1,
            seed=seed,
            config=config,
            num_pages=num_pages,
        )
        t_elapsed = time.perf_counter() - t_start
        trial_records.append(record)
        all_raw_latencies.extend(raw_lats)

        print(f"  [Trial {trial_idx+1:2d}/{num_trials}] Seed {seed:4d} | Savings: {record['memory_savings_percent']:5.2f}% | "
              f"Clusters: {record['num_clusters']:3d} | Recon Latency: {record['mean_reconstruction_us']:.2f} µs | Time: {t_elapsed:.2f}s")

    # Aggregate Global Latency Distribution across all trials
    pooled_lats = np.array(all_raw_latencies)
    pooled_p50 = float(np.median(pooled_lats)) if len(pooled_lats) > 0 else 0.0
    pooled_p95 = float(np.percentile(pooled_lats, 95)) if len(pooled_lats) > 0 else 0.0
    pooled_p99 = float(np.percentile(pooled_lats, 99)) if len(pooled_lats) > 0 else 0.0

    # Compute Statistics across Trials
    stat_savings = compute_metric_stats([r["memory_savings_percent"] for r in trial_records])
    stat_comp = compute_metric_stats([r["compression_ratio"] for r in trial_records])
    stat_footprint = compute_metric_stats([r["amdec_memory_mb"] for r in trial_records])
    stat_clusters = compute_metric_stats([r["num_clusters"] for r in trial_records])
    stat_delta_bytes = compute_metric_stats([r["mean_delta_bytes"] for r in trial_records])
    stat_lsh_red = compute_metric_stats([r["pair_reduction_percent"] for r in trial_records])
    stat_m1_red = compute_metric_stats([r["m1_reduction_pct"] for r in trial_records])
    stat_recon_mean = compute_metric_stats([r["mean_reconstruction_us"] for r in trial_records])
    stat_speedup = compute_metric_stats([r["simd_speedup"] for r in trial_records])
    stat_runtime = compute_metric_stats([r["pipeline_time_ms"] for r in trial_records])
    stat_tp = compute_metric_stats([r["pipeline_throughput_pages_s"] for r in trial_records])

    # Print Official Publication Table
    print("\n" + "=" * 95)
    print(f"  AMDEC OFFICIAL MULTI-TRIAL EVALUATION REPORT (N = {num_pages:,} Pages, {num_trials} Trials)")
    print("=" * 95)
    print(f"  Metric                             | Mean ± Std.Dev        | 95% Confidence Interval | Median [Min, Max]")
    print("-" * 95)
    print(f"  Baseline Footprint (MB)            | {trial_records[0]['baseline_memory_mb']:6.2f} MB (Fixed)     | [Fixed Baseline]        | {trial_records[0]['baseline_memory_mb']:6.2f} MB")
    print(f"  AMDEC Logical Storage (MB)         | {stat_footprint['mean']:5.2f} ± {stat_footprint['std']:4.2f} MB    | [{stat_footprint['ci_lower']:5.2f}, {stat_footprint['ci_upper']:5.2f}] MB     | {stat_footprint['median']:5.2f} [{stat_footprint['min']:5.2f}, {stat_footprint['max']:5.2f}]")
    print(f"  Logical Memory Savings (%)         | {stat_savings['mean']:5.2f}% ± {stat_savings['std']:4.2f}%    | [{stat_savings['ci_lower']:5.2f}%, {stat_savings['ci_upper']:5.2f}%]    | {stat_savings['median']:5.2f}% [{stat_savings['min']:5.2f}%, {stat_savings['max']:5.2f}%]")
    print(f"  Effective Compression Ratio        | {stat_comp['mean']:5.2f}x ± {stat_comp['std']:4.2f}x     | [{stat_comp['ci_lower']:5.2f}x, {stat_comp['ci_upper']:5.2f}x]    | {stat_comp['median']:5.2f}x [{stat_comp['min']:5.2f}x, {stat_comp['max']:5.2f}x]")
    print("-" * 95)
    print(f"  M1 Volatile Scan Reduction (%)     | {stat_m1_red['mean']:5.2f}% ± {stat_m1_red['std']:4.2f}%    | [{stat_m1_red['ci_lower']:5.2f}%, {stat_m1_red['ci_upper']:5.2f}%]    | {stat_m1_red['median']:5.2f}% [{stat_m1_red['min']:5.2f}%, {stat_m1_red['max']:5.2f}%]")
    print(f"  M2 LSH Pair Search Reduction (%)   | {stat_lsh_red['mean']:5.3f}% ± {stat_lsh_red['std']:4.3f}%   | [{stat_lsh_red['ci_lower']:5.3f}%, {stat_lsh_red['ci_upper']:5.3f}%]   | {stat_lsh_red['median']:5.3f}% [{stat_lsh_red['min']:5.3f}%, {stat_lsh_red['max']:5.3f}%]")
    print(f"  Similarity Clusters Formed         | {stat_clusters['mean']:5.1f} ± {stat_clusters['std']:4.1f}       | [{stat_clusters['ci_lower']:5.1f}, {stat_clusters['ci_upper']:5.1f}]       | {stat_clusters['median']:5.1f} [{stat_clusters['min']:.0f}, {stat_clusters['max']:.0f}]")
    print(f"  Mean Delta Footprint (w/ meta, B)  | {stat_delta_bytes['mean']:5.1f} ± {stat_delta_bytes['std']:4.1f} B     | [{stat_delta_bytes['ci_lower']:5.1f}, {stat_delta_bytes['ci_upper']:5.1f}] B     | {stat_delta_bytes['median']:5.1f} [{stat_delta_bytes['min']:5.1f}, {stat_delta_bytes['max']:5.1f}]")
    print("-" * 95)
    print(f"  Reconstruction Latency (Mean, µs)  | {stat_recon_mean['mean']:5.2f} ± {stat_recon_mean['std']:4.2f} µs   | [{stat_recon_mean['ci_lower']:5.2f}, {stat_recon_mean['ci_upper']:5.2f}] µs   | {stat_recon_mean['median']:5.2f} [{stat_recon_mean['min']:5.2f}, {stat_recon_mean['max']:5.2f}]")
    print(f"  Pooled Latency (P50 / P95 / P99)   | {pooled_p50:5.2f} / {pooled_p95:5.2f} / {pooled_p99:5.2f} µs (Empirical Aggregate across {len(pooled_lats):,} accesses)")
    print(f"  M4 SIMD Vector Speedup Factor      | {stat_speedup['mean']:5.1f}x ± {stat_speedup['std']:4.1f}x     | [{stat_speedup['ci_lower']:5.1f}x, {stat_speedup['ci_upper']:5.1f}x]    | {stat_speedup['median']:5.1f}x [{stat_speedup['min']:5.1f}x, {stat_speedup['max']:5.1f}x]")
    print(f"  Total Pipeline Time (ms)           | {stat_runtime['mean']:5.1f} ± {stat_runtime['std']:4.1f} ms    | [{stat_runtime['ci_lower']:5.1f}, {stat_runtime['ci_upper']:5.1f}] ms    | {stat_runtime['median']:5.1f} [{stat_runtime['min']:5.1f}, {stat_runtime['max']:5.1f}]")
    print(f"  Pipeline Throughput (pages/sec)    | {stat_tp['mean']:5.0f} ± {stat_tp['std']:4.0f} pps      | [{stat_tp['ci_lower']:5.0f}, {stat_tp['ci_upper']:5.0f}] pps      | {stat_tp['median']:5.0f} [{stat_tp['min']:5.0f}, {stat_tp['max']:5.0f}]")
    print("=" * 95)

    # Export structured CSV outputs
    if output_csv and trial_records:
        csv_canonical = os.path.join(results_dir, "primary_trials.csv")
        csv_timestamped = os.path.join(results_dir, f"primary_trials_{timestamp_str}.csv")
        fieldnames = list(trial_records[0].keys())

        for p in [csv_canonical, csv_timestamped]:
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trial_records)

        # Save pooled latency samples for figure plotting
        lat_csv_path = os.path.join(results_dir, "primary_latency_samples.csv")
        with open(lat_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["latency_us"])
            for l in all_raw_latencies:
                writer.writerow([l])

        print(f"\n[Raw Trial Data Exported]   : {csv_canonical}")
        print(f"[Timestamped Log Exported]  : {csv_timestamped}")
        print(f"[Pooled Latencies Exported] : {lat_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Multi-Trial Research Evaluation Pipeline")
    parser.add_argument("pages", type=int, nargs="?", default=10000, help="Number of workload pages (default: 10000)")
    parser.add_argument("--trials", type=int, default=10, help="Number of independent trials (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed (default: 42)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    args = parser.parse_args()

    run_full_evaluation(
        num_pages=args.pages,
        num_trials=args.trials,
        initial_seed=args.seed,
        config_path=args.config,
    )
