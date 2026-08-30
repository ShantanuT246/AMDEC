"""
AMDEC — Single-Command End-to-End Evaluation Pipeline.

Runs full evaluation across all 5 modules (M1 -> M2 -> M3 -> M4 -> M5), computes all
primary metrics (storage savings, compression ratio, candidate reduction, cluster purity,
reconstruction latency, throughput, acceleration), and logs timestamped CSV results to results/raw/.
"""

import sys
import os
import time
import csv
from datetime import datetime
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager
from src.modules.m4_acceleration import AccelerationBenchmark
from src.modules.m5_cow import SimulatedCOWHandler, PageState


def run_full_evaluation(num_pages: int = 10000, output_csv: bool = True):
    print("=" * 80)
    print(f"  AMDEC FULL END-TO-END SYSTEM EVALUATION (N = {num_pages:,} Pages)")
    print("=" * 80)

    # 1. Workload Generation
    print("\n[1/5] Generating Synthetic Anonymous Memory Workload...")
    generator = SyntheticWorkloadGenerator(
        seed=42, near_dup_ratio=0.80, exact_dup_ratio=0.07, unique_ratio=0.13, num_regions=8, zipf_param=1.3
    )
    t0 = time.perf_counter()
    pages = generator.generate_workload(num_pages=num_pages)
    t_gen_ms = (time.perf_counter() - t0) * 1000.0
    sim_end_time = generator.simulate_write_traffic(pages, num_ticks=40, writes_per_tick=250, current_time=100.0)

    # 2. Module 1: Region-Aware Sampling
    print("[2/5] Executing Module 1: Region-Aware Sampling & Feedback Engine...")
    m1_engine = RegionSamplingEngine(hot_write_rate_threshold=0.06, cold_write_rate_threshold=0.005, min_age_for_cold=10.0)
    t0 = time.perf_counter()
    candidates, m1_stats = m1_engine.filter_and_prioritize(pages, current_time=sim_end_time)
    t_m1_ms = (time.perf_counter() - t0) * 1000.0
    candidate_pages = [c.page for c in candidates]
    cand_reduction_pct = (m1_stats["hot_pages_skipped"] / num_pages) * 100.0

    # 3. Module 2: MinHash + LSH Clustering
    print("[3/5] Executing Module 2: MinHash-LSH Similarity Clustering Engine...")
    m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
    t0 = time.perf_counter()
    clusters, m2_stats = m2_engine.cluster_pages(candidate_pages, verify_similarity=True)
    t_m2_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Module 3: Base-Delta Storage Manager
    print("[4/5] Executing Module 3: Base-Delta Storage & Adaptive Patching Manager...")
    m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.30, base_selection_strategy="medoid")
    t0 = time.perf_counter()
    compressed_clusters, m3_metrics = m3_manager.compress_workload(clusters, pages)
    t_m3_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Module 4: Acceleration Benchmark
    print("[5/5] Executing Module 4 & Module 5: Acceleration & COW Reconstruction...")
    accel_bench = AccelerationBenchmark()
    accel_res = accel_bench.run_benchmark(pages[:200], num_iterations=3)

    cow_handler = SimulatedCOWHandler(delta_manager=m3_manager, sampling_engine=m1_engine)
    cow_handler.populate_from_compressed_clusters(compressed_clusters, pages)

    # Simulate read and write accesses
    delta_page_ids = [pid for pid, entry in cow_handler.page_table.items() if entry.state == PageState.DELTA_PAGE]
    sample_reads = delta_page_ids[:min(500, len(delta_page_ids))]
    for pid in sample_reads:
        _ = cow_handler.read_page(pid)

    sample_writes = delta_page_ids[:min(200, len(delta_page_ids))]
    curr_t = sim_end_time + 10.0
    for pid in sample_writes:
        curr_t += 0.1
        _ = cow_handler.handle_write_fault(pid, offset=16, data=b"WRITE_EVENT", current_time=curr_t)

    lat_stats = cow_handler.get_latency_metrics()
    total_pipeline_time_ms = t_gen_ms + t_m1_ms + t_m2_ms + t_m3_ms

    # Print Summary Table
    print("\n" + "=" * 80)
    print("  AMDEC OFFICIAL SYSTEM EVALUATION REPORT")
    print("=" * 80)
    print(f"  Input Dataset Size              : {num_pages:,} pages ({num_pages*4096/(1024*1024):.2f} MB)")
    print(f"  Baseline Uncompressed Footprint : {m3_metrics['baseline_storage_bytes']/(1024*1024):.2f} MB")
    print(f"  AMDEC Logical Storage Footprint : {m3_metrics['amdec_storage_bytes']/(1024*1024):.2f} MB")
    print(f"  Logical Memory Savings          : {m3_metrics['logical_memory_savings_pct']:.2f}%")
    print(f"  Effective Compression Ratio     : {m3_metrics['compression_ratio']:.2f}:1")
    print("-" * 80)
    print(f"  M1 Volatile Pages Skipped       : {m1_stats['hot_pages_skipped']:,} ({cand_reduction_pct:.1f}%)")
    print(f"  M2 LSH Pair Search Reduction    : {m2_stats['candidate_reduction_pct']:.3f}%")
    print(f"  Similarity Clusters Formed      : {len(clusters):,} clusters ({m2_stats['clustered_pages_count']:,} pages)")
    print(f"  Base Pages / Delta Pages Stored : {m3_metrics['total_base_pages']:,} / {m3_metrics['total_delta_pages']:,}")
    print(f"  Mean Delta Footprint (w/ meta)  : {m3_metrics['avg_delta_bytes']:.1f} bytes / page")
    print("-" * 80)
    print(f"  Reconstruction Latency (Mean)   : {lat_stats['mean_us']:.2f} µs")
    print(f"  Reconstruction Latency (P95/P99): {lat_stats['p95_us']:.2f} µs / {lat_stats['p99_us']:.2f} µs")
    print(f"  M4 SIMD Vector Speedup Factor   : {accel_res['speedup_factor']:.2f}x ({accel_res['vectorized'].throughput_mb_s:,.0f} MB/s)")
    print(f"  Total Pipeline Processing Time  : {total_pipeline_time_ms:.2f} ms ({num_pages/(total_pipeline_time_ms/1000.0):,.0f} pages/sec)")
    print("=" * 80)

    # Export CSV
    if output_csv:
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
        os.makedirs(results_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, f"full_eval_{timestamp_str}.csv")

        row = {
            "timestamp": timestamp_str,
            "num_pages": num_pages,
            "baseline_mb": m3_metrics['baseline_storage_bytes'] / (1024 * 1024),
            "amdec_mb": m3_metrics['amdec_storage_bytes'] / (1024 * 1024),
            "memory_savings_pct": m3_metrics['logical_memory_savings_pct'],
            "compression_ratio": m3_metrics['compression_ratio'],
            "m1_candidate_reduction_pct": cand_reduction_pct,
            "m2_pairwise_reduction_pct": m2_stats['candidate_reduction_pct'],
            "num_clusters": len(clusters),
            "total_base_pages": m3_metrics['total_base_pages'],
            "total_delta_pages": m3_metrics['total_delta_pages'],
            "avg_delta_bytes": m3_metrics['avg_delta_bytes'],
            "p95_delta_bytes": m3_metrics['p95_delta_bytes'],
            "reconstruct_mean_us": lat_stats['mean_us'],
            "reconstruct_p95_us": lat_stats['p95_us'],
            "reconstruct_p99_us": lat_stats['p99_us'],
            "simd_speedup": accel_res['speedup_factor'],
            "total_runtime_ms": total_pipeline_time_ms,
        }

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        print(f"\n[CSV Log Exported]: {csv_path}")


if __name__ == "__main__":
    n = 10000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_full_evaluation(num_pages=n)
