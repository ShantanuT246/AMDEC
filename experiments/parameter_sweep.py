"""
AMDEC — Parameter Sweep Experiment.

Sweeps across:
1. Mutation Rates (0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%, 30% byte drift)
2. Near-Duplicate Proportions (20%, 40%, 60%, 80%)
3. Workload Scale (1K, 5K, 10K, 25K pages)

Logs raw CSV data to results/raw/parameter_sweep_<timestamp>.csv for sensitivity curve plotting.
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


def run_parameter_sweep():
    print("=" * 80)
    print("  AMDEC PARAMETER SWEEP: SENSITIVITY CURVES & SCALABILITY")
    print("=" * 80)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"parameter_sweep_{timestamp_str}.csv")

    fieldnames = [
        "experiment_type", "mutation_rate", "near_dup_ratio", "num_pages",
        "memory_savings_pct", "compression_ratio", "num_clusters",
        "total_delta_pages", "avg_delta_bytes", "runtime_ms"
    ]

    records = []

    # 1. Mutation Rate Sweep (0.1% -> 30%)
    print("\n[1/3] Executing Mutation Rate Sweep (Sensitivity Curve)...")
    mutation_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    for m_rate in mutation_rates:
        generator = SyntheticWorkloadGenerator(seed=42, near_dup_ratio=0.80, exact_dup_ratio=0.07, unique_ratio=0.13)
        # Override mutation generator for this run
        pages = generator.generate_workload(num_pages=3000)
        # Apply specific mutation rate to near-duplicate pages
        for p in pages:
            if p.metadata.get("type") == "near_duplicate":
                base_c = generator._create_template_page_content("heap_struct")
                p.content = generator._mutate_page_content(base_c, mutation_rate=m_rate)

        t0 = time.perf_counter()
        m1_engine = RegionSamplingEngine()
        candidates, _ = m1_engine.filter_and_prioritize(pages, current_time=100.0)
        candidate_pages = [c.page for c in candidates]

        m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.60)
        clusters, _ = m2_engine.cluster_pages(candidate_pages)

        m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.35)
        _, metrics = m3_manager.compress_workload(clusters, pages)
        rt_ms = (time.perf_counter() - t0) * 1000.0

        rec = {
            "experiment_type": "mutation_sweep",
            "mutation_rate": m_rate,
            "near_dup_ratio": 0.80,
            "num_pages": 3000,
            "memory_savings_pct": metrics["logical_memory_savings_pct"],
            "compression_ratio": metrics["compression_ratio"],
            "num_clusters": len(clusters),
            "total_delta_pages": metrics["total_delta_pages"],
            "avg_delta_bytes": metrics["avg_delta_bytes"],
            "runtime_ms": rt_ms,
        }
        records.append(rec)
        print(f"  Mutation Rate: {m_rate*100:4.1f}% | Savings: {metrics['logical_memory_savings_pct']:5.2f}% | Avg Delta: {metrics['avg_delta_bytes']:5.1f} B | Runtime: {rt_ms:6.1f} ms")

    # 2. Near-Duplicate Proportion Sweep (20% -> 80%)
    print("\n[2/3] Executing Similarity Composition Sweep...")
    near_ratios = [0.20, 0.40, 0.60, 0.80]
    for n_ratio in near_ratios:
        u_ratio = round(1.0 - n_ratio - 0.07, 2)
        generator = SyntheticWorkloadGenerator(seed=42, near_dup_ratio=n_ratio, exact_dup_ratio=0.07, unique_ratio=u_ratio)
        pages = generator.generate_workload(num_pages=3000)

        t0 = time.perf_counter()
        m1_engine = RegionSamplingEngine()
        candidates, _ = m1_engine.filter_and_prioritize(pages, current_time=100.0)
        candidate_pages = [c.page for c in candidates]

        m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
        clusters, _ = m2_engine.cluster_pages(candidate_pages)

        m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.30)
        _, metrics = m3_manager.compress_workload(clusters, pages)
        rt_ms = (time.perf_counter() - t0) * 1000.0

        rec = {
            "experiment_type": "near_dup_sweep",
            "mutation_rate": 0.005,
            "near_dup_ratio": n_ratio,
            "num_pages": 3000,
            "memory_savings_pct": metrics["logical_memory_savings_pct"],
            "compression_ratio": metrics["compression_ratio"],
            "num_clusters": len(clusters),
            "total_delta_pages": metrics["total_delta_pages"],
            "avg_delta_bytes": metrics["avg_delta_bytes"],
            "runtime_ms": rt_ms,
        }
        records.append(rec)
        print(f"  Near-Dup Mix: {n_ratio*100:4.1f}% | Savings: {metrics['logical_memory_savings_pct']:5.2f}% | Ratio: {metrics['compression_ratio']:4.2f}:1 | Runtime: {rt_ms:6.1f} ms")

    # 3. Scalability Sweep (1K -> 10K pages)
    print("\n[3/3] Executing Page-Count Scalability Sweep...")
    page_counts = [1000, 3000, 6000, 10000]
    for count in page_counts:
        generator = SyntheticWorkloadGenerator(seed=42)
        pages = generator.generate_workload(num_pages=count)

        t0 = time.perf_counter()
        m1_engine = RegionSamplingEngine()
        candidates, _ = m1_engine.filter_and_prioritize(pages, current_time=100.0)
        candidate_pages = [c.page for c in candidates]

        m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
        clusters, _ = m2_engine.cluster_pages(candidate_pages)

        m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.30)
        _, metrics = m3_manager.compress_workload(clusters, pages)
        rt_ms = (time.perf_counter() - t0) * 1000.0

        rec = {
            "experiment_type": "scalability_sweep",
            "mutation_rate": 0.005,
            "near_dup_ratio": 0.80,
            "num_pages": count,
            "memory_savings_pct": metrics["logical_memory_savings_pct"],
            "compression_ratio": metrics["compression_ratio"],
            "num_clusters": len(clusters),
            "total_delta_pages": metrics["total_delta_pages"],
            "avg_delta_bytes": metrics["avg_delta_bytes"],
            "runtime_ms": rt_ms,
        }
        records.append(rec)
        print(f"  Page Count: {count:>6,} | Savings: {metrics['logical_memory_savings_pct']:5.2f}% | Runtime: {rt_ms:6.1f} ms ({count/(rt_ms/1000.0):,.0f} p/s)")

    # Save to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[CSV Log Exported]: {csv_path}")


if __name__ == "__main__":
    run_parameter_sweep()
