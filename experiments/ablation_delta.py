"""
AMDEC — Ablation Study 2: Adaptive vs. Fixed Delta Thresholding.

Compares:
A) Adaptive Thresholding (Fallback to unmerged page if delta > max_delta_ratio * PAGE_SIZE)
vs.
B) Fixed / No Fallback (Always store delta, even if bloated)

Measures:
- Total Logical Storage (MB)
- Memory Savings (%)
- Tail (P95) Delta Sizes
- Unmerged Fallback Count
"""

import sys
import os
import csv
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def run_ablation_delta(num_pages: int = 5000):
    print("=" * 80)
    print(f"  AMDEC ABLATION STUDY 2: ADAPTIVE vs. FIXED DELTA THRESHOLDING (N = {num_pages:,})")
    print("=" * 80)

    # Workload with diverse divergence (some near-duplicates have larger drift)
    generator = SyntheticWorkloadGenerator(seed=42)
    pages = generator.generate_workload(num_pages=num_pages)

    m1_engine = RegionSamplingEngine()
    candidates, _ = m1_engine.filter_and_prioritize(pages, current_time=100.0)
    candidate_pages = [c.page for c in candidates]

    m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
    clusters, _ = m2_engine.cluster_pages(candidate_pages)

    # -------------------------------------------------------------
    # Configuration A: Adaptive Thresholding (max_delta_ratio = 0.25)
    # -------------------------------------------------------------
    manager_adaptive = BaseDeltaStorageManager(max_delta_ratio=0.25)
    _, metrics_a = manager_adaptive.compress_workload(clusters, pages)

    # -------------------------------------------------------------
    # Configuration B: Fixed Thresholding / No Fallback (max_delta_ratio = 1.0)
    # -------------------------------------------------------------
    manager_fixed = BaseDeltaStorageManager(max_delta_ratio=1.0)
    _, metrics_b = manager_fixed.compress_workload(clusters, pages)

    print("\n" + "-" * 80)
    print("  ABLATION 2 RESULTS: ADAPTIVE vs. FIXED DELTA THRESHOLDING")
    print("-" * 80)
    print(f"  Metric                           | Adaptive (Ours)   | Fixed / No Fallback | Difference")
    print("-" * 80)
    print(f"  Total Compressed Footprint (MB)  | {metrics_a['amdec_storage_bytes']/(1024*1024):>17.2f} | {metrics_b['amdec_storage_bytes']/(1024*1024):>19.2f} | {metrics_b['amdec_storage_bytes']/(1024*1024) - metrics_a['amdec_storage_bytes']/(1024*1024):>+.2f} MB")
    print(f"  Logical Memory Savings (%)       | {metrics_a['logical_memory_savings_pct']:>16.2f}% | {metrics_b['logical_memory_savings_pct']:>18.2f}% | {metrics_a['logical_memory_savings_pct'] - metrics_b['logical_memory_savings_pct']:>+.2f}%")
    print(f"  Mean Delta Footprint (bytes)     | {metrics_a['avg_delta_bytes']:>17.1f} | {metrics_b['avg_delta_bytes']:>19.1f} | {metrics_a['avg_delta_bytes'] - metrics_b['avg_delta_bytes']:>+.1f} B")
    print(f"  P95 Delta Footprint (bytes)      | {metrics_a['p95_delta_bytes']:>17.1f} | {metrics_b['p95_delta_bytes']:>19.1f} | {metrics_a['p95_delta_bytes'] - metrics_b['p95_delta_bytes']:>+.1f} B")
    print(f"  Unmerged Fallback Pages          | {metrics_a['total_unmerged_cluster_pages']:>17,} | {metrics_b['total_unmerged_cluster_pages']:>19,} | {metrics_a['total_unmerged_cluster_pages'] - metrics_b['total_unmerged_cluster_pages']:>+d}")
    print("=" * 80)

    # Save to CSV
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"ablation_delta_{timestamp_str}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "compressed_mb", "savings_pct", "avg_delta_bytes", "p95_delta_bytes", "unmerged_pages"])
        writer.writerow(["adaptive", metrics_a['amdec_storage_bytes']/(1024*1024), metrics_a['logical_memory_savings_pct'], metrics_a['avg_delta_bytes'], metrics_a['p95_delta_bytes'], metrics_a['total_unmerged_cluster_pages']])
        writer.writerow(["fixed_no_fallback", metrics_b['amdec_storage_bytes']/(1024*1024), metrics_b['logical_memory_savings_pct'], metrics_b['avg_delta_bytes'], metrics_b['p95_delta_bytes'], metrics_b['total_unmerged_cluster_pages']])

    print(f"[CSV Log Exported]: {csv_path}")


if __name__ == "__main__":
    run_ablation_delta()
