"""
AMDEC — Ablation Study 1: Module 1 Sampling Efficacy.

Compares:
A) AMDEC with M1 Enabled (Region-Aware Sampling & Feedback)
vs.
B) AMDEC with M1 Disabled (Uniform Full Scan)

Measures:
- Candidates forwarded to M2
- M2 Clustering Runtime (ms)
- Total Processing Time (ms)
- COW Fault Breaks avoided
"""

import sys
import os
import time
import csv
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def run_ablation_m1(num_pages: int = 10000):
    print("=" * 80)
    print(f"  AMDEC ABLATION STUDY 1: MODULE 1 SAMPLING EFFICACY (N = {num_pages:,} Pages)")
    print("=" * 80)

    generator = SyntheticWorkloadGenerator(seed=42, zipf_param=1.4)
    pages = generator.generate_workload(num_pages=num_pages)
    sim_end_time = generator.simulate_write_traffic(pages, num_ticks=50, writes_per_tick=300, current_time=100.0)

    m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
    m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.30)

    # -------------------------------------------------------------
    # Configuration A: Full Pipeline with Module 1 Enabled
    # -------------------------------------------------------------
    print("\n[A] Running AMDEC with Module 1 ENABLED...")
    t0 = time.perf_counter()
    m1_engine = RegionSamplingEngine(hot_write_rate_threshold=0.06, cold_write_rate_threshold=0.005)
    candidates, m1_stats = m1_engine.filter_and_prioritize(pages, current_time=sim_end_time)
    t_m1_a = (time.perf_counter() - t0) * 1000.0

    candidate_pages_a = [c.page for c in candidates]

    t0 = time.perf_counter()
    clusters_a, m2_stats_a = m2_engine.cluster_pages(candidate_pages_a)
    t_m2_a = (time.perf_counter() - t0) * 1000.0

    _, m3_metrics_a = m3_manager.compress_workload(clusters_a, pages)
    total_time_a = t_m1_a + t_m2_a

    # -------------------------------------------------------------
    # Configuration B: Pipeline with Module 1 DISABLED (Uniform Scan)
    # -------------------------------------------------------------
    print("[B] Running AMDEC with Module 1 DISABLED (Uniform Full Scan)...")
    # All pages forwarded directly to M2 with zero filtering
    t0 = time.perf_counter()
    clusters_b, m2_stats_b = m2_engine.cluster_pages(pages)
    t_m2_b = (time.perf_counter() - t0) * 1000.0

    _, m3_metrics_b = m3_manager.compress_workload(clusters_b, pages)
    total_time_b = t_m2_b

    # Comparison Metrics
    cand_skipped = m1_stats["hot_pages_skipped"]
    time_saved_ms = total_time_b - total_time_a
    time_saved_pct = (time_saved_ms / total_time_b) * 100.0

    print("\n" + "-" * 80)
    print("  ABLATION 1 RESULTS: M1 ENABLED vs. M1 DISABLED")
    print("-" * 80)
    print(f"  Metric                           | M1 Enabled (Ours) | M1 Disabled (Full) | Delta / Benefit")
    print("-" * 80)
    print(f"  Pages Forwarded to M2            | {len(candidate_pages_a):>17,} | {num_pages:>18,} | -{cand_skipped:,} pages ({cand_skipped/num_pages*100:.1f}%)")
    print(f"  M2 Clustering Time (ms)          | {t_m2_a:>17.2f} | {t_m2_b:>18.2f} | {t_m2_b - t_m2_a:>+.2f} ms")
    print(f"  Total Pipeline Time (ms)         | {total_time_a:>17.2f} | {total_time_b:>18.2f} | {time_saved_ms:>+.2f} ms ({time_saved_pct:+.1f}%)")
    print(f"  Logical Memory Savings (%)       | {m3_metrics_a['logical_memory_savings_pct']:>16.2f}% | {m3_metrics_b['logical_memory_savings_pct']:>17.2f}% | {m3_metrics_a['logical_memory_savings_pct'] - m3_metrics_b['logical_memory_savings_pct']:>+.2f}%")
    print("=" * 80)

    # Save to CSV
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"ablation_m1_{timestamp_str}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "pages_scanned", "m2_time_ms", "total_time_ms", "memory_savings_pct"])
        writer.writerow(["m1_enabled", len(candidate_pages_a), t_m2_a, total_time_a, m3_metrics_a['logical_memory_savings_pct']])
        writer.writerow(["m1_disabled", num_pages, t_m2_b, total_time_b, m3_metrics_b['logical_memory_savings_pct']])

    print(f"[CSV Log Exported]: {csv_path}")


if __name__ == "__main__":
    run_ablation_m1()
