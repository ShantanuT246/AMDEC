"""
AMDEC — Review 2 Live Demo: Integrated M1 Sampling + M2 LSH Clustering + M3 Base-Delta Storage.

Provides a live 3-stage demonstration and multi-scale cluster size analysis for Review 2.
"""

import sys
import os
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def run_review2_demo():
    print("=" * 78)
    print("  AMDEC REVIEW 2 CHECKPOINT: Workload -> M1 Sampling -> M2 LSH -> M3 Base-Delta")
    print("=" * 78)

    num_pages = 10000
    print(f"\n[STAGE 1] Generating Synthetic Anonymous Memory Workload (N = {num_pages:,} pages)...")
    generator = SyntheticWorkloadGenerator(
        seed=42,
        near_dup_ratio=0.80,
        exact_dup_ratio=0.07,
        unique_ratio=0.13,
        num_regions=8,
        zipf_param=1.3,
    )

    t0 = time.perf_counter()
    pages = generator.generate_workload(num_pages=num_pages, cluster_size_range=(4, 24))
    t_gen = (time.perf_counter() - t0) * 1000
    print(f"  --> Generated {len(pages):,} pages in {t_gen:.2f} ms ({num_pages*4096 / (1024*1024):.1f} MB memory pool)")

    print(f"\n[STAGE 2] Module 1: Region-Aware Sampling & Stability Tiering...")
    sim_end_time = generator.simulate_write_traffic(pages, num_ticks=40, writes_per_tick=250, current_time=100.0)
    m1_engine = RegionSamplingEngine(hot_write_rate_threshold=0.06, cold_write_rate_threshold=0.005, min_age_for_cold=10.0)

    t0 = time.perf_counter()
    candidates, m1_stats = m1_engine.filter_and_prioritize(pages, current_time=sim_end_time)
    t_m1 = (time.perf_counter() - t0) * 1000

    candidate_pages = [c.page for c in candidates]
    skipped = m1_stats["hot_pages_skipped"]
    print(f"  --> M1 Runtime: {t_m1:.2f} ms | Filtered HOT Pages: {skipped:,} ({(skipped/num_pages)*100:.1f}%)")
    print(f"  --> Forwarded to M2: {len(candidate_pages):,} prioritized candidate pages")

    print(f"\n[STAGE 3] Module 2: MinHash + LSH Similarity Clustering...")
    m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)

    t0 = time.perf_counter()
    clusters, m2_stats = m2_engine.cluster_pages(candidate_pages, verify_similarity=True)
    t_m2 = (time.perf_counter() - t0) * 1000

    print(f"  --> M2 Runtime: {t_m2:.2f} ms | Throughput: {len(candidate_pages)/(t_m2/1000.0):,.0f} pages/sec")
    print(f"  --> Pairwise Comparisons Saved: {m2_stats['candidate_reduction_pct']:.3f}% (LSH evaluated {m2_stats['candidate_pairs_generated']:,} pairs vs. {m2_stats['total_possible_pairs']:,} brute-force)")
    print(f"  --> Similarity Clusters Formed: {len(clusters):,} clusters ({m2_stats['clustered_pages_count']:,} pages clustered)")

    print(f"\n[STAGE 4] Module 3: Base-Delta Storage & Adaptive Compression...")
    m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.30, base_selection_strategy="medoid")

    t0 = time.perf_counter()
    compressed_clusters, m3_metrics = m3_manager.compress_workload(clusters, pages)
    t_m3 = (time.perf_counter() - t0) * 1000

    base_mb = m3_metrics["baseline_storage_bytes"] / (1024 * 1024)
    amdec_mb = m3_metrics["amdec_storage_bytes"] / (1024 * 1024)
    savings = m3_metrics["logical_memory_savings_pct"]
    ratio = m3_metrics["compression_ratio"]

    print(f"  --> M3 Runtime: {t_m3:.2f} ms")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ Review 2 End-to-End Compression & Savings Summary            │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │ Baseline Uncompressed Storage : {base_mb:>7.2f} MB ({num_pages:,} pages)         │")
    print(f"  │ AMDEC Compressed Storage      : {amdec_mb:>7.2f} MB                       │")
    print(f"  │ Logical Memory Savings        : {savings:>7.2f}%                         │")
    print(f"  │ Compression Ratio             : {ratio:>7.2f}:1                        │")
    print(f"  │ Base Pages Stored             : {m3_metrics['total_base_pages']:>7,}                           │")
    print(f"  │ Delta Pages Stored            : {m3_metrics['total_delta_pages']:>7,}                           │")
    print(f"  │ Mean Delta Footprint (w/ meta): {m3_metrics['avg_delta_bytes']:>7.1f} bytes / page              │")
    print(f"  │ P95 Delta Footprint           : {m3_metrics['p95_delta_bytes']:>7.1f} bytes / page              │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 78)
    print("  [SUCCESS] All Review 2 Checkpoint Deliverables Operational!")
    print("=" * 78)


if __name__ == "__main__":
    run_review2_demo()
