"""
AMDEC — Adversarial Delta-Threshold Ablation (Fix #1).

Tests the adaptive delta fallback mechanism under high-mutation workloads
where near-duplicate pages have large byte divergence from their cluster base.

Sweeps mutation rates from 1% to 60% (word-level) to find the regime where
the 30% adaptive threshold activates fallbacks vs. a fixed/no-fallback (100%) policy.

Uses identical workloads for both configurations (paired comparison).
"""

import argparse
import csv
import os
import random as _random
import struct
import sys
import time
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator, Page, PAGE_SIZE
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine, SimilarityCluster
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def _build_adversarial_workload(seed: int, num_pages: int, mutation_rate: float) -> List[Page]:
    """
    Builds a workload where ALL near-duplicate pages use a specific mutation_rate
    instead of the default 0.2–1.2% range. This forces high byte divergence
    to exercise the delta fallback mechanism.

    The workload composition (80% near-dup, 7% exact, 13% unique) is preserved.
    """
    gen = SyntheticWorkloadGenerator(seed=seed)
    # Generate the standard workload first to get the same structure
    pages = gen.generate_workload(num_pages=num_pages)

    # Now re-mutate all near-duplicate pages at the specified mutation_rate
    # We need a fresh RNG for the re-mutation to be deterministic per seed
    mut_rng = np.random.default_rng(seed + 10000)
    mut_py_rng = _random.Random(seed + 10000)

    for p in pages:
        if p.metadata.get("type") == "near_duplicate":
            # Create a clean template and mutate at the adversarial rate
            base = gen._create_template_page_content("heap_struct")
            mutated = bytearray(base)
            num_mutations = max(1, int((PAGE_SIZE // 8) * mutation_rate))
            for _ in range(num_mutations):
                word_idx = mut_py_rng.randint(0, (PAGE_SIZE // 8) - 1)
                offset = word_idx * 8
                delta_val = mut_py_rng.randint(1, 1000)
                orig_val = struct.unpack_from("<Q", mutated, offset)[0]
                new_val = (orig_val + delta_val) & 0xFFFFFFFFFFFFFFFF
                struct.pack_into("<Q", mutated, offset, new_val)
            p.content = mutated

    return pages


def run_adversarial_delta_ablation(
    num_pages: int = 3000,
    num_trials: int = 5,
    initial_seed: int = 42,
):
    mutation_rates = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    print("=" * 95)
    print(f"  AMDEC ADVERSARIAL DELTA-THRESHOLD ABLATION ({num_trials} Trials, N = {num_pages:,})")
    print(f"  Mutation rates: {[f'{r*100:.0f}%' for r in mutation_rates]}")
    print(f"  Adaptive threshold = 30% ({int(PAGE_SIZE * 0.30)}B) vs Fixed = 100% (no fallback)")
    print("=" * 95)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(raw_dir, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    for m_rate in mutation_rates:
        for t_idx in range(num_trials):
            seed = initial_seed + t_idx
            pages = _build_adversarial_workload(seed, num_pages, m_rate)

            # Cluster with LSH (same for both configs)
            m2 = MinHashLSHClusteringEngine(
                block_size=128, num_hashes=32, num_bands=8,
                similarity_threshold=0.70, seed=seed,
            )
            clusters, _ = m2.cluster_pages(pages)

            # Config A: Adaptive (max_delta_ratio = 0.30)
            m3_adaptive = BaseDeltaStorageManager(max_delta_ratio=0.30, seed=seed)
            _, metrics_a = m3_adaptive.compress_workload(clusters, pages)

            # Config B: Fixed / no fallback (max_delta_ratio = 1.0) on IDENTICAL clusters
            m3_fixed = BaseDeltaStorageManager(max_delta_ratio=1.0, seed=seed)
            _, metrics_b = m3_fixed.compress_workload(clusters, pages)

            rec = {
                "mutation_rate": m_rate,
                "trial_id": t_idx + 1,
                "seed": seed,
                "num_pages": num_pages,
                "num_clusters": len(clusters),
                # Adaptive
                "adaptive_savings_pct": metrics_a["logical_memory_savings_pct"],
                "adaptive_mb": metrics_a["amdec_storage_bytes"] / (1024.0 * 1024.0),
                "adaptive_avg_delta_bytes": metrics_a["avg_delta_bytes"],
                "adaptive_p95_delta_bytes": metrics_a["p95_delta_bytes"],
                "adaptive_fallback_pages": metrics_a["total_unmerged_cluster_pages"],
                "adaptive_delta_pages": metrics_a["total_delta_pages"],
                "adaptive_base_pages": metrics_a["total_base_pages"],
                # Fixed
                "fixed_savings_pct": metrics_b["logical_memory_savings_pct"],
                "fixed_mb": metrics_b["amdec_storage_bytes"] / (1024.0 * 1024.0),
                "fixed_avg_delta_bytes": metrics_b["avg_delta_bytes"],
                "fixed_p95_delta_bytes": metrics_b["p95_delta_bytes"],
                "fixed_fallback_pages": metrics_b["total_unmerged_cluster_pages"],
                "fixed_delta_pages": metrics_b["total_delta_pages"],
                # Difference
                "savings_diff_pct": metrics_a["logical_memory_savings_pct"] - metrics_b["logical_memory_savings_pct"],
                "fallback_diff": metrics_a["total_unmerged_cluster_pages"] - metrics_b["total_unmerged_cluster_pages"],
            }
            all_records.append(rec)

        # Aggregate for this mutation rate
        rate_recs = [r for r in all_records if r["mutation_rate"] == m_rate]
        a_sav = [r["adaptive_savings_pct"] for r in rate_recs]
        f_sav = [r["fixed_savings_pct"] for r in rate_recs]
        a_fb = [r["adaptive_fallback_pages"] for r in rate_recs]
        a_d = [r["adaptive_avg_delta_bytes"] for r in rate_recs]
        f_d = [r["fixed_avg_delta_bytes"] for r in rate_recs]

        print(f"\n  Mutation {m_rate*100:4.0f}% | Adaptive: {np.mean(a_sav):5.2f}% ± {np.std(a_sav, ddof=1):4.2f}% "
              f"(Fallback: {np.mean(a_fb):4.0f} pages, Δ={np.mean(a_d):6.1f}B) | "
              f"Fixed: {np.mean(f_sav):5.2f}% ± {np.std(f_sav, ddof=1):4.2f}% (Δ={np.mean(f_d):6.1f}B) | "
              f"Diff: {np.mean(a_sav) - np.mean(f_sav):+.2f}%")

    # Export CSV
    csv_path = os.path.join(raw_dir, "ablation_delta_adversarial_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n  [Raw CSV Exported]: {csv_path}")

    # Summary table
    print("\n" + "=" * 95)
    print(f"  {'Mutation':>8} | {'Adaptive Savings':>18} | {'Fixed Savings':>18} | {'Fallbacks':>10} | {'Δ_adap (B)':>11} | {'Δ_fixed (B)':>12}")
    print("-" * 95)
    for m_rate in mutation_rates:
        rr = [r for r in all_records if r["mutation_rate"] == m_rate]
        a_s = np.mean([r["adaptive_savings_pct"] for r in rr])
        f_s = np.mean([r["fixed_savings_pct"] for r in rr])
        fb = np.mean([r["adaptive_fallback_pages"] for r in rr])
        ad = np.mean([r["adaptive_avg_delta_bytes"] for r in rr])
        fd = np.mean([r["fixed_avg_delta_bytes"] for r in rr])
        print(f"  {m_rate*100:6.0f}%  | {a_s:16.2f}%  | {f_s:16.2f}%  | {fb:10.0f} | {ad:11.1f} | {fd:12.1f}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Adversarial Delta Threshold Ablation")
    parser.add_argument("--pages", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_adversarial_delta_ablation(num_pages=args.pages, num_trials=args.trials, initial_seed=args.seed)
