"""
AMDEC — Multi-Trial Paired Ablation Study 2: Adaptive vs. Fixed Delta Thresholding.

Executes paired trials comparing Adaptive Thresholding (max_delta_ratio = 0.30)
vs Fixed/No Fallback (max_delta_ratio = 1.0) on identical workload seeds.
Exports raw trial logs to results/raw/ablation_delta_trials.csv.
"""

import argparse
import csv
from datetime import datetime
import os
import sys
from typing import Any, Dict, List
import numpy as np
import scipy.stats as stats

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def run_ablation_delta(num_pages: int = 5000, num_trials: int = 5, initial_seed: int = 42):
    print("=" * 85)
    print(f"  AMDEC PAIRED ABLATION STUDY 2: ADAPTIVE vs. FIXED THRESHOLDING ({num_trials} Trials, N = {num_pages:,})")
    print("=" * 85)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(raw_dir, exist_ok=True)
    trials_records: List[Dict[str, Any]] = []

    for t_idx in range(num_trials):
        seed = initial_seed + t_idx
        generator = SyntheticWorkloadGenerator(seed=seed)
        pages = generator.generate_workload(num_pages=num_pages)

        m1_engine = RegionSamplingEngine()
        candidates, _ = m1_engine.filter_and_prioritize(pages, current_time=100.0)
        candidate_pages = [c.page for c in candidates]

        m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70, seed=seed)
        clusters, _ = m2_engine.cluster_pages(candidate_pages)

        # Config A: Adaptive Thresholding (0.30)
        manager_adaptive = BaseDeltaStorageManager(max_delta_ratio=0.30, seed=seed)
        _, metrics_a = manager_adaptive.compress_workload(clusters, pages)

        # Config B: Fixed No Fallback (1.0) on IDENTICAL clusters
        manager_fixed = BaseDeltaStorageManager(max_delta_ratio=1.0, seed=seed)
        _, metrics_b = manager_fixed.compress_workload(clusters, pages)

        rec = {
            "trial_id": t_idx + 1,
            "seed": seed,
            "num_pages": num_pages,
            "adaptive_mb": metrics_a["amdec_storage_bytes"] / (1024.0 * 1024.0),
            "fixed_mb": metrics_b["amdec_storage_bytes"] / (1024.0 * 1024.0),
            "adaptive_savings_pct": metrics_a["logical_memory_savings_pct"],
            "fixed_savings_pct": metrics_b["logical_memory_savings_pct"],
            "adaptive_avg_delta_bytes": metrics_a["avg_delta_bytes"],
            "fixed_avg_delta_bytes": metrics_b["avg_delta_bytes"],
            "adaptive_p95_delta_bytes": metrics_a["p95_delta_bytes"],
            "fixed_p95_delta_bytes": metrics_b["p95_delta_bytes"],
            "unmerged_fallback_pages": metrics_a["total_unmerged_cluster_pages"],
        }
        trials_records.append(rec)
        print(f"  [Trial {t_idx+1}/{num_trials}] Seed {seed} | Adaptive: {metrics_a['logical_memory_savings_pct']:5.2f}% (P95 Delta: {metrics_a['p95_delta_bytes']:.0f}B) | Fixed: {metrics_b['logical_memory_savings_pct']:5.2f}% (P95 Delta: {metrics_b['p95_delta_bytes']:.0f}B)")

    csv_path = os.path.join(raw_dir, "ablation_delta_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials_records[0].keys()))
        writer.writeheader()
        writer.writerows(trials_records)

    p95_a = [r["adaptive_p95_delta_bytes"] for r in trials_records]
    p95_b = [r["fixed_p95_delta_bytes"] for r in trials_records]
    print("\n" + "=" * 85)
    print(f"  ABLATION 2 PAIRED SUMMARY: P95 Delta Capped at {np.mean(p95_a):.1f}B (Adaptive) vs {np.mean(p95_b):.1f}B (Fixed)")
    print(f"  [Raw CSV Exported]: {csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Multi-Trial Ablation 2 (Delta Threshold)")
    parser.add_argument("--pages", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ablation_delta(num_pages=args.pages, num_trials=args.trials, initial_seed=args.seed)
