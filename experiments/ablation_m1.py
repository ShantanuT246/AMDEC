"""
AMDEC — Multi-Trial Paired Ablation Study 1: Module 1 Sampling Efficacy.

Executes paired trials comparing M1 Enabled vs M1 Disabled under identical workload seeds.
Exports raw trial logs to results/raw/ablation_m1_trials.csv.
"""

import argparse
import csv
from datetime import datetime
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import scipy.stats as stats

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def run_ablation_m1(num_pages: int = 5000, num_trials: int = 5, initial_seed: int = 42):
    print("=" * 85)
    print(f"  AMDEC PAIRED ABLATION STUDY 1: MODULE 1 SAMPLING ({num_trials} Trials, N = {num_pages:,})")
    print("=" * 85)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "processed"))
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    trials_records: List[Dict[str, Any]] = []

    for t_idx in range(num_trials):
        seed = initial_seed + t_idx
        generator = SyntheticWorkloadGenerator(seed=seed, zipf_param=1.4)
        pages = generator.generate_workload(num_pages=num_pages)
        sim_end_time = generator.simulate_write_traffic(pages, num_ticks=40, writes_per_tick=200, current_time=100.0)

        m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70, seed=seed)
        m3_manager = BaseDeltaStorageManager(max_delta_ratio=0.30, seed=seed)

        # Config A: M1 Enabled
        t0 = time.perf_counter()
        m1_engine = RegionSamplingEngine(hot_write_rate_threshold=0.06, cold_write_rate_threshold=0.005)
        candidates, m1_stats = m1_engine.filter_and_prioritize(pages, current_time=sim_end_time)
        t_m1_a = (time.perf_counter() - t0) * 1000.0
        candidate_pages_a = [c.page for c in candidates]

        t0 = time.perf_counter()
        clusters_a, _ = m2_engine.cluster_pages(candidate_pages_a)
        t_m2_a = (time.perf_counter() - t0) * 1000.0
        _, metrics_a = m3_manager.compress_workload(clusters_a, pages)
        total_time_a = t_m1_a + t_m2_a

        # Config B: M1 Disabled (Uniform Full Scan on IDENTICAL pages)
        t0 = time.perf_counter()
        clusters_b, _ = m2_engine.cluster_pages(pages)
        t_m2_b = (time.perf_counter() - t0) * 1000.0
        _, metrics_b = m3_manager.compress_workload(clusters_b, pages)
        total_time_b = t_m2_b

        time_saved_ms = total_time_b - total_time_a
        time_saved_pct = (time_saved_ms / total_time_b) * 100.0 if total_time_b > 0 else 0.0

        rec = {
            "trial_id": t_idx + 1,
            "seed": seed,
            "num_pages": num_pages,
            "m1_enabled_pages": len(candidate_pages_a),
            "m1_disabled_pages": num_pages,
            "hot_pages_skipped": m1_stats["hot_pages_skipped"],
            "m1_enabled_time_ms": total_time_a,
            "m1_disabled_time_ms": total_time_b,
            "time_saved_ms": time_saved_ms,
            "time_saved_pct": time_saved_pct,
            "m1_enabled_savings_pct": metrics_a["logical_memory_savings_pct"],
            "m1_disabled_savings_pct": metrics_b["logical_memory_savings_pct"],
        }
        trials_records.append(rec)
        print(f"  [Trial {t_idx+1}/{num_trials}] Seed {seed} | Skipped: {m1_stats['hot_pages_skipped']:3d} ({m1_stats['hot_pages_skipped']/num_pages*100:4.1f}%) | Time Saved: {time_saved_ms:5.1f} ms ({time_saved_pct:4.1f}%)")

    # Export CSV
    csv_path = os.path.join(raw_dir, "ablation_m1_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials_records[0].keys()))
        writer.writeheader()
        writer.writerows(trials_records)

    # Statistical summary
    time_sav = [r["time_saved_pct"] for r in trials_records]
    print("\n" + "=" * 85)
    print(f"  ABLATION 1 PAIRED SUMMARY: Mean Time Saved = {np.mean(time_sav):.1f}% ± {np.std(time_sav, ddof=1):.1f}% (95% CI: [{np.mean(time_sav) - stats.sem(time_sav)*stats.t.ppf(0.975, len(time_sav)-1):.1f}%, {np.mean(time_sav) + stats.sem(time_sav)*stats.t.ppf(0.975, len(time_sav)-1):.1f}%])")
    print(f"  [Raw CSV Exported]: {csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Multi-Trial Ablation 1 (M1 Sampling)")
    parser.add_argument("--pages", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ablation_m1(num_pages=args.pages, num_trials=args.trials, initial_seed=args.seed)
