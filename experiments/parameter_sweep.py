"""
AMDEC — Multi-Trial Multi-Parameter Sweep Experiment.

Executes independent multi-trial sweeps across:
1. Mutation Rates (0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%, 30% byte drift)
2. Near-Duplicate Proportions (20%, 40%, 60%, 80%)
3. Workload Scale (1K, 2.5K, 5K, 10K, 15K, 20K pages)

Logs raw trial-level CSV data to results/raw/ and processed statistical summaries to results/processed/.
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


def compute_summary_row(records: List[Dict[str, Any]], group_key: str, group_val: Any) -> Dict[str, Any]:
    """Aggregates trial records for a parameter group into mean, std, and 95% CI."""
    n = len(records)
    savings = [r["memory_savings_pct"] for r in records]
    comps = [r["compression_ratio"] for r in records]
    deltas = [r["avg_delta_bytes"] for r in records]
    runtimes = [r["runtime_ms"] for r in records]
    tps = [r["throughput_pps"] for r in records]

    m_sav, s_sav = float(np.mean(savings)), float(np.std(savings, ddof=1)) if n > 1 else 0.0
    se_sav = float(stats.sem(savings)) if n > 1 else 0.0
    h_sav = float(stats.t.ppf(0.975, n - 1) * se_sav) if n > 1 else 0.0

    return {
        group_key: group_val,
        "num_trials": n,
        "savings_mean": m_sav,
        "savings_std": s_sav,
        "savings_ci_hw": h_sav,
        "savings_ci_lower": m_sav - h_sav,
        "savings_ci_upper": m_sav + h_sav,
        "compression_ratio_mean": float(np.mean(comps)),
        "avg_delta_bytes_mean": float(np.mean(deltas)),
        "runtime_ms_mean": float(np.mean(runtimes)),
        "runtime_ms_std": float(np.std(runtimes, ddof=1)) if n > 1 else 0.0,
        "throughput_pps_mean": float(np.mean(tps)),
    }


def run_parameter_sweeps(trials_per_point: int = 5, initial_seed: int = 42):
    print("=" * 85)
    print(f"  AMDEC MULTI-TRIAL PARAMETER SWEEPS ({trials_per_point} Trials Per Data Point)")
    print("=" * 85)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "processed"))
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Mutation Rate Sweep (0.1% -> 30%)
    # -------------------------------------------------------------
    print("\n[1/3] Executing Mutation Rate Sweep (0.1% -> 30% drift)...")
    mutation_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    mut_raw_records: List[Dict[str, Any]] = []
    mut_summary_records: List[Dict[str, Any]] = []

    for m_rate in mutation_rates:
        point_trials: List[Dict[str, Any]] = []
        for t_idx in range(trials_per_point):
            seed = initial_seed + t_idx
            generator = SyntheticWorkloadGenerator(seed=seed, near_dup_ratio=0.80, exact_dup_ratio=0.07, unique_ratio=0.13)
            pages = generator.generate_workload(num_pages=3000)
            for p in pages:
                if p.metadata.get("type") == "near_duplicate":
                    base_c = generator._create_template_page_content("heap_struct")
                    p.content = generator._mutate_page_content(base_c, mutation_rate=m_rate)

            t0 = time.perf_counter()
            m1 = RegionSamplingEngine()
            candidates, _ = m1.filter_and_prioritize(pages, current_time=100.0)
            m2 = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70, seed=seed)
            clusters, _ = m2.cluster_pages([c.page for c in candidates])
            m3 = BaseDeltaStorageManager(max_delta_ratio=0.30, seed=seed)
            _, metrics = m3.compress_workload(clusters, pages)
            rt_ms = (time.perf_counter() - t0) * 1000.0

            rec = {
                "mutation_rate": m_rate,
                "trial_id": t_idx + 1,
                "seed": seed,
                "num_pages": 3000,
                "memory_savings_pct": metrics["logical_memory_savings_pct"],
                "compression_ratio": metrics["compression_ratio"],
                "num_clusters": len(clusters),
                "avg_delta_bytes": metrics["avg_delta_bytes"],
                "runtime_ms": rt_ms,
                "throughput_pps": 3000 / (rt_ms / 1000.0),
            }
            point_trials.append(rec)
            mut_raw_records.append(rec)

        s_row = compute_summary_row(point_trials, "mutation_rate", m_rate)
        mut_summary_records.append(s_row)
        print(f"  Mutation Rate: {m_rate*100:4.1f}% | Savings: {s_row['savings_mean']:5.2f}% ± {s_row['savings_std']:4.2f}% | 95% CI: [{s_row['savings_ci_lower']:5.2f}%, {s_row['savings_ci_upper']:5.2f}%]")

    # -------------------------------------------------------------
    # 2. Near-Duplicate Proportion Sweep (20% -> 80%)
    # -------------------------------------------------------------
    print("\n[2/3] Executing Similarity Composition Sweep (20% -> 80% near-dup)...")
    near_ratios = [0.20, 0.40, 0.60, 0.80]
    comp_raw_records: List[Dict[str, Any]] = []
    comp_summary_records: List[Dict[str, Any]] = []

    for n_ratio in near_ratios:
        point_trials: List[Dict[str, Any]] = []
        u_ratio = round(1.0 - n_ratio - 0.07, 2)
        for t_idx in range(trials_per_point):
            seed = initial_seed + t_idx
            generator = SyntheticWorkloadGenerator(seed=seed, near_dup_ratio=n_ratio, exact_dup_ratio=0.07, unique_ratio=u_ratio)
            pages = generator.generate_workload(num_pages=3000)

            t0 = time.perf_counter()
            m1 = RegionSamplingEngine()
            candidates, _ = m1.filter_and_prioritize(pages, current_time=100.0)
            m2 = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70, seed=seed)
            clusters, _ = m2.cluster_pages([c.page for c in candidates])
            m3 = BaseDeltaStorageManager(max_delta_ratio=0.30, seed=seed)
            _, metrics = m3.compress_workload(clusters, pages)
            rt_ms = (time.perf_counter() - t0) * 1000.0

            rec = {
                "near_dup_ratio": n_ratio,
                "trial_id": t_idx + 1,
                "seed": seed,
                "num_pages": 3000,
                "memory_savings_pct": metrics["logical_memory_savings_pct"],
                "compression_ratio": metrics["compression_ratio"],
                "num_clusters": len(clusters),
                "avg_delta_bytes": metrics["avg_delta_bytes"],
                "runtime_ms": rt_ms,
                "throughput_pps": 3000 / (rt_ms / 1000.0),
            }
            point_trials.append(rec)
            comp_raw_records.append(rec)

        s_row = compute_summary_row(point_trials, "near_dup_ratio", n_ratio)
        comp_summary_records.append(s_row)
        print(f"  Near-Dup Ratio: {n_ratio*100:4.1f}% | Savings: {s_row['savings_mean']:5.2f}% ± {s_row['savings_std']:4.2f}% | Comp: {s_row['compression_ratio_mean']:.2f}x")

    # -------------------------------------------------------------
    # 3. Scalability Sweep (1K -> 20K pages)
    # -------------------------------------------------------------
    print("\n[3/3] Executing Scalability Sweep (1K -> 20K pages)...")
    scale_sizes = [1000, 2500, 5000, 10000, 15000, 20000]
    scale_raw_records: List[Dict[str, Any]] = []
    scale_summary_records: List[Dict[str, Any]] = []

    for sz in scale_sizes:
        point_trials: List[Dict[str, Any]] = []
        for t_idx in range(trials_per_point):
            seed = initial_seed + t_idx
            generator = SyntheticWorkloadGenerator(seed=seed, near_dup_ratio=0.80, exact_dup_ratio=0.07, unique_ratio=0.13)
            pages = generator.generate_workload(num_pages=sz)

            t0 = time.perf_counter()
            m1 = RegionSamplingEngine()
            candidates, _ = m1.filter_and_prioritize(pages, current_time=100.0)
            m2 = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70, seed=seed)
            clusters, _ = m2.cluster_pages([c.page for c in candidates])
            m3 = BaseDeltaStorageManager(max_delta_ratio=0.30, seed=seed)
            _, metrics = m3.compress_workload(clusters, pages)
            rt_ms = (time.perf_counter() - t0) * 1000.0

            rec = {
                "num_pages": sz,
                "trial_id": t_idx + 1,
                "seed": seed,
                "memory_savings_pct": metrics["logical_memory_savings_pct"],
                "compression_ratio": metrics["compression_ratio"],
                "num_clusters": len(clusters),
                "avg_delta_bytes": metrics["avg_delta_bytes"],
                "runtime_ms": rt_ms,
                "throughput_pps": sz / (rt_ms / 1000.0),
            }
            point_trials.append(rec)
            scale_raw_records.append(rec)

        s_row = compute_summary_row(point_trials, "num_pages", sz)
        scale_summary_records.append(s_row)
        print(f"  Dataset Size: {sz:6,d} pages | Runtime: {s_row['runtime_ms_mean']:6.1f} ± {s_row['runtime_ms_std']:4.1f} ms | Throughput: {s_row['throughput_pps_mean']:5.0f} pps")

    # Export all CSVs
    def export_csv(records: List[Dict[str, Any]], path: str):
        if records:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

    export_csv(mut_raw_records, os.path.join(raw_dir, "mutation_trials.csv"))
    export_csv(mut_summary_records, os.path.join(proc_dir, "mutation_summary.csv"))
    export_csv(comp_raw_records, os.path.join(raw_dir, "composition_trials.csv"))
    export_csv(comp_summary_records, os.path.join(proc_dir, "composition_summary.csv"))
    export_csv(scale_raw_records, os.path.join(raw_dir, "scalability_trials.csv"))
    export_csv(scale_summary_records, os.path.join(proc_dir, "scalability_summary.csv"))

    print("\n" + "=" * 85)
    print("  [SUCCESS] Multi-Trial Sweeps Completed & Exported to results/raw/ and results/processed/")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Multi-Trial Parameter Sweeps")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per point (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed (default: 42)")
    args = parser.parse_args()

    run_parameter_sweeps(trials_per_point=args.trials, initial_seed=args.seed)
