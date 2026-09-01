"""
AMDEC — Multi-Trial Paired Ablation Study 3: MinHash-LSH vs. Brute-Force All-Pairs Search.

Executes paired trials comparing MinHash-LSH vs Brute-Force O(N^2) exhaustive pairwise search
on identical workload seeds. Exports raw trial logs to results/raw/ablation_lsh_trials.csv.
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
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine, UnionFind


def run_ablation_lsh(num_pages: int = 1500, num_trials: int = 5, initial_seed: int = 42):
    print("=" * 85)
    print(f"  AMDEC PAIRED ABLATION STUDY 3: MinHash-LSH vs. BRUTE-FORCE ({num_trials} Trials, N = {num_pages:,})")
    print("=" * 85)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(raw_dir, exist_ok=True)
    trials_records: List[Dict[str, Any]] = []

    for t_idx in range(num_trials):
        seed = initial_seed + t_idx
        generator = SyntheticWorkloadGenerator(seed=seed)
        pages = generator.generate_workload(num_pages=num_pages)

        engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70, seed=seed)

        # 1. Config A: MinHash + LSH
        t0 = time.perf_counter()
        clusters_lsh, stats_lsh = engine.cluster_pages(pages, verify_similarity=True)
        t_lsh_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Config B: Brute-Force All-Pairs on IDENTICAL pages
        t0 = time.perf_counter()
        page_minhashes = {p.page_id: engine.compute_minhash(p.content) for p in pages}
        uf_bf = UnionFind([p.page_id for p in pages])
        n = len(pages)
        bf_pairs_evaluated = 0
        bf_verified = 0

        for i in range(n):
            p1_id = pages[i].page_id
            mh1 = page_minhashes[p1_id]
            for j in range(i + 1, n):
                p2_id = pages[j].page_id
                mh2 = page_minhashes[p2_id]
                bf_pairs_evaluated += 1
                if mh1.jaccard(mh2) >= engine.similarity_threshold:
                    uf_bf.union(p1_id, p2_id)
                    bf_verified += 1

        t_bf_ms = (time.perf_counter() - t0) * 1000.0
        speedup = t_bf_ms / t_lsh_ms if t_lsh_ms > 0 else 1.0

        rec = {
            "trial_id": t_idx + 1,
            "seed": seed,
            "num_pages": num_pages,
            "lsh_candidates": stats_lsh["candidate_pairs_generated"],
            "bf_comparisons": bf_pairs_evaluated,
            "search_reduction_pct": stats_lsh["candidate_reduction_pct"],
            "lsh_time_ms": t_lsh_ms,
            "bf_time_ms": t_bf_ms,
            "speedup_factor": speedup,
            "lsh_clusters": len(clusters_lsh),
            "bf_clusters": len([c for c in uf_bf.get_clusters().values() if len(c) > 1]),
        }
        trials_records.append(rec)
        print(f"  [Trial {t_idx+1}/{num_trials}] Seed {seed} | LSH Pairs: {stats_lsh['candidate_pairs_generated']:>6,d} vs BF: {bf_pairs_evaluated:>9,d} | Reduction: {stats_lsh['candidate_reduction_pct']:.3f}% | Speedup: {speedup:.1f}x")

    csv_path = os.path.join(raw_dir, "ablation_lsh_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials_records[0].keys()))
        writer.writeheader()
        writer.writerows(trials_records)

    reds = [r["search_reduction_pct"] for r in trials_records]
    spds = [r["speedup_factor"] for r in trials_records]
    print("\n" + "=" * 85)
    print(f"  ABLATION 3 PAIRED SUMMARY: Search Reduction = {np.mean(reds):.3f}% ± {np.std(reds, ddof=1):.3f}% | Speedup = {np.mean(spds):.1f}x ± {np.std(spds, ddof=1):.1f}x")
    print(f"  [Raw CSV Exported]: {csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC Multi-Trial Ablation 3 (LSH vs Brute Force)")
    parser.add_argument("--pages", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ablation_lsh(num_pages=args.pages, num_trials=args.trials, initial_seed=args.seed)
