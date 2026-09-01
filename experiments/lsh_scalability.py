"""
AMDEC — LSH Scalability Experiment (Fix #3).

Extends the LSH vs brute-force comparison to larger dataset sizes:
N = 1,000 / 2,500 / 5,000 / 10,000

For each N:
- Runs brute-force O(N²) all-pairs IF feasible (timeout after 5 minutes per trial).
- Always runs MinHash-LSH.
- Reports: candidate pairs, total pairs, reduction %, runtime, speedup, cluster agreement.
- Uses identical workloads/seeds for fair comparison.
"""

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine, UnionFind

# Maximum time to allow brute-force per trial (seconds)
BF_TIMEOUT_S = 8.0


def run_lsh_scalability(
    scale_sizes: Optional[List[int]] = None,
    num_trials: int = 5,
    initial_seed: int = 42,
):
    if scale_sizes is None:
        scale_sizes = [1000, 2500, 5000, 10000]

    print("=" * 100)
    print(f"  AMDEC LSH SCALABILITY EXPERIMENT ({num_trials} Trials)")
    print(f"  Scale points: {scale_sizes}")
    print(f"  Brute-force timeout: {BF_TIMEOUT_S:.0f}s per trial")
    print("=" * 100)

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(raw_dir, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    bf_feasible_up_to = 0

    for n_pages in scale_sizes:
        total_possible = (n_pages * (n_pages - 1)) // 2
        print(f"\n  --- N = {n_pages:,} ({total_possible:,} total possible pairs) ---")

        for t_idx in range(num_trials):
            seed = initial_seed + t_idx
            gen = SyntheticWorkloadGenerator(seed=seed)
            pages = gen.generate_workload(num_pages=n_pages)

            engine = MinHashLSHClusteringEngine(
                block_size=128, num_hashes=32, num_bands=8,
                similarity_threshold=0.70, seed=seed,
            )

            # --- LSH ---
            t0 = time.perf_counter()
            clusters_lsh, stats_lsh = engine.cluster_pages(pages, verify_similarity=True)
            t_lsh_s = time.perf_counter() - t0

            # --- Brute Force (if feasible) ---
            bf_pairs_evaluated = 0
            bf_verified = 0
            bf_clusters_count = 0
            t_bf_s = -1.0  # sentinel: not attempted or timed out
            bf_completed = False

            # Compute minhashes for brute force comparisons
            page_minhashes = {p.page_id: engine.compute_minhash(p.content) for p in pages}

            t0_bf = time.perf_counter()
            uf_bf = UnionFind([p.page_id for p in pages])

            try:
                for i in range(len(pages)):
                    p1_id = pages[i].page_id
                    mh1 = page_minhashes[p1_id]
                    for j in range(i + 1, len(pages)):
                        bf_pairs_evaluated += 1
                        if bf_pairs_evaluated % 50000 == 0:
                            if time.perf_counter() - t0_bf > BF_TIMEOUT_S:
                                raise TimeoutError(f"Brute force exceeded {BF_TIMEOUT_S}s timeout at pair {bf_pairs_evaluated:,}")

                        p2_id = pages[j].page_id
                        mh2 = page_minhashes[p2_id]
                        if mh1.jaccard(mh2) >= engine.similarity_threshold:
                            uf_bf.union(p1_id, p2_id)
                            bf_verified += 1

                t_bf_s = time.perf_counter() - t0_bf
                bf_completed = True
                bf_clusters = [c for c in uf_bf.get_clusters().values() if len(c) > 1]
                bf_clusters_count = len(bf_clusters)
                bf_feasible_up_to = max(bf_feasible_up_to, n_pages)

            except TimeoutError as e:
                t_bf_s = time.perf_counter() - t0_bf
                bf_completed = False
                bf_clusters_count = -1
                print(f"    [Trial {t_idx+1}] Brute force TIMED OUT after {t_bf_s:.1f}s "
                      f"({bf_pairs_evaluated:,}/{total_possible:,} pairs evaluated)")

            # Speedup (only meaningful if BF completed)
            speedup = t_bf_s / t_lsh_s if (bf_completed and t_lsh_s > 0) else -1.0

            # Cluster agreement (if both completed)
            if bf_completed:
                lsh_cluster_set = set()
                for c in clusters_lsh:
                    lsh_cluster_set.add(frozenset(c.page_ids))
                bf_cluster_set = set()
                for c_members in uf_bf.get_clusters().values():
                    if len(c_members) > 1:
                        bf_cluster_set.add(frozenset(c_members))
                agreement = len(lsh_cluster_set & bf_cluster_set) / max(len(bf_cluster_set), 1) * 100.0
            else:
                agreement = -1.0

            rec = {
                "num_pages": n_pages,
                "trial_id": t_idx + 1,
                "seed": seed,
                "total_possible_pairs": total_possible,
                "lsh_candidate_pairs": stats_lsh["candidate_pairs_generated"],
                "lsh_verified_pairs": stats_lsh["verified_pairs_count"],
                "pair_reduction_pct": stats_lsh["candidate_reduction_pct"],
                "lsh_clusters": len(clusters_lsh),
                "lsh_time_s": t_lsh_s,
                "bf_pairs_evaluated": bf_pairs_evaluated,
                "bf_verified_pairs": bf_verified,
                "bf_clusters": bf_clusters_count,
                "bf_time_s": t_bf_s,
                "bf_completed": bf_completed,
                "speedup": speedup,
                "cluster_agreement_pct": agreement,
            }
            all_records.append(rec)

            status = "✓" if bf_completed else "TIMEOUT"
            spd_str = f"{speedup:.1f}x" if speedup > 0 else "N/A"
            agr_str = f"{agreement:.1f}%" if agreement >= 0 else "N/A"
            print(f"    [Trial {t_idx+1}] LSH: {stats_lsh['candidate_pairs_generated']:>8,} pairs / {t_lsh_s:.2f}s | "
                  f"BF: {bf_pairs_evaluated:>12,} / {t_bf_s:.1f}s [{status}] | "
                  f"Reduction: {stats_lsh['candidate_reduction_pct']:.3f}% | "
                  f"Speedup: {spd_str} | Agreement: {agr_str}", flush=True)

    # Export CSV
    csv_path = os.path.join(raw_dir, "lsh_scalability_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    # Summary table
    print("\n" + "=" * 100)
    print(f"  {'N':>7} | {'Total Pairs':>14} | {'LSH Candidates':>14} | {'Reduction %':>12} | "
          f"{'LSH Time':>10} | {'BF Time':>10} | {'Speedup':>8} | {'Agreement':>10}")
    print("-" * 100)

    for n in scale_sizes:
        rr = [r for r in all_records if r["num_pages"] == n]
        lsh_cands = np.mean([r["lsh_candidate_pairs"] for r in rr])
        red = np.mean([r["pair_reduction_pct"] for r in rr])
        lsh_t = np.mean([r["lsh_time_s"] for r in rr])
        completed = [r for r in rr if r["bf_completed"]]
        if completed:
            bf_t = np.mean([r["bf_time_s"] for r in completed])
            spd = np.mean([r["speedup"] for r in completed])
            agr = np.mean([r["cluster_agreement_pct"] for r in completed])
            bf_str = f"{bf_t:8.1f}s"
            spd_str = f"{spd:6.1f}x"
            agr_str = f"{agr:8.1f}%"
        else:
            bf_str = "TIMEOUT"
            spd_str = "N/A"
            agr_str = "N/A"

        total_p = (n * (n - 1)) // 2
        print(f"  {n:>6,} | {total_p:>14,} | {lsh_cands:>14,.0f} | {red:>10.3f}%  | "
              f"{lsh_t:>8.2f}s | {bf_str:>10} | {spd_str:>8} | {agr_str:>10}")

    print("-" * 100)
    print(f"  Largest N where brute force completed: {bf_feasible_up_to:,}")
    print(f"  [Raw CSV Exported]: {csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMDEC LSH Scalability Experiment")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_lsh_scalability(num_trials=args.trials, initial_seed=args.seed)
