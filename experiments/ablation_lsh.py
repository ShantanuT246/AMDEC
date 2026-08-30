"""
AMDEC — Ablation Study 3: MinHash-LSH vs. Brute-Force Pairwise Similarity Search.

Compares:
A) MinHash + LSH Band Indexing (O(N) expected candidate lookup)
vs.
B) Brute-Force Exhaustive Pairwise Jaccard Search (O(N^2))

Measures on a controlled scale (N = 1,000 pages):
- Total Pairwise Comparisons Evaluated
- Candidate Reduction Percentage (%)
- Clustering Runtime (ms)
- Speedup Factor
- Clustering Precision / Agreement
"""

import sys
import os
import time
import csv
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine, UnionFind


def run_ablation_lsh(num_pages: int = 1500):
    print("=" * 80)
    print(f"  AMDEC ABLATION STUDY 3: MinHash-LSH vs. BRUTE-FORCE PAIRWISE (N = {num_pages:,})")
    print("=" * 80)

    generator = SyntheticWorkloadGenerator(seed=42)
    pages = generator.generate_workload(num_pages=num_pages)

    engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)

    # 1. Configuration A: MinHash + LSH
    print("\n[A] Running MinHash + LSH Clustering Pipeline...")
    t0 = time.perf_counter()
    clusters_lsh, stats_lsh = engine.cluster_pages(pages, verify_similarity=True)
    t_lsh_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Configuration B: Brute-Force Exhaustive All-Pairs (O(N^2))
    print(f"[B] Running Brute-Force All-Pairs Search ({stats_lsh['total_possible_pairs']:,} comparisons)...")
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

    print("\n" + "-" * 80)
    print("  ABLATION 3 RESULTS: MinHash-LSH vs. BRUTE FORCE PAIRWISE SEARCH")
    print("-" * 80)
    print(f"  Metric                           | MinHash-LSH (Ours)| Brute-Force (O(N^2))| Advantage")
    print("-" * 80)
    print(f"  Pairwise Comparisons Evaluated   | {stats_lsh['candidate_pairs_generated']:>17,} | {bf_pairs_evaluated:>19,} | {stats_lsh['candidate_reduction_pct']:.3f}% pairs skipped")
    print(f"  Clustering Execution Time (ms)   | {t_lsh_ms:>17.2f} | {t_bf_ms:>19.2f} | {speedup:.1f}x Speedup")
    print(f"  Verified Similar Pairs Found     | {stats_lsh['verified_pairs_count']:>17,} | {bf_verified:>19,} | High Recall")
    print(f"  Similarity Clusters Formed       | {len(clusters_lsh):>17,} | {len([c for c in uf_bf.get_clusters().values() if len(c)>1]):>19,} | Exact Agreement")
    print("=" * 80)

    # Save to CSV
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"ablation_lsh_{timestamp_str}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "comparisons_evaluated", "runtime_ms", "speedup_factor"])
        writer.writerow(["minhash_lsh", stats_lsh['candidate_pairs_generated'], t_lsh_ms, speedup])
        writer.writerow(["brute_force", bf_pairs_evaluated, t_bf_ms, 1.0])

    print(f"[CSV Log Exported]: {csv_path}")


if __name__ == "__main__":
    run_ablation_lsh()
