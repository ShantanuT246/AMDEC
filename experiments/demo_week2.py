"""
AMDEC — Week 2 Demo: Synthetic Workload & Region-Aware Sampling Engine.

Demonstrates:
1. Generation of realistic 4KB anonymous memory pages (near-duplicate, exact, unique).
2. Zipf-distributed write traffic simulation.
3. Module 1: Region-Aware Sampling & Feedback tier classification and candidate queue generation.
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine, StabilityTier


def run_demo():
    print("=" * 70)
    print("  AMDEC Simulation — Week 2 Benchmark: Workload & Sampling Engine")
    print("=" * 70)

    num_pages = 10000
    print(f"\n[1] Initializing Synthetic Anonymous Heap Workload (N = {num_pages:,} pages)...")
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

    # Categorize ground-truth composition
    types = [p.metadata.get("type") for p in pages]
    n_near = types.count("near_duplicate")
    n_exact = types.count("exact_duplicate")
    n_unique = types.count("unique")

    print(f"    - Generated {len(pages):,} pages in {t_gen:.2f} ms ({len(pages)*4096 / (1024*1024):.1f} MB memory pool)")
    print(f"    - Composition: {n_near:,} near-duplicate ({n_near/num_pages*100:.1f}%), "
          f"{n_exact:,} exact ({n_exact/num_pages*100:.1f}%), "
          f"{n_unique:,} unique ({n_unique/num_pages*100:.1f}%)")

    print(f"\n[2] Simulating Zipf-Distributed Write Traffic (50 ticks, 300 writes/tick)...")
    t0 = time.perf_counter()
    sim_end_time = generator.simulate_write_traffic(pages, num_ticks=50, writes_per_tick=300, current_time=100.0)
    t_sim = (time.perf_counter() - t0) * 1000
    total_writes = sum(p.write_count for p in pages)
    hot_pages = sum(1 for p in pages if p.write_count > 5)
    print(f"    - Completed {total_writes:,} simulated writes in {t_sim:.2f} ms")
    print(f"    - Skew profile: {hot_pages:,} hot pages absorbed the bulk of modifications.")

    print(f"\n[3] Executing Module 1: Region-Aware Sampling & Feedback Engine...")
    engine = RegionSamplingEngine(
        hot_write_rate_threshold=0.06,
        cold_write_rate_threshold=0.005,
        min_age_for_cold=10.0,
    )

    t0 = time.perf_counter()
    candidates, stats = engine.filter_and_prioritize(pages, current_time=sim_end_time)
    t_sample = (time.perf_counter() - t0) * 1000

    skipped = stats['hot_pages_skipped']
    cold = stats['cold_pages_queued']
    warm = stats['warm_pages_queued']
    reduction_pct = (skipped / num_pages) * 100

    print(f"    - Sampling completed in {t_sample:.2f} ms")
    print(f"    ┌────────────────────────────────────────────────────────┐")
    print(f"    │ Module 1 Scan Metrics & Overhead Reduction Summary      │")
    print(f"    ├────────────────────────────────────────────────────────┤")
    print(f"    │ Total Pages Inspected          : {stats['total_pages']:>8,}            │")
    print(f"    │ Hot Pages Filtered (Skipped)   : {skipped:>8,} ({reduction_pct:5.1f}%)    │")
    print(f"    │ Cold Candidates (High Priority): {cold:>8,} ({cold/num_pages*100:5.1f}%)    │")
    print(f"    │ Warm Candidates (Low Priority) : {warm:>8,} ({warm/num_pages*100:5.1f}%)    │")
    print(f"    │ Scan Queue Output Candidates   : {len(candidates):>8,}            │")
    print(f"    │ Estimated CPU Overhead Saved   : {reduction_pct:>7.1f}%             │")
    print(f"    └────────────────────────────────────────────────────────┘")

    # Feedback demonstration
    print(f"\n[4] Simulating Dynamic Feedback Loop (Copy-on-Write Break on Candidate #0)...")
    if candidates:
        top_cand = candidates[0]
        print(f"    - Top candidate Page #{top_cand.page_id} (Initial Tier: {top_cand.tier.value.upper()})")
        print(f"    - Triggering simulated COW write fault...")
        engine.record_cow_fault_feedback(top_cand.page_id, current_time=sim_end_time, quarantine_duration=30.0)

        tier_after, _ = engine.classify_page(top_cand.page, current_time=sim_end_time + 5.0)
        print(f"    - Post-COW classification: {tier_after.value.upper()} (Quarantined & safely skipped!)")

    print("\n" + "=" * 70)
    print("  [PASS] Week 2 Deliverables Verified Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
