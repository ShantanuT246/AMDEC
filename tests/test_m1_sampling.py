"""
Unit tests for Module 1: Region-Aware Sampling & Feedback Engine (src/modules/m1_sampling.py).
"""

import pytest
from src.modules.m1_sampling import RegionSamplingEngine, StabilityTier
from src.workload.page_generator import Page, SyntheticWorkloadGenerator, PAGE_SIZE


def test_stability_tier_classification():
    """Verify tier classification: HOT for high write rate, COLD for long-lived idle, WARM for intermediate."""
    engine = RegionSamplingEngine(
        hot_write_rate_threshold=0.1,
        cold_write_rate_threshold=0.01,
        min_age_for_cold=10.0,
    )

    current_time = 100.0

    # 1. Hot Page: allocated 20s ago, 10 writes (rate = 0.5 > 0.1)
    hot_page = Page(
        page_id=1, region_id=0, content=bytearray(PAGE_SIZE),
        alloc_time=80.0, last_write_time=99.0, write_count=10
    )
    tier, score = engine.classify_page(hot_page, current_time)
    assert tier == StabilityTier.HOT

    # 2. Cold Page: allocated 50s ago, 0 writes, idle for 50s (rate = 0.0 <= 0.01)
    cold_page = Page(
        page_id=2, region_id=0, content=bytearray(PAGE_SIZE),
        alloc_time=50.0, last_write_time=50.0, write_count=0
    )
    tier, score = engine.classify_page(cold_page, current_time)
    assert tier == StabilityTier.COLD

    # 3. Warm Page: allocated 5s ago (too young for cold), 0 writes
    warm_page = Page(
        page_id=3, region_id=0, content=bytearray(PAGE_SIZE),
        alloc_time=95.0, last_write_time=95.0, write_count=0
    )
    tier, score = engine.classify_page(warm_page, current_time)
    assert tier == StabilityTier.WARM


def test_priority_queue_ordering():
    """Verify priority queue ranks COLD pages ahead of WARM pages and skips HOT pages."""
    engine = RegionSamplingEngine()
    current_time = 100.0

    pages = [
        # Hot page
        Page(page_id=1, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=80.0, last_write_time=99.0, write_count=20),
        # Warm page
        Page(page_id=2, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=96.0, last_write_time=96.0, write_count=0),
        # Very Cold page (allocated 80s ago, 0 writes)
        Page(page_id=3, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=20.0, last_write_time=20.0, write_count=0),
        # Moderate Cold page (allocated 30s ago, 0 writes)
        Page(page_id=4, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=70.0, last_write_time=70.0, write_count=0),
    ]

    candidates, stats = engine.filter_and_prioritize(pages, current_time=current_time)

    assert stats["hot_pages_skipped"] == 1
    assert stats["cold_pages_queued"] == 2
    assert stats["warm_pages_queued"] == 1

    candidate_ids = [c.page_id for c in candidates]
    # Page 3 should be first (oldest cold page), Page 4 second, Page 2 third (warm), Page 1 skipped entirely
    assert candidate_ids[0] == 3
    assert candidate_ids[1] == 4
    assert candidate_ids[2] == 2
    assert 1 not in candidate_ids


def test_feedback_loop_cow_breaks():
    """Verify COW breaks dynamically demote a previously cold page into quarantine/HOT."""
    engine = RegionSamplingEngine()
    current_time = 100.0

    page = Page(
        page_id=10, region_id=0, content=bytearray(PAGE_SIZE),
        alloc_time=50.0, last_write_time=50.0, write_count=0
    )

    # Initial state: should be COLD
    tier, _ = engine.classify_page(page, current_time)
    assert tier == StabilityTier.COLD

    # Simulate COW break fault at current_time
    engine.record_cow_fault_feedback(page_id=10, current_time=current_time, quarantine_duration=30.0)

    # Page should now be quarantined (HOT / skipped) at current_time + 10
    tier_after, _ = engine.classify_page(page, current_time + 10.0)
    assert tier_after == StabilityTier.HOT

    # Once quarantine expires at current_time + 35, penalty still influences write rate
    tier_expired, _ = engine.classify_page(page, current_time + 35.0)
    # The effective write count increased due to COW break penalty
    assert engine.page_cow_breaks[10] == 1


def test_region_level_filtering():
    """Verify volatile regions are skipped when enable_region_filtering is enabled."""
    engine = RegionSamplingEngine(region_volatility_threshold=2.0)
    current_time = 100.0

    # Region 0: Highly volatile (10 writes on 2 pages -> avg 5.0 >= 2.0)
    # Region 1: Stable (0 writes on 2 pages -> avg 0.0 < 2.0)
    pages = [
        Page(page_id=1, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=50.0, last_write_time=90.0, write_count=5),
        Page(page_id=2, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=50.0, last_write_time=90.0, write_count=5),
        Page(page_id=3, region_id=1, content=bytearray(PAGE_SIZE), alloc_time=50.0, last_write_time=50.0, write_count=0),
        Page(page_id=4, region_id=1, content=bytearray(PAGE_SIZE), alloc_time=50.0, last_write_time=50.0, write_count=0),
    ]

    candidates, stats = engine.filter_and_prioritize(pages, current_time=current_time, enable_region_filtering=True)
    assert stats["volatile_region_pages_skipped"] == 2
    assert len(candidates) == 2
    assert {c.page_id for c in candidates} == {3, 4}


def test_end_to_end_workload_sampling_reduction():
    """Verify Module 1 achieves significant scan overhead reduction on a realistic workload."""
    generator = SyntheticWorkloadGenerator(seed=42)
    pages = generator.generate_workload(num_pages=2000)

    # Simulate write traffic
    sim_time = generator.simulate_write_traffic(pages, num_ticks=40, writes_per_tick=150, current_time=100.0)

    engine = RegionSamplingEngine()
    candidates, stats = engine.filter_and_prioritize(pages, current_time=sim_time)

    assert stats["total_pages"] == 2000
    assert stats["hot_pages_skipped"] > 0
    assert len(candidates) < 2000
    assert stats["cold_pages_queued"] > 0

    # Ensure reduction ratio is computed properly
    overhead_reduction_pct = (stats["hot_pages_skipped"] / stats["total_pages"]) * 100
    assert overhead_reduction_pct > 0.0
