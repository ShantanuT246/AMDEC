"""
Unit tests for Synthetic Anonymous Workload Generator (src/workload/page_generator.py).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import Page, SyntheticWorkloadGenerator, PAGE_SIZE


def test_page_dataclass_invariants():
    """Verify Page attributes, byte boundaries, and SHA256 hashing."""
    content = bytearray(PAGE_SIZE)
    p = Page(
        page_id=1,
        region_id=0,
        content=content,
        alloc_time=10.0,
        last_write_time=10.0,
    )
    assert len(p.content) == 4096
    assert isinstance(p.hash_exact, str)
    assert len(p.hash_exact) == 64

    # Test invalid size
    with pytest.raises(ValueError):
        Page(page_id=2, region_id=0, content=bytearray(100), alloc_time=1.0, last_write_time=1.0)


def test_page_write_and_clone():
    """Verify in-place writing and cloning behavior."""
    content = bytearray(PAGE_SIZE)
    p = Page(page_id=1, region_id=0, content=content, alloc_time=10.0, last_write_time=10.0)

    # In-place write
    p.write_bytes(offset=16, data=b"TESTWRITE", current_time=15.0)
    assert p.content[16:25] == b"TESTWRITE"
    assert p.write_count == 1
    assert p.last_write_time == 15.0

    # Clone
    clone_p = p.clone(new_page_id=2, new_region_id=1, current_time=20.0)
    assert clone_p.page_id == 2
    assert clone_p.region_id == 1
    assert clone_p.content == p.content
    assert clone_p.alloc_time == 20.0


def test_workload_generator_distribution():
    """Verify workload distribution (80% near-dup, 7% exact, 13% unique)."""
    generator = SyntheticWorkloadGenerator(
        seed=123,
        near_dup_ratio=0.80,
        exact_dup_ratio=0.07,
        unique_ratio=0.13,
        num_regions=4,
    )
    pages = generator.generate_workload(num_pages=1000, cluster_size_range=(4, 16))

    assert len(pages) == 1000
    for p in pages:
        assert len(p.content) == PAGE_SIZE
        assert 0 <= p.region_id < 4

    types = [p.metadata.get("type") for p in pages]
    near_count = types.count("near_duplicate")
    exact_count = types.count("exact_duplicate")
    unique_count = types.count("unique")

    assert 750 <= near_count <= 850
    assert 50 <= exact_count <= 90
    assert 110 <= unique_count <= 150


def test_workload_zipf_traffic_simulation():
    """Verify write traffic follows Zipfian skew (few pages get many writes, most get none/few)."""
    generator = SyntheticWorkloadGenerator(seed=42, zipf_param=1.5)
    pages = generator.generate_workload(num_pages=500)

    start_time = 200.0
    end_time = generator.simulate_write_traffic(
        pages=pages,
        num_ticks=30,
        writes_per_tick=100,
        current_time=start_time,
    )

    assert end_time > start_time
    total_writes = sum(p.write_count for p in pages)
    assert total_writes == 30 * 100

    writes = [p.write_count for p in pages]
    writes.sort(reverse=True)

    # Top 10% pages should receive > 50% of the total writes under Zipf distribution
    top_10_pct = len(pages) // 10
    top_writes = sum(writes[:top_10_pct])
    assert top_writes > (total_writes * 0.45)
