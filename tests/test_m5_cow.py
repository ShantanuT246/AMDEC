"""
Unit tests for Module 5: Simulated COW Fault Handler & Reconstruction Engine (src/modules/m5_cow.py).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import Page, SyntheticWorkloadGenerator, PAGE_SIZE
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import SimilarityCluster
from src.modules.m3_delta_storage import BaseDeltaStorageManager
from src.modules.m5_cow import PageState, SimulatedCOWHandler


def test_page_table_population():
    """Verify page table correctly registers BASE_PAGE, DELTA_PAGE, and FULL_PAGE entries."""
    generator = SyntheticWorkloadGenerator(seed=42)
    m3_manager = BaseDeltaStorageManager()
    cow_handler = SimulatedCOWHandler(delta_manager=m3_manager)

    base_content = generator._create_template_page_content("heap_struct")
    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=generator._mutate_page_content(base_content, 0.005), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=2, region_id=0, content=generator._create_unique_page_content(), alloc_time=10.0, last_write_time=10.0),
    ]

    cluster = SimilarityCluster(cluster_id=0, page_ids=[0, 1])
    compressed_clusters, _ = m3_manager.compress_workload([cluster], pages)

    cow_handler.populate_from_compressed_clusters(compressed_clusters, pages)

    summary = cow_handler.get_page_table_summary()
    assert summary["base_page"] == 1
    assert summary["delta_page"] == 1
    assert summary["full_page"] == 1
    assert summary["total"] == 3

    assert cow_handler.page_table[0].state == PageState.BASE_PAGE
    assert cow_handler.page_table[1].state == PageState.DELTA_PAGE
    assert cow_handler.page_table[2].state == PageState.FULL_PAGE


def test_read_path_reconstruction():
    """Verify transparent read path reconstructs compressed DELTA_PAGE with byte-for-byte exactness."""
    generator = SyntheticWorkloadGenerator(seed=42)
    m3_manager = BaseDeltaStorageManager()
    cow_handler = SimulatedCOWHandler(delta_manager=m3_manager)

    base_content = generator._create_template_page_content("heap_struct")
    target_content = generator._mutate_page_content(base_content, 0.005)

    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=bytearray(target_content), alloc_time=10.0, last_write_time=10.0),
    ]

    cluster = SimilarityCluster(cluster_id=0, page_ids=[0, 1])
    compressed_clusters, _ = m3_manager.compress_workload([cluster], pages)
    cow_handler.populate_from_compressed_clusters(compressed_clusters, pages)

    # Read base page
    read_base = cow_handler.read_page(0)
    assert read_base == base_content

    # Read delta page (triggers reconstruction)
    read_delta = cow_handler.read_page(1)
    assert read_delta == target_content

    lat_metrics = cow_handler.get_latency_metrics()
    assert lat_metrics["count"] == 1
    assert lat_metrics["mean_us"] > 0.0


def test_write_fault_cow_break_and_feedback():
    """Verify write fault to DELTA_PAGE allocates private FULL_PAGE and triggers M1 feedback."""
    generator = SyntheticWorkloadGenerator(seed=42)
    m1_engine = RegionSamplingEngine()
    m3_manager = BaseDeltaStorageManager()
    cow_handler = SimulatedCOWHandler(delta_manager=m3_manager, sampling_engine=m1_engine)

    base_content = generator._create_template_page_content("heap_struct")
    target_content = generator._mutate_page_content(base_content, 0.005)

    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=bytearray(target_content), alloc_time=10.0, last_write_time=10.0),
    ]

    cluster = SimilarityCluster(cluster_id=0, page_ids=[0, 1])
    compressed_clusters, _ = m3_manager.compress_workload([cluster], pages)
    cow_handler.populate_from_compressed_clusters(compressed_clusters, pages)

    assert cow_handler.page_table[1].state == PageState.DELTA_PAGE

    # Trigger write fault on delta page 1
    new_data = b"OVERWRITTEN_DATA"
    modified_content = cow_handler.handle_write_fault(page_id=1, offset=32, data=new_data, current_time=150.0)

    # State should now be FULL_PAGE
    assert cow_handler.page_table[1].state == PageState.FULL_PAGE
    assert cow_handler.page_table[1].content[32:32+len(new_data)] == new_data
    assert cow_handler.page_table[1].cow_fault_count == 1

    # Base page should remain intact
    assert cow_handler.read_page(0) == base_content

    # M1 feedback should record COW break
    assert m1_engine.page_cow_breaks[1] == 1


def test_write_to_base_page_does_not_corrupt_deltas():
    """Verify writing to the base page breaks COW for the base without corrupting dependent delta reconstructions."""
    generator = SyntheticWorkloadGenerator(seed=42)
    m3_manager = BaseDeltaStorageManager()
    cow_handler = SimulatedCOWHandler(delta_manager=m3_manager)

    base_content = generator._create_template_page_content("heap_struct")
    target_content = generator._mutate_page_content(base_content, 0.005)

    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=bytearray(target_content), alloc_time=10.0, last_write_time=10.0),
    ]

    cluster = SimilarityCluster(cluster_id=0, page_ids=[0, 1])
    compressed_clusters, _ = m3_manager.compress_workload([cluster], pages)
    cow_handler.populate_from_compressed_clusters(compressed_clusters, pages)

    # 1. Modify Base Page (Page 0)
    cow_handler.handle_write_fault(page_id=0, offset=0, data=b"CORRUPT_BASE_HEADER", current_time=100.0)
    assert cow_handler.page_table[0].state == PageState.FULL_PAGE
    assert cow_handler.read_page(0)[:19] == b"CORRUPT_BASE_HEADER"

    # 2. Verify Delta Page (Page 1) still reconstructs to its original target content correctly
    read_delta = cow_handler.read_page(1)
    assert read_delta == target_content, "Delta reconstruction must not be corrupted by base page writes!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
