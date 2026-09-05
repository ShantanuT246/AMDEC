"""
Unit tests for Module 3: Base-Delta Storage & Adaptive Patching Manager (src/modules/m3_delta_storage.py).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.page_generator import Page, SyntheticWorkloadGenerator, PAGE_SIZE
from src.modules.m2_lsh_clustering import SimilarityCluster
from src.modules.m3_delta_storage import BaseDeltaStorageManager, DeltaRecord, DeltaSpan


def test_xor_diff_and_rle_spans():
    """Verify XOR diff calculation and RLE non-zero span encoding."""
    manager = BaseDeltaStorageManager()

    base = bytearray(PAGE_SIZE)
    target = bytearray(PAGE_SIZE)

    # Modify 2 separate spans
    target[100:108] = b"AAAAAAAA"  # Span 1: offset 100, length 8
    target[500:516] = b"BBBBBBBBCCCCCCCC"  # Span 2: offset 500, length 16

    xor_diff = manager.compute_xor_diff(target, base)
    spans = manager.encode_rle_spans(xor_diff)

    assert len(spans) == 2
    assert spans[0].offset == 100
    assert spans[0].length == 8
    assert spans[0].data == b"AAAAAAAA"

    assert spans[1].offset == 500
    assert spans[1].length == 16
    assert spans[1].data == b"BBBBBBBBCCCCCCCC"


def test_round_trip_reconstruction_fidelity():
    """Verify 100% byte-for-byte exact round-trip page reconstruction."""
    generator = SyntheticWorkloadGenerator(seed=42)
    manager = BaseDeltaStorageManager()

    base_content = generator._create_template_page_content("heap_struct")
    target_content = generator._mutate_page_content(base_content, mutation_rate=0.01)

    xor_diff = manager.compute_xor_diff(target_content, base_content)
    spans = manager.encode_rle_spans(xor_diff)

    delta = DeltaRecord(
        page_id=2,
        base_page_id=1,
        spans=spans,
        raw_delta_bytes=sum(len(s.data) for s in spans),
    )

    reconstructed = manager.reconstruct_page(base_content, delta)
    assert reconstructed == target_content
    assert len(reconstructed) == PAGE_SIZE


def test_metadata_aware_storage_accounting():
    """Verify delta size accurately includes base reference (4B), span headers (4B/span), and record headers (2B)."""
    delta = DeltaRecord(
        page_id=2,
        base_page_id=1,
        spans=[
            DeltaSpan(offset=10, length=8, data=b"12345678"),      # 4B header + 8B data = 12B
            DeltaSpan(offset=100, length=16, data=b"ABCDEFGH12345678"),  # 4B header + 16B data = 20B
        ],
        raw_delta_bytes=24,
    )
    # Total = 4B (base_ref) + 2B (span_count) + 12B (span 1) + 20B (span 2) = 38 bytes
    assert delta.total_storage_bytes == 38


def test_adaptive_threshold_fallback():
    """Verify pages with large deltas exceeding delta_max fallback to unmerged full pages."""
    manager = BaseDeltaStorageManager(max_delta_ratio=0.10)  # max 409 bytes

    base = Page(page_id=1, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=10.0, last_write_time=10.0)

    # Small delta page (< 409 bytes)
    small_p = Page(page_id=2, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=10.0, last_write_time=10.0)
    small_p.content[10:30] = b"X" * 20

    # Large delta page (> 409 bytes)
    large_p = Page(page_id=3, region_id=0, content=bytearray(PAGE_SIZE), alloc_time=10.0, last_write_time=10.0)
    large_p.content[100:600] = b"Y" * 500

    cluster = SimilarityCluster(cluster_id=0, page_ids=[1, 2, 3])
    page_map = {1: base, 2: small_p, 3: large_p}

    cc = manager.compress_cluster(cluster, page_map)

    assert cc.base_page_id == 1
    assert 2 in cc.deltas
    assert 3 in cc.unmerged_full_pages  # Exceeded max delta ratio -> fallback to full


def test_end_to_end_cluster_compression_savings():
    """Verify end-to-end compression achieves substantial logical memory reduction on near-duplicate clusters."""
    generator = SyntheticWorkloadGenerator(seed=42)
    manager = BaseDeltaStorageManager()

    base_content = generator._create_template_page_content("heap_struct")
    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0)
    ]
    for i in range(1, 10):
        # Mutate 1-2 words
        mutated = generator._mutate_page_content(base_content, mutation_rate=0.005)
        pages.append(Page(page_id=i, region_id=0, content=mutated, alloc_time=10.0, last_write_time=10.0))

    cluster = SimilarityCluster(cluster_id=0, page_ids=list(range(10)))
    compressed_clusters, metrics = manager.compress_workload([cluster], pages)

    # 10 uncompressed pages = 40,960 bytes
    # AMDEC storage = 1 base (4096B) + 9 compact deltas (~400-800B total) ~= 4,800B
    # Savings should exceed 75%
    assert metrics["logical_memory_savings_pct"] >= 75.0
    assert metrics["compression_ratio"] >= 4.0
    assert metrics["total_base_pages"] == 1
    assert metrics["total_delta_pages"] == 9


def test_exact_hash_zero_span_delta():
    """Phase A: Identical pages should produce a zero-span delta (no XOR needed)."""
    manager = BaseDeltaStorageManager()
    base_content = bytearray(b"X" * PAGE_SIZE)
    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
    ]
    cluster = SimilarityCluster(cluster_id=0, page_ids=[0, 1])
    cc = manager.compress_cluster(cluster, {p.page_id: p for p in pages})

    assert 1 in cc.deltas
    delta = cc.deltas[1]
    assert len(delta.spans) == 0, "Exact duplicate must produce zero-span delta"
    assert delta.total_storage_bytes == 6  # 4B base_ref + 2B span_count = 6B only


def test_secondary_base_fallback_for_transitive_divergence():
    """M3 secondary-base fallback: transitive divergence creates a secondary base to capture deltas."""
    manager = BaseDeltaStorageManager(max_delta_ratio=0.10)  # Tight 10% threshold (409 bytes max delta)

    # base: all zeros
    # p1: similar to base (40B diff <= 409B -> primary delta)
    # p2: large diff from base (500B > 409B)
    # p3: large diff from base (508B > 409B), but only 8B diff from p2
    base_content = bytearray(PAGE_SIZE)
    p1_content = bytearray(PAGE_SIZE); p1_content[0:40] = b"A" * 40
    p2_content = bytearray(PAGE_SIZE); p2_content[0:40] = b"A" * 40; p2_content[40:500] = b"Z" * 460
    p3_content = bytearray(PAGE_SIZE); p3_content[0:40] = b"A" * 40; p3_content[40:500] = b"Z" * 460; p3_content[500:508] = b"W" * 8

    pages = [
        Page(page_id=0, region_id=0, content=base_content, alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=p1_content,   alloc_time=10.0, last_write_time=10.0),
        Page(page_id=2, region_id=0, content=p2_content,   alloc_time=10.0, last_write_time=10.0),
        Page(page_id=3, region_id=0, content=p3_content,   alloc_time=10.0, last_write_time=10.0),
    ]
    cluster = SimilarityCluster(cluster_id=0, page_ids=[0, 1, 2, 3])
    cc = manager.compress_cluster(cluster, {p.page_id: p for p in pages})

    # Primary medoid is page 1 (min total pairwise byte distance across cluster)
    assert cc.base_page_id == 1
    # Page 0 matches primary base (40B diff <= 409B)
    assert 0 in cc.deltas
    assert cc.deltas[0].base_page_id == 1

    # Pages 2 & 3 diverged from primary base (>409B), but match each other:
    # One becomes secondary base, the other is delta-compressed against it
    assert len(cc.secondary_bases) == 1
    sec_base_id = list(cc.secondary_bases.keys())[0]
    sec_delta_id = 3 if sec_base_id == 2 else 2

    assert sec_delta_id in cc.deltas
    assert cc.deltas[sec_delta_id].base_page_id == sec_base_id
    assert cc.deltas[sec_delta_id].total_storage_bytes <= manager.max_delta_bytes
    assert len(cc.unmerged_full_pages) == 0, "Secondary-base fallback should prevent unmerged full pages"

    # Verify reconstruction of secondary delta
    reconstructed = manager.reconstruct_page(cc.secondary_bases[sec_base_id], cc.deltas[sec_delta_id])
    expected_content = p3_content if sec_delta_id == 3 else p2_content
    assert reconstructed == expected_content


def test_global_base_deduplication_registry():
    """Identical base pages across different clusters should be counted only once in storage."""
    manager = BaseDeltaStorageManager()
    base_content = bytearray(b"TEMPLATE" * (PAGE_SIZE // 8))

    # Two clusters that happen to end up with the same base page content
    pages = [
        Page(page_id=0, region_id=0, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=1, region_id=0, content=bytearray(base_content[:PAGE_SIZE - 10] + b"A" * 10),
             alloc_time=10.0, last_write_time=10.0),
        Page(page_id=2, region_id=1, content=bytearray(base_content), alloc_time=10.0, last_write_time=10.0),
        Page(page_id=3, region_id=1, content=bytearray(base_content[:PAGE_SIZE - 10] + b"B" * 10),
             alloc_time=10.0, last_write_time=10.0),
    ]
    # Cluster 0 medoid = page 0 (base_content), Cluster 1 medoid = page 2 (base_content)
    clusters = [
        SimilarityCluster(cluster_id=0, page_ids=[0, 1]),
        SimilarityCluster(cluster_id=1, page_ids=[2, 3]),
    ]
    _, metrics = manager.compress_workload(clusters, pages)

    # The two base pages are identical → second one should be deduped
    assert metrics["deduped_base_pages"] == 1
    # Storage should be smaller than if we counted both bases
    assert metrics["amdec_storage_bytes"] < 2 * PAGE_SIZE + 2 * PAGE_SIZE


if __name__ == "__main__":
    pytest.main(["-v", __file__])
