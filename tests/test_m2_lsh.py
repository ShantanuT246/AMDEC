"""
Unit tests for Module 2: Feature Extraction & MinHash + LSH Similarity Clustering Engine (src/modules/m2_lsh_clustering.py).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine, SimilarityCluster, UnionFind
from src.workload.page_generator import Page, SyntheticWorkloadGenerator, PAGE_SIZE


def test_union_find_disjoint_set():
    """Verify Union-Find initialization, union, find, and cluster grouping."""
    uf = UnionFind([1, 2, 3, 4, 5, 6])

    assert uf.find(1) == 1
    assert uf.find(2) == 2

    # Union (1, 2) and (2, 3) -> component {1, 2, 3}
    assert uf.union(1, 2) is True
    assert uf.union(2, 3) is True
    assert uf.union(1, 3) is False  # Already connected

    # Union (4, 5) -> component {4, 5}
    assert uf.union(4, 5) is True

    clusters = uf.get_clusters()
    # Should have 3 components: {1, 2, 3}, {4, 5}, {6}
    component_sizes = sorted([len(members) for members in clusters.values()])
    assert component_sizes == [1, 2, 3]

    # Verify roots
    assert uf.find(1) == uf.find(3)
    assert uf.find(4) == uf.find(5)
    assert uf.find(1) != uf.find(4)


def test_large_bucket_exact_hash_grouping():
    """
    Regression test: A workload of 200 identical zero-pages must form a single cluster.
    Previously, the chaining strategy could fragment this into multiple clusters if
    any consecutive Jaccard verification link failed.
    """
    engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)

    # 200 byte-for-byte identical zero-pages
    num_zero_pages = 200
    zero_content = bytearray(PAGE_SIZE)
    zero_pages = [
        Page(
            page_id=i,
            region_id=0,
            content=bytearray(zero_content),
            alloc_time=10.0,
            last_write_time=10.0,
        )
        for i in range(num_zero_pages)
    ]

    clusters, stats = engine.cluster_pages(zero_pages, verify_similarity=True)

    # All 200 identical pages MUST form a single cluster — no fragmentation allowed
    assert len(clusters) == 1, (
        f"Identical zero-pages split into {len(clusters)} clusters! Exact-hash grouping failed."
    )
    assert len(clusters[0].page_ids) == num_zero_pages


def test_page_chunking():
    """Verify 4096-byte page is chunked into expected count and size of sub-blocks."""
    engine_128 = MinHashLSHClusteringEngine(block_size=128)
    content = bytearray(PAGE_SIZE)
    chunks_128 = engine_128.chunk_page(content)
    assert len(chunks_128) == 32
    assert all(len(c) == 130 for c in chunks_128)  # 128 bytes + 2 bytes index

    engine_64 = MinHashLSHClusteringEngine(block_size=64)
    chunks_64 = engine_64.chunk_page(content)
    assert len(chunks_64) == 64
    assert all(len(c) == 66 for c in chunks_64)  # 64 bytes + 2 bytes index


def test_minhash_similarity_estimation():
    """Verify MinHash accurately detects high similarity on near-duplicates and low similarity on distinct pages."""
    generator = SyntheticWorkloadGenerator(seed=42)
    engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=64)

    base_content = generator._create_template_page_content("heap_struct")
    # Mutate ~0.4% of words (2 words modified -> ~93% chunk similarity)
    near_content = generator._mutate_page_content(base_content, mutation_rate=0.004)
    # Distinct unique page
    unique_content = generator._create_unique_page_content()

    mh_base = engine.compute_minhash(base_content)
    mh_near = engine.compute_minhash(near_content)
    mh_unique = engine.compute_minhash(unique_content)

    sim_near = mh_base.jaccard(mh_near)
    sim_unique = mh_base.jaccard(mh_unique)

    assert sim_near >= 0.70, f"Expected high similarity for near-duplicate, got {sim_near}"
    assert sim_unique <= 0.20, f"Expected low similarity for unique page, got {sim_unique}"


def test_lsh_banding_and_clustering():
    """Verify LSH correctly clusters near-duplicate pages while keeping unique pages separate."""
    generator = SyntheticWorkloadGenerator(seed=100)
    engine = MinHashLSHClusteringEngine(
        block_size=128,
        num_hashes=32,
        num_bands=8,
        similarity_threshold=0.65,
    )

    base1 = generator._create_template_page_content("heap_struct")
    base2 = generator._create_template_page_content("pointer_table")

    # Cluster A: 3 pages from base1 (1 base + 2 near duplicates with 1-2 mutated fields)
    p1 = Page(page_id=1, region_id=0, content=bytearray(base1), alloc_time=10.0, last_write_time=10.0)
    p2 = Page(page_id=2, region_id=0, content=generator._mutate_page_content(base1, 0.004), alloc_time=10.0, last_write_time=10.0)
    p3 = Page(page_id=3, region_id=0, content=generator._mutate_page_content(base1, 0.006), alloc_time=10.0, last_write_time=10.0)

    # Cluster B: 2 pages from base2 (1 base + 1 near duplicate)
    p4 = Page(page_id=4, region_id=1, content=bytearray(base2), alloc_time=10.0, last_write_time=10.0)
    p5 = Page(page_id=5, region_id=1, content=generator._mutate_page_content(base2, 0.004), alloc_time=10.0, last_write_time=10.0)

    # Unique Page C: 1 unique page
    p6 = Page(page_id=6, region_id=2, content=generator._create_unique_page_content(), alloc_time=10.0, last_write_time=10.0)

    pages = [p1, p2, p3, p4, p5, p6]
    clusters, stats = engine.cluster_pages(pages)

    assert stats["total_pages_in"] == 6
    assert stats["num_multi_page_clusters"] == 2
    assert stats["clustered_pages_count"] == 5
    assert stats["unclustered_pages_count"] == 1

    # Extract member sets
    cluster_member_sets = [set(c.page_ids) for c in clusters]
    assert {1, 2, 3} in cluster_member_sets
    assert {4, 5} in cluster_member_sets
    assert 6 not in [pid for c in clusters for pid in c.page_ids]


def test_m1_m2_pipeline_integration():
    """Verify integrated end-to-end flow: Workload -> M1 Sampling -> M2 LSH Clustering."""
    generator = SyntheticWorkloadGenerator(seed=42)
    # Generate 500 pages
    pages = generator.generate_workload(num_pages=500, cluster_size_range=(4, 16))
    sim_time = generator.simulate_write_traffic(pages, num_ticks=20, writes_per_tick=50, current_time=100.0)

    # 1. Module 1 Sampling
    m1_engine = RegionSamplingEngine()
    candidates, m1_stats = m1_engine.filter_and_prioritize(pages, current_time=sim_time)

    assert len(candidates) > 0
    assert len(candidates) <= 500

    # 2. Extract candidate Page objects for Module 2
    candidate_pages = [c.page for c in candidates]

    # 3. Module 2 Clustering
    m2_engine = MinHashLSHClusteringEngine(block_size=128, num_hashes=32, num_bands=8, similarity_threshold=0.70)
    clusters, m2_stats = m2_engine.cluster_pages(candidate_pages)

    assert m2_stats["candidate_reduction_pct"] > 80.0, "LSH should eliminate >80% of theoretical pairwise comparisons on small dataset"
    assert m2_stats["num_multi_page_clusters"] > 0
    assert m2_stats["clustering_time_ms"] > 0.0

    # Verify cluster integrity
    for cluster in clusters:
        assert cluster.size >= 2
        assert cluster.avg_intra_similarity >= 0.50
