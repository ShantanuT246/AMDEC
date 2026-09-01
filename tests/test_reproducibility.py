"""
Tests for AMDEC Experimental Reproducibility & Deterministic Randomness.

Verifies:
1. Deterministic bitwise workload repeatability under identical seeds.
2. Seed variance (different seeds yield distinct but valid workloads).
3. End-to-end pipeline invariance for identical seed + config.
"""

import pytest
from src.common.config import AMDECConfig
from src.workload.page_generator import SyntheticWorkloadGenerator
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m2_lsh_clustering import MinHashLSHClusteringEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager


def test_workload_seed_determinism():
    """Identical seeds must produce byte-for-byte identical pages and hash digests."""
    gen1 = SyntheticWorkloadGenerator(seed=42)
    pages1 = gen1.generate_workload(num_pages=500)

    gen2 = SyntheticWorkloadGenerator(seed=42)
    pages2 = gen2.generate_workload(num_pages=500)

    assert len(pages1) == len(pages2)
    for p1, p2 in zip(pages1, pages2):
        assert p1.page_id == p2.page_id
        assert p1.region_id == p2.region_id
        assert p1.hash_exact == p2.hash_exact
        assert p1.content == p2.content


def test_workload_seed_divergence():
    """Different seeds must produce distinct workloads with differing page contents."""
    gen1 = SyntheticWorkloadGenerator(seed=42)
    pages1 = gen1.generate_workload(num_pages=200)

    gen2 = SyntheticWorkloadGenerator(seed=999)
    pages2 = gen2.generate_workload(num_pages=200)

    # Some pages should differ in exact hash
    hashes1 = {p.hash_exact for p in pages1}
    hashes2 = {p.hash_exact for p in pages2}
    assert hashes1 != hashes2


def test_pipeline_end_to_end_determinism():
    """Same seed and configuration must produce identical cluster counts and memory savings."""
    def run_pipeline(seed: int):
        gen = SyntheticWorkloadGenerator(seed=seed)
        pages = gen.generate_workload(num_pages=1000)
        sim_time = gen.simulate_write_traffic(pages, num_ticks=20, writes_per_tick=50, current_time=100.0)

        m1 = RegionSamplingEngine()
        cands, m1_stats = m1.filter_and_prioritize(pages, current_time=sim_time)
        m2 = MinHashLSHClusteringEngine(seed=seed)
        clusters, m2_stats = m2.cluster_pages([c.page for c in cands])
        m3 = BaseDeltaStorageManager(seed=seed)
        _, m3_metrics = m3.compress_workload(clusters, pages)
        return {
            "skipped": m1_stats["hot_pages_skipped"],
            "clusters": len(clusters),
            "savings": m3_metrics["logical_memory_savings_pct"],
            "base_pages": m3_metrics["total_base_pages"],
            "delta_pages": m3_metrics["total_delta_pages"],
        }

    res_run1 = run_pipeline(seed=42)
    res_run2 = run_pipeline(seed=42)

    assert res_run1 == res_run2, "Two pipeline executions with seed=42 produced divergent metrics!"


def test_config_json_roundtrip(tmp_path):
    """Configuration dataclass must serialize to and load from JSON identically."""
    cfg = AMDECConfig(seed=123)
    json_file = str(tmp_path / "test_config.json")
    cfg.save_json(json_file)

    loaded_cfg = AMDECConfig.load_json(json_file)
    assert loaded_cfg.seed == 123
    assert loaded_cfg.workload.num_pages == 10000
    assert loaded_cfg.m2_clustering.similarity_threshold == 0.70
