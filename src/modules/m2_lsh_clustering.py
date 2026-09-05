"""
Module 2 — Feature Extraction & MinHash + LSH Similarity Clustering Engine.

Processes candidate memory pages, chunks them into sub-blocks (64B/128B),
computes MinHash signatures, indexes them across LSH bands, verifies candidate pairs,
and merges them into similarity clusters using Union-Find.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from datasketch import MinHash
import numpy as np

from src.workload.page_generator import Page, PAGE_SIZE


@dataclass
class SimilarityCluster:
    """Represents a cluster of near-duplicate memory pages."""
    cluster_id: int
    page_ids: List[int]
    base_page_id: Optional[int] = None
    avg_intra_similarity: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.page_ids)


class UnionFind:
    """High-performance disjoint-set data structure with path compression and rank optimization."""

    def __init__(self, elements: Optional[List[int]] = None):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
        if elements:
            for elem in elements:
                self.parent[elem] = elem
                self.rank[elem] = 0

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.add(x)
            return x
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

    def get_clusters(self) -> Dict[int, List[int]]:
        """Groups all elements by their root parent."""
        clusters: Dict[int, List[int]] = defaultdict(list)
        for elem in self.parent:
            root = self.find(elem)
            clusters[root].append(elem)
        return dict(clusters)


class MinHashLSHClusteringEngine:
    """
    Feature Extraction and Locality-Sensitive Hashing (LSH) Clustering Engine (Module 2).

    Parameters:
        block_size: Size of sub-blocks in bytes (e.g. 64B, 128B, 256B).
        num_hashes: Number of MinHash permutations (K).
        num_bands: Number of LSH bands (b). Note: num_hashes must be divisible by num_bands.
        similarity_threshold: Jaccard similarity cutoff for merging candidate pairs.
        seed: Random seed for hash permutations.
    """

    def __init__(
        self,
        block_size: int = 128,
        num_hashes: int = 32,
        num_bands: int = 8,
        similarity_threshold: float = 0.70,
        seed: int = 42,
    ):
        if PAGE_SIZE % block_size != 0:
            raise ValueError(f"PAGE_SIZE ({PAGE_SIZE}) must be divisible by block_size ({block_size})")
        if num_hashes % num_bands != 0:
            raise ValueError(f"num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands})")

        self.block_size = block_size
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def chunk_page(self, content: bytearray) -> List[bytes]:
        """
        Splits a 4096-byte page into fixed-size byte sub-blocks.
        Prefixed with positional offset index to preserve structural page layout.
        """
        chunks: List[bytes] = []
        for idx, offset in enumerate(range(0, PAGE_SIZE, self.block_size)):
            # Prepend 2-byte chunk index to make sub-blocks position-aware
            chunk_data = struct.pack("<H", idx) + bytes(content[offset:offset + self.block_size])
            chunks.append(chunk_data)
        return chunks

    def compute_minhash(self, content: bytearray) -> MinHash:
        """Computes a MinHash signature from the page's position-aware sub-block chunks."""
        m = MinHash(num_perm=self.num_hashes, seed=self.seed)
        for chunk in self.chunk_page(content):
            m.update(chunk)
        return m

    def extract_band_buckets(self, minhash_obj: MinHash) -> List[int]:
        """
        Splits MinHash digest into bands and returns integer bucket hashes for each band.
        """
        digest = minhash_obj.digest()  # array of uint64 or uint32
        band_hashes: List[int] = []

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = digest[start:end]
            # Convert band slice to a 64-bit integer hash
            band_bytes = band_slice.tobytes()
            # Fast 64-bit hash of band bytes combined with band index
            h = int.from_bytes(hashlib.md5(struct.pack("<I", b) + band_bytes).digest()[:8], "little")
            band_hashes.append(h)

        return band_hashes

    def cluster_pages(
        self,
        pages: List[Page],
        verify_similarity: bool = True,
    ) -> Tuple[List[SimilarityCluster], Dict[str, Any]]:
        """
        Executes the complete M2 pipeline:
        1. Feature extraction & MinHash signatures
        2. LSH band indexing
        3. Candidate pair extraction
        4. Jaccard verification
        5. Union-Find cluster assembly

        Returns:
            Tuple[List[SimilarityCluster], clustering_stats]
        """
        t0 = time.perf_counter()
        n_pages = len(pages)
        if n_pages == 0:
            return [], {"total_pages_in": 0, "clustering_time_ms": 0.0}

        # 1. Compute MinHash signatures for all input pages
        page_dict: Dict[int, Page] = {p.page_id: p for p in pages}
        page_minhashes: Dict[int, MinHash] = {}
        page_content_map: Dict[int, bytearray] = {}
        for p in pages:
            page_minhashes[p.page_id] = self.compute_minhash(p.content)
            page_content_map[p.page_id] = p.content

        # 2. Index pages into LSH Band Buckets: (band_idx, bucket_hash) -> List[page_id]
        lsh_tables: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for p_id, mh in page_minhashes.items():
            band_hashes = self.extract_band_buckets(mh)
            for band_idx, b_hash in enumerate(band_hashes):
                lsh_tables[(band_idx, b_hash)].append(p_id)

        # 3. Extract Candidate Pairs from buckets containing >= 2 pages
        candidate_pairs: Set[Tuple[int, int]] = set()
        for bucket_pages in lsh_tables.values():
            b_len = len(bucket_pages)
            if b_len > 1:
                if b_len <= 64:
                    for i in range(b_len):
                        p1 = bucket_pages[i]
                        for j in range(i + 1, b_len):
                            p2 = bucket_pages[j]
                            candidate_pairs.add((min(p1, p2), max(p1, p2)))
                else:
                    # Large-bucket strategy for massive uniform workloads (e.g. zero-pages)
                    # Step 1: Group exact duplicates via SHA-256 content hash.
                    # This guarantees exact-duplicate pages are always in the same
                    # Union-Find component regardless of Jaccard threshold, preventing
                    # artificial cluster fragmentation on a single failed verification link.
                    hash_groups: Dict[str, List[int]] = {}
                    for pid in bucket_pages:
                        page_content = page_content_map.get(pid)
                        if page_content is not None:
                            h = hashlib.sha256(bytes(page_content)).hexdigest()
                        else:
                            h = str(pid)
                        hash_groups.setdefault(h, []).append(pid)

                    # Step 2: Within each exact-hash group, chain consecutive pairs
                    # (guaranteed J=1.0, never fails Jaccard verification)
                    group_representatives: List[int] = []
                    for group_ids in hash_groups.values():
                        for i in range(len(group_ids) - 1):
                            p1, p2 = group_ids[i], group_ids[i + 1]
                            candidate_pairs.add((min(p1, p2), max(p1, p2)))
                        group_representatives.append(group_ids[0])  # One rep per unique hash

                    # Step 3: Connect representatives and cross-sample between them
                    n_reps = len(group_representatives)
                    if n_reps > 1:
                        if n_reps <= 64:
                            for i in range(n_reps):
                                for j in range(i + 1, n_reps):
                                    p1, p2 = group_representatives[i], group_representatives[j]
                                    candidate_pairs.add((min(p1, p2), max(p1, p2)))
                        else:
                            for i in range(n_reps - 1):
                                p1, p2 = group_representatives[i], group_representatives[i + 1]
                                candidate_pairs.add((min(p1, p2), max(p1, p2)))
                            # Sample additional cross pairs across representatives
                            sample_k = min(n_reps * 2, 500)
                            for _ in range(sample_k):
                                i, j = self.rng.choice(n_reps, size=2, replace=False)
                                p1, p2 = group_representatives[i], group_representatives[j]
                                candidate_pairs.add((min(p1, p2), max(p1, p2)))

        # Theoretical brute-force comparisons
        total_possible_pairs = (n_pages * (n_pages - 1)) // 2
        candidate_count = len(candidate_pairs)
        candidate_reduction = (
            (1.0 - (candidate_count / total_possible_pairs)) * 100.0
            if total_possible_pairs > 0 else 0.0
        )

        # 4. Verify candidate pairs via MinHash Jaccard similarity & merge with Union-Find
        uf = UnionFind([p.page_id for p in pages])
        verified_pairs_count = 0
        pair_similarities: Dict[Tuple[int, int], float] = {}

        for p1_id, p2_id in candidate_pairs:
            mh1 = page_minhashes[p1_id]
            mh2 = page_minhashes[p2_id]
            sim = mh1.jaccard(mh2)

            if not verify_similarity or sim >= self.similarity_threshold:
                uf.union(p1_id, p2_id)
                verified_pairs_count += 1
                pair_similarities[(p1_id, p2_id)] = sim

        # 5. Extract clusters from Union-Find
        raw_clusters = uf.get_clusters()
        clusters: List[SimilarityCluster] = []
        cluster_id_counter = 0

        multi_page_clustered = 0
        single_page_unclustered = 0

        for root_id, member_ids in raw_clusters.items():
            m_len = len(member_ids)
            if m_len > 1:
                # Multi-page similarity cluster
                # Compute average intra-cluster similarity efficiently
                intra_sims: List[float] = []
                if m_len <= 16:
                    for i in range(m_len):
                        for j in range(i + 1, m_len):
                            p_a = min(member_ids[i], member_ids[j])
                            p_b = max(member_ids[i], member_ids[j])
                            sim = pair_similarities.get(
                                (p_a, p_b),
                                page_minhashes[p_a].jaccard(page_minhashes[p_b])
                            )
                            intra_sims.append(sim)
                else:
                    # Sample up to 40 pairs for large clusters to bound runtime
                    for _ in range(40):
                        i, j = self.rng.choice(m_len, size=2, replace=False)
                        p_a = min(member_ids[i], member_ids[j])
                        p_b = max(member_ids[i], member_ids[j])
                        sim = pair_similarities.get(
                            (p_a, p_b),
                            page_minhashes[p_a].jaccard(page_minhashes[p_b])
                        )
                        intra_sims.append(sim)

                avg_sim = float(np.mean(intra_sims)) if intra_sims else 1.0
                cluster = SimilarityCluster(
                    cluster_id=cluster_id_counter,
                    page_ids=member_ids,
                    avg_intra_similarity=avg_sim,
                )
                clusters.append(cluster)
                cluster_id_counter += 1
                multi_page_clustered += m_len
            else:
                single_page_unclustered += 1

        clustering_time_ms = (time.perf_counter() - t0) * 1000.0

        stats = {
            "total_pages_in": n_pages,
            "total_possible_pairs": total_possible_pairs,
            "num_lsh_buckets_active": len(lsh_tables),
            "candidate_pairs_generated": candidate_count,
            "candidate_reduction_pct": candidate_reduction,
            "verified_pairs_count": verified_pairs_count,
            "num_multi_page_clusters": len(clusters),
            "clustered_pages_count": multi_page_clustered,
            "unclustered_pages_count": single_page_unclustered,
            "clustering_time_ms": clustering_time_ms,
        }

        return clusters, stats
