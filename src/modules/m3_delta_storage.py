"""
Module 3 — Base-Delta Storage & Adaptive Patching Manager.

Compresses clusters of near-duplicate memory pages into representative base pages
and compact metadata-aware delta records using XOR differencing + Run-Length Encoding (RLE).
"""

from dataclasses import dataclass, field
import hashlib
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.workload.page_generator import Page, PAGE_SIZE
from src.modules.m2_lsh_clustering import SimilarityCluster


@dataclass
class DeltaSpan:
    """Represents a contiguous non-zero byte difference span."""
    offset: int      # 2-byte uint16 (0 - 4095)
    length: int      # 2-byte uint16 (1 - 4096)
    data: bytes      # raw byte differences


@dataclass
class DeltaRecord:
    """
    Metadata-aware delta record for a compressed page.
    Storage size includes:
    - 4-byte base page reference
    - 2-byte span count header
    - 4 bytes per span (2B offset + 2B length) + data bytes
    """
    page_id: int
    base_page_id: int
    spans: List[DeltaSpan]
    raw_delta_bytes: int

    @property
    def total_storage_bytes(self) -> int:
        """Computes true metadata-aware storage cost in bytes."""
        header_overhead = 4 + 2  # base_ref (4B) + num_spans (2B)
        span_overhead = sum(4 + len(s.data) for s in self.spans)
        return header_overhead + span_overhead


@dataclass
class CompressedCluster:
    """Represents a stored cluster consisting of a base page and delta-compressed pages."""
    cluster_id: int
    base_page_id: int
    base_page_content: bytearray
    deltas: Dict[int, DeltaRecord] = field(default_factory=dict)
    unmerged_full_pages: Dict[int, bytearray] = field(default_factory=dict)
    secondary_bases: Dict[int, bytearray] = field(default_factory=dict)

    @property
    def total_logical_bytes(self) -> int:
        base_cost = (1 + len(self.secondary_bases)) * PAGE_SIZE
        delta_cost = sum(d.total_storage_bytes for d in self.deltas.values())
        unmerged_cost = len(self.unmerged_full_pages) * PAGE_SIZE
        return base_cost + delta_cost + unmerged_cost

    @property
    def uncompressed_bytes(self) -> int:
        total_pages = 1 + len(self.secondary_bases) + len(self.deltas) + len(self.unmerged_full_pages)
        return total_pages * PAGE_SIZE


class BaseDeltaStorageManager:
    """
    Base-Delta Storage & Adaptive Patching Manager (Module 3).

    Parameters:
        max_delta_ratio: Max delta size as fraction of PAGE_SIZE before fallback (default 0.30 = 1228B).
        base_selection_strategy: 'medoid' (min sum of distances), 'first', or 'stablest'.
    """

    def __init__(
        self,
        max_delta_ratio: float = 0.30,      # Default 30% of PAGE_SIZE (1228B threshold)
        base_selection_strategy: str = "medoid",  # "medoid", "stablest", "first"
        seed: int = 42,
    ):
        self.max_delta_ratio = max_delta_ratio
        self.max_delta_bytes = int(PAGE_SIZE * max_delta_ratio)
        self.base_selection_strategy = base_selection_strategy
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compute_xor_diff(self, page_content: bytearray, base_content: bytearray) -> bytearray:
        """Computes raw byte-level XOR diff between target and base."""
        # NumPy vectorized byte-level XOR differencing
        arr_p = np.frombuffer(page_content, dtype=np.uint8)
        arr_b = np.frombuffer(base_content, dtype=np.uint8)
        xor_arr = np.bitwise_xor(arr_p, arr_b)
        return bytearray(xor_arr.tobytes())

    def encode_rle_spans(self, xor_diff: bytearray) -> List[DeltaSpan]:
        """Encodes contiguous non-zero XOR diff bytes into compact spans."""
        spans: List[DeltaSpan] = []
        n = len(xor_diff)
        i = 0

        while i < n:
            if xor_diff[i] != 0:
                start = i
                while i < n and xor_diff[i] != 0:
                    i += 1
                length = i - start
                spans.append(DeltaSpan(offset=start, length=length, data=bytes(xor_diff[start:i])))
            else:
                i += 1

        return spans

    def reconstruct_page(self, base_content: bytearray, delta: DeltaRecord) -> bytearray:
        """
        Reconstructs the original 4096-byte page from base content + delta record.

        Correctness invariant:
            `reconstructed` starts as a byte-for-byte copy of `base_content`.
            For each span: `reconstructed[i] XOR span.data[i]`
                         = `base[i] XOR xor_diff[i]`
                         = `base[i] XOR (original[i] XOR base[i])`
                         = `original[i]`
        IMPORTANT: Do NOT change `reconstructed[offset:...]` to `base_content[offset:...]`
        in the zip() — they are equivalent only because `reconstructed` is an
        independent copy of `base_content` at this point. If in-place patching is
        ever introduced, this invariant must be re-verified.
        """
        reconstructed = bytearray(base_content)  # independent copy — do not reuse base_content directly
        for span in delta.spans:
            # Apply: reconstructed[i] = base[i] ^ xor_diff[i] = original[i]
            reconstructed[span.offset : span.offset + span.length] = bytes(
                a ^ b for a, b in zip(reconstructed[span.offset : span.offset + span.length], span.data)
            )
        return reconstructed

    def select_base_page(self, pages: List[Page]) -> Page:
        """Selects the representative base page using the configured strategy."""
        if len(pages) == 1:
            return pages[0]

        if self.base_selection_strategy == "medoid":
            # For small clusters, compute exact medoid (min pairwise byte distance)
            if len(pages) <= 16:
                min_dist = float("inf")
                best_page = pages[0]
                for p_cand in pages:
                    cand_arr = np.frombuffer(p_cand.content, dtype=np.uint8)
                    total_dist = 0
                    for p_other in pages:
                        if p_other.page_id != p_cand.page_id:
                            other_arr = np.frombuffer(p_other.content, dtype=np.uint8)
                            total_dist += int(np.count_nonzero(cand_arr != other_arr))
                    if total_dist < min_dist:
                        min_dist = total_dist
                        best_page = p_cand
                return best_page
            else:
                # Sample candidates and evaluate distance against a representative sample of cluster members
                num_cands = min(8, len(pages))
                sample_candidates = self.rng.choice(pages, size=num_cands, replace=False)
                eval_sample = self.rng.choice(pages, size=min(32, len(pages)), replace=False)
                eval_arrays = [np.frombuffer(p.content, dtype=np.uint8) for p in eval_sample]

                min_dist = float("inf")
                best_page = sample_candidates[0]
                for p_cand in sample_candidates:
                    cand_arr = np.frombuffer(p_cand.content, dtype=np.uint8)
                    total_dist = sum(
                        int(np.count_nonzero(cand_arr != other_arr))
                        for other_arr in eval_arrays
                    )
                    if total_dist < min_dist:
                        min_dist = total_dist
                        best_page = p_cand
                return best_page

        elif self.base_selection_strategy == "stablest":
            # Pick page with oldest allocation age and lowest write count
            return min(pages, key=lambda p: (p.write_count, -p.alloc_time))
        else:
            return pages[0]

    def _compute_delta(self, page: Page, base_page: Page) -> DeltaRecord:
        """Computes the DeltaRecord for page vs base_page."""
        # Phase A: Exact-hash fast-track (shift vulnerability: common case bypass)
        # If page content is byte-for-byte identical to base, store a zero-span delta.
        if page.content == base_page.content:
            return DeltaRecord(
                page_id=page.page_id,
                base_page_id=base_page.page_id,
                spans=[],
                raw_delta_bytes=0,
            )
        xor_diff = self.compute_xor_diff(page.content, base_page.content)
        spans = self.encode_rle_spans(xor_diff)
        raw_bytes = sum(len(s.data) for s in spans)
        return DeltaRecord(
            page_id=page.page_id,
            base_page_id=base_page.page_id,
            spans=spans,
            raw_delta_bytes=raw_bytes,
        )

    def compress_cluster(self, cluster: SimilarityCluster, page_map: Dict[int, Page]) -> CompressedCluster:
        """
        Compresses a single similarity cluster into a base page + delta records,
        employing secondary-base fallback for transitively divergent members.
        """
        cluster_pages = [page_map[pid] for pid in cluster.page_ids if pid in page_map]
        if not cluster_pages:
            raise ValueError(f"No pages found for cluster {cluster.cluster_id}")

        base_page = self.select_base_page(cluster_pages)
        compressed = CompressedCluster(
            cluster_id=cluster.cluster_id,
            base_page_id=base_page.page_id,
            base_page_content=bytearray(base_page.content),
        )

        divergent_pages: List[Page] = []

        for page in cluster_pages:
            if page.page_id == base_page.page_id:
                continue

            delta = self._compute_delta(page, base_page)

            # Adaptive Thresholding: If within max ratio, record delta
            if delta.total_storage_bytes <= self.max_delta_bytes:
                compressed.deltas[page.page_id] = delta
            else:
                # Transitive divergence: page exceeds threshold when diffed against primary base
                divergent_pages.append(page)

        # Secondary-Base Fallback (Threat T2 Mitigation):
        # If divergent pages exist, attempt to group them under secondary bases within the cluster
        if divergent_pages:
            remaining_divergent = list(divergent_pages)
            while remaining_divergent:
                # Check if any remaining page matches an already-selected secondary base
                unmatched: List[Page] = []
                for p in remaining_divergent:
                    matched_sec = False
                    for sec_id, sec_content in compressed.secondary_bases.items():
                        sec_dummy_page = Page(
                            page_id=sec_id,
                            region_id=p.region_id,
                            content=sec_content,
                            alloc_time=p.alloc_time,
                            last_write_time=p.last_write_time,
                        )
                        delta_sec = self._compute_delta(p, sec_dummy_page)
                        if delta_sec.total_storage_bytes <= self.max_delta_bytes:
                            compressed.deltas[p.page_id] = delta_sec
                            matched_sec = True
                            break
                    if not matched_sec:
                        unmatched.append(p)

                if len(unmatched) >= 2:
                    # Select secondary base from the unmatched divergent pages
                    sec_base = self.select_base_page(unmatched)
                    compressed.secondary_bases[sec_base.page_id] = bytearray(sec_base.content)
                    unmatched = [p for p in unmatched if p.page_id != sec_base.page_id]
                    remaining_divergent = unmatched
                else:
                    # Single leftover page cannot amortize a base page; store as unmerged full page
                    for p in unmatched:
                        compressed.unmerged_full_pages[p.page_id] = bytearray(p.content)
                    break

        return compressed

    def compress_workload(
        self,
        clusters: List[SimilarityCluster],
        all_pages: List[Page],
    ) -> Tuple[List[CompressedCluster], Dict[str, Any]]:
        """
        Compresses all clustered pages and accounts for unclustered unique pages.
        Returns:
            Tuple[List[CompressedCluster], storage_metrics]
        """
        t0 = time.perf_counter()
        page_map = {p.page_id: p for p in all_pages}
        compressed_clusters: List[CompressedCluster] = []

        total_base_pages = 0
        total_delta_pages = 0
        total_unmerged_pages = 0
        total_delta_bytes = 0
        delta_sizes: List[int] = []
        global_base_registry: Dict[str, int] = {}   # sha256(base) -> cluster_id of canonical base
        deduped_base_count = 0

        clustered_page_ids = set()

        for cluster in clusters:
            cc = self.compress_cluster(cluster, page_map)

            # Global base deduplication for primary base
            base_fingerprint = hashlib.sha256(bytes(cc.base_page_content)).hexdigest()
            if base_fingerprint in global_base_registry:
                deduped_base_count += 1
                count_this_base = 0
            else:
                global_base_registry[base_fingerprint] = cc.cluster_id
                count_this_base = 1

            # Account for secondary bases
            sec_base_count = 0
            for sec_content in cc.secondary_bases.values():
                sec_fp = hashlib.sha256(bytes(sec_content)).hexdigest()
                if sec_fp in global_base_registry:
                    deduped_base_count += 1
                else:
                    global_base_registry[sec_fp] = cc.cluster_id
                    sec_base_count += 1

            compressed_clusters.append(cc)
            total_base_pages += (count_this_base + sec_base_count)
            total_delta_pages += len(cc.deltas)
            total_unmerged_pages += len(cc.unmerged_full_pages)

            for d in cc.deltas.values():
                d_size = d.total_storage_bytes
                total_delta_bytes += d_size
                delta_sizes.append(d_size)

            clustered_page_ids.update(cluster.page_ids)

        # Unique / Singleton unclustered pages
        unclustered_page_ids = set(page_map.keys()) - clustered_page_ids
        unclustered_count = len(unclustered_page_ids)

        # Storage Accounting
        baseline_storage_bytes = len(all_pages) * PAGE_SIZE
        amdec_storage_bytes = (
            (total_base_pages * PAGE_SIZE)
            + total_delta_bytes
            + (total_unmerged_pages * PAGE_SIZE)
            + (unclustered_count * PAGE_SIZE)
        )

        savings_pct = (
            ((baseline_storage_bytes - amdec_storage_bytes) / baseline_storage_bytes) * 100.0
            if baseline_storage_bytes > 0 else 0.0
        )
        compression_ratio = (
            baseline_storage_bytes / amdec_storage_bytes
            if amdec_storage_bytes > 0 else 1.0
        )

        duration_ms = (time.perf_counter() - t0) * 1000.0

        metrics = {
            "total_pages": len(all_pages),
            "baseline_storage_bytes": baseline_storage_bytes,
            "amdec_storage_bytes": amdec_storage_bytes,
            "logical_memory_savings_pct": savings_pct,
            "compression_ratio": compression_ratio,
            "total_base_pages": total_base_pages,
            "total_delta_pages": total_delta_pages,
            "total_unmerged_cluster_pages": total_unmerged_pages,
            "total_unclustered_unique_pages": unclustered_count,
            "deduped_base_pages": deduped_base_count,
            "avg_delta_bytes": float(np.mean(delta_sizes)) if delta_sizes else 0.0,
            "median_delta_bytes": float(np.median(delta_sizes)) if delta_sizes else 0.0,
            "p95_delta_bytes": float(np.percentile(delta_sizes, 95)) if delta_sizes else 0.0,
            "compression_time_ms": duration_ms,
        }

        return compressed_clusters, metrics
