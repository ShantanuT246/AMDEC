"""
Module 5 — Simulated Copy-on-Write (COW) Fault Handler & Reconstruction Engine.

Manages the simulated page table state machine (FULL_PAGE, BASE_PAGE, DELTA_PAGE),
handles read accesses and COW write faults on compressed pages, measures reconstruction
latency distributions (Mean, Median, P95, P99), and updates dynamic M1 feedback.
"""

from dataclasses import dataclass, field
from enum import Enum
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.workload.page_generator import Page, PAGE_SIZE
from src.modules.m1_sampling import RegionSamplingEngine
from src.modules.m3_delta_storage import BaseDeltaStorageManager, CompressedCluster, DeltaRecord


class PageState(Enum):
    FULL_PAGE = "full_page"      # Uncompressed, private writable page
    BASE_PAGE = "base_page"      # Shared representative base page (read-only)
    DELTA_PAGE = "delta_page"    # Delta-compressed page (read-only, requires reconstruction on write)


@dataclass
class PageTableEntry:
    """Entry in the simulated page table."""
    page_id: int
    state: PageState
    content: Optional[bytearray] = None       # Populated for FULL_PAGE and BASE_PAGE
    delta: Optional[DeltaRecord] = None       # Populated for DELTA_PAGE
    base_page_id: Optional[int] = None        # Populated for DELTA_PAGE
    cluster_id: Optional[int] = None
    read_count: int = 0
    write_count: int = 0
    cow_fault_count: int = 0


class SimulatedCOWHandler:
    """
    Simulated Copy-on-Write (COW) Fault Handler & Reconstruction Engine (Module 5).
    """

    def __init__(
        self,
        delta_manager: Optional[BaseDeltaStorageManager] = None,
        sampling_engine: Optional[RegionSamplingEngine] = None,
    ):
        self.delta_manager = delta_manager or BaseDeltaStorageManager()
        self.sampling_engine = sampling_engine
        self.page_table: Dict[int, PageTableEntry] = {}
        self.cluster_bases: Dict[int, bytearray] = {}  # cluster_id -> base_page_content
        self.page_bases: Dict[int, bytearray] = {}     # base_page_id -> immutable base content
        self.reconstruction_latencies_us: List[float] = []

    def populate_from_compressed_clusters(
        self,
        compressed_clusters: List[CompressedCluster],
        all_pages: List[Page],
    ):
        """Populates the simulated page table from Module 3 compressed cluster outputs."""
        self.page_table.clear()
        self.cluster_bases.clear()
        self.page_bases.clear()
        self.reconstruction_latencies_us.clear()

        page_map = {p.page_id: p for p in all_pages}
        accounted_page_ids = set()

        for cc in compressed_clusters:
            # Register immutable cluster base content
            self.cluster_bases[cc.cluster_id] = bytearray(cc.base_page_content)
            self.page_bases[cc.base_page_id] = bytearray(cc.base_page_content)

            # 1. Register Primary Base Page
            self.page_table[cc.base_page_id] = PageTableEntry(
                page_id=cc.base_page_id,
                state=PageState.BASE_PAGE,
                content=bytearray(cc.base_page_content),
                cluster_id=cc.cluster_id,
            )
            accounted_page_ids.add(cc.base_page_id)

            # Register Secondary Base Pages (Threat T2 sub-clusters)
            for sec_id, sec_content in cc.secondary_bases.items():
                self.page_bases[sec_id] = bytearray(sec_content)
                self.page_table[sec_id] = PageTableEntry(
                    page_id=sec_id,
                    state=PageState.BASE_PAGE,
                    content=bytearray(sec_content),
                    cluster_id=cc.cluster_id,
                )
                accounted_page_ids.add(sec_id)

            # 2. Register Delta Pages
            for page_id, delta in cc.deltas.items():
                self.page_table[page_id] = PageTableEntry(
                    page_id=page_id,
                    state=PageState.DELTA_PAGE,
                    delta=delta,
                    base_page_id=delta.base_page_id if hasattr(delta, "base_page_id") else cc.base_page_id,
                    cluster_id=cc.cluster_id,
                )
                accounted_page_ids.add(page_id)

            # 3. Register Unmerged Pages within Cluster
            for page_id, full_content in cc.unmerged_full_pages.items():
                self.page_table[page_id] = PageTableEntry(
                    page_id=page_id,
                    state=PageState.FULL_PAGE,
                    content=bytearray(full_content),
                    cluster_id=cc.cluster_id,
                )
                accounted_page_ids.add(page_id)

        # 4. Register Unclustered / Unique Pages as FULL_PAGE
        for p in all_pages:
            if p.page_id not in accounted_page_ids:
                self.page_table[p.page_id] = PageTableEntry(
                    page_id=p.page_id,
                    state=PageState.FULL_PAGE,
                    content=bytearray(p.content),
                    cluster_id=-1,
                )

    def read_page(self, page_id: int) -> bytearray:
        """
        Simulates read access to a page.
        If DELTA_PAGE, reconstructs content from base + delta and logs latency.
        Returns a defensive bytearray copy to prevent out-of-band page corruption.
        """
        if page_id not in self.page_table:
            raise KeyError(f"Page #{page_id} not found in simulated page table")

        entry = self.page_table[page_id]
        entry.read_count += 1

        if entry.state == PageState.FULL_PAGE or entry.state == PageState.BASE_PAGE:
            return bytearray(entry.content)

        elif entry.state == PageState.DELTA_PAGE:
            base_content = self.page_bases.get(entry.base_page_id, self.cluster_bases.get(entry.cluster_id))
            t0 = time.perf_counter_ns()
            reconstructed = self.delta_manager.reconstruct_page(base_content, entry.delta)
            lat_us = (time.perf_counter_ns() - t0) / 1000.0
            self.reconstruction_latencies_us.append(lat_us)
            return reconstructed

        else:
            raise RuntimeError(f"Unknown page state {entry.state}")

    def handle_write_fault(
        self,
        page_id: int,
        offset: int,
        data: bytes,
        current_time: float,
    ) -> bytearray:
        """
        Simulates a write access to a page, triggering Copy-on-Write (COW) fault
        if the page is stored in compressed DELTA_PAGE or shared BASE_PAGE state.

        Returns:
            The modified private page bytearray.
        """
        if page_id not in self.page_table:
            raise KeyError(f"Page #{page_id} not found in simulated page table")

        entry = self.page_table[page_id]
        entry.write_count += 1

        # Case 1: Page is already private FULL_PAGE (Fast Path)
        if entry.state == PageState.FULL_PAGE:
            entry.content[offset : offset + len(data)] = data
            return entry.content

        # Case 2: Page is DELTA_PAGE -> Copy-on-Write Write Fault
        elif entry.state == PageState.DELTA_PAGE:
            entry.cow_fault_count += 1
            base_content = self.page_bases.get(entry.base_page_id, self.cluster_bases.get(entry.cluster_id))

            # Reconstruct page from immutable base + delta
            t0 = time.perf_counter_ns()
            reconstructed = self.delta_manager.reconstruct_page(base_content, entry.delta)
            lat_us = (time.perf_counter_ns() - t0) / 1000.0
            self.reconstruction_latencies_us.append(lat_us)

            # Allocate private page copy, apply write, and update state to FULL_PAGE
            private_content = bytearray(reconstructed)
            private_content[offset : offset + len(data)] = data

            entry.state = PageState.FULL_PAGE
            entry.content = private_content
            entry.delta = None
            entry.base_page_id = None

            # Feedback trigger to Module 1
            if self.sampling_engine is not None:
                self.sampling_engine.record_cow_fault_feedback(page_id, current_time)

            return entry.content

        # Case 3: Page is BASE_PAGE written to
        elif entry.state == PageState.BASE_PAGE:
            entry.cow_fault_count += 1
            # Allocate private writable copy for this writer (leaving cluster_bases intact)
            private_content = bytearray(entry.content)
            private_content[offset : offset + len(data)] = data

            entry.state = PageState.FULL_PAGE
            entry.content = private_content

            if self.sampling_engine is not None:
                self.sampling_engine.record_cow_fault_feedback(page_id, current_time)

            return entry.content

        else:
            raise RuntimeError(f"Unknown page state {entry.state}")

    def get_raw_latency_samples(self) -> List[float]:
        """Returns the complete list of individual measured reconstruction latency samples in microseconds."""
        return list(self.reconstruction_latencies_us)

    def get_latency_metrics(self) -> Dict[str, float]:
        """Calculates statistical percentiles for page reconstruction latencies."""
        if not self.reconstruction_latencies_us:
            return {
                "count": 0,
                "mean_us": 0.0,
                "median_us": 0.0,
                "p95_us": 0.0,
                "p99_us": 0.0,
                "min_us": 0.0,
                "max_us": 0.0,
            }

        latencies = np.array(self.reconstruction_latencies_us)
        return {
            "count": len(latencies),
            "mean_us": float(np.mean(latencies)),
            "median_us": float(np.median(latencies)),
            "p95_us": float(np.percentile(latencies, 95)),
            "p99_us": float(np.percentile(latencies, 99)),
            "min_us": float(np.min(latencies)),
            "max_us": float(np.max(latencies)),
        }

    def get_page_table_summary(self) -> Dict[str, int]:
        """Returns distribution of page states in page table."""
        summary = {"full_page": 0, "base_page": 0, "delta_page": 0, "total": len(self.page_table)}
        for entry in self.page_table.values():
            summary[entry.state.value] += 1
        return summary
