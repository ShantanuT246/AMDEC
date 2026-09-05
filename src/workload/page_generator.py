"""
Synthetic Anonymous Memory Workload Generator.

Models realistic heap-like memory page allocations with configurable near-duplicate,
exact duplicate, and unique ratios, along with Zipf-distributed write access patterns.
"""

from dataclasses import dataclass, field
import hashlib
import os
import random
import struct
import time
from typing import Dict, List, Optional, Tuple
# pyrefly: ignore [missing-import]
import numpy as np


PAGE_SIZE = 4096  # Standard 4KB OS memory page


@dataclass
class Page:
    """Represents a simulated 4KB physical/anonymous memory page."""
    page_id: int
    region_id: int
    content: bytearray
    alloc_time: float
    last_write_time: float
    write_count: int = 0
    cow_fault_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.content) != PAGE_SIZE:
            raise ValueError(f"Page content must be exactly {PAGE_SIZE} bytes, got {len(self.content)}")

    @property
    def hash_exact(self) -> str:
        """SHA-256 hash of exact page contents."""
        return hashlib.sha256(self.content).hexdigest()

    def write_bytes(self, offset: int, data: bytes, current_time: float):
        """Simulate in-place write to page memory."""
        if offset + len(data) > PAGE_SIZE:
            raise ValueError(f"Write exceeds page boundary: offset={offset}, len={len(data)}")
        self.content[offset:offset + len(data)] = data
        self.write_count += 1
        self.last_write_time = current_time

    def clone(self, new_page_id: int, new_region_id: Optional[int] = None, current_time: Optional[float] = None) -> "Page":
        """Create an exact duplicate page."""
        t = current_time if current_time is not None else self.alloc_time
        return Page(
            page_id=new_page_id,
            region_id=self.region_id if new_region_id is None else new_region_id,
            content=bytearray(self.content),
            alloc_time=t,
            last_write_time=t,
            write_count=0,
            cow_fault_count=0,
            metadata=dict(self.metadata)
        )


class SyntheticWorkloadGenerator:
    """
    Generates synthetic anonymous heap workloads mirroring real OS/hypervisor heaps:
    - ~80% near-duplicate pages (derived from cluster base templates with small field variations)
    - ~7% exact duplicates
    - ~13% unique / high-entropy pages
    """


    def __init__(
        self,
        seed: int = 42,
        near_dup_ratio: float = 0.80,
        exact_dup_ratio: float = 0.07,
        unique_ratio: float = 0.13,
        num_regions: int = 8,
        zipf_param: float = 1.3,
    ):
        total_ratio = near_dup_ratio + exact_dup_ratio + unique_ratio
        if not (0.999 <= total_ratio <= 1.001):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        self.seed = seed
        self.near_dup_ratio = near_dup_ratio
        self.exact_dup_ratio = exact_dup_ratio
        self.unique_ratio = unique_ratio
        self.num_regions = num_regions
        self.zipf_param = zipf_param

        # Instance-isolated deterministic RNGs
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _create_template_page_content(self, template_type: str = "heap_struct") -> bytearray:
        """Create structured heap-like baseline template (e.g. object array, struct headers, text buffers)."""
        buffer = bytearray(PAGE_SIZE)

        if template_type == "heap_struct":
            # Model an array of 64-byte structs across 4096 bytes (64 structs per page)
            struct_size = 64
            for i in range(PAGE_SIZE // struct_size):
                offset = i * struct_size
                # Struct layout: [8B vtable_ptr, 8B obj_id, 8B timestamp, 16B string tag, 24B payload]
                vtable = 0x7FFF00102000
                obj_id = i + 1000
                timestamp = int(1725000000 + i * 10)
                tag = b"AMDEC_HEAP_NODE_"
                payload = bytes([(i * 7 + j) % 256 for j in range(24)])
                struct_data = struct.pack("<QQQ16s24s", vtable, obj_id, timestamp, tag, payload)
                buffer[offset:offset + struct_size] = struct_data

        elif template_type == "pointer_table":
            # Array of 512 pointers (8 bytes each)
            base_ptr = 0x100080000
            for i in range(PAGE_SIZE // 8):
                ptr_val = base_ptr + (i * 0x80)
                buffer[i*8:(i+1)*8] = struct.pack("<Q", ptr_val)

        else:
            # Semi-sparse buffer with repetitive ASCII strings and deterministic seeded padding
            pattern = b"JSON_PAYLOAD_CACHE_CHUNK_STATIC_HEADER_V1:" + bytes(self.rng.getrandbits(8) for _ in range(32))
            for i in range(0, PAGE_SIZE, len(pattern)):
                chunk_len = min(len(pattern), PAGE_SIZE - i)
                buffer[i:i + chunk_len] = pattern[:chunk_len]

        return buffer

    def _mutate_page_content(self, base_content: bytearray, mutation_rate: float = 0.05) -> bytearray:
        """
        Create a near-duplicate by modifying a few bytes/words (e.g. pointers, counters, timestamps),
        preserving high similarity (e.g. 90-98% identical chunks).
        """
        mutated = bytearray(base_content)
        # Number of 8-byte words to mutate
        num_mutations = max(1, int((PAGE_SIZE // 8) * mutation_rate))

        for _ in range(num_mutations):
            word_idx = self.rng.randint(0, (PAGE_SIZE // 8) - 1)
            offset = word_idx * 8
            # Pick a mutation strategy: change lower bits of pointer/counter
            delta = self.rng.randint(1, 1000)
            orig_val = struct.unpack_from("<Q", mutated, offset)[0]
            new_val = (orig_val + delta) & 0xFFFFFFFFFFFFFFFF
            struct.pack_into("<Q", mutated, offset, new_val)

        return mutated

    def _create_unique_page_content(self) -> bytearray:
        """Create a high-entropy / distinct page content using instance-seeded pseudo-random bytes."""
        return bytearray(self.np_rng.bytes(PAGE_SIZE))

    def generate_workload(
        self,
        num_pages: int = 10000,
        cluster_size_range: Tuple[int, int] = (4, 32),
        sim_start_time: float = 100.0,
    ) -> List[Page]:
        """
        Generates a list of Page objects with the configured exact, near-duplicate, and unique distribution.
        """
        pages: List[Page] = []
        page_id_counter = 0

        num_exact = int(num_pages * self.exact_dup_ratio)
        num_unique = int(num_pages * self.unique_ratio)
        num_near = num_pages - num_exact - num_unique

        # 1. Generate Near-Duplicate Clusters
        near_generated = 0
        cluster_id = 0
        while near_generated < num_near:
            c_size = self.rng.randint(cluster_size_range[0], cluster_size_range[1])
            c_size = min(c_size, num_near - near_generated)
            if c_size <= 0:
                break

            template_type = self.rng.choice(["heap_struct", "pointer_table", "sparse_buffer"])
            base_content = self._create_template_page_content(template_type)
            region_id = self.rng.randint(0, self.num_regions - 1)

            # Alloc time distribution
            base_alloc_time = sim_start_time + self.rng.uniform(0, 50.0)

            for member_idx in range(c_size):
                if member_idx == 0:
                    content = bytearray(base_content)
                else:
                    # Near-duplicate with 0.2% - 1.2% word mutation (1-6 mutated 8-byte fields)
                    mutation_rate = self.rng.uniform(0.002, 0.012)
                    content = self._mutate_page_content(base_content, mutation_rate=mutation_rate)

                page = Page(
                    page_id=page_id_counter,
                    region_id=region_id,
                    content=content,
                    alloc_time=base_alloc_time + self.rng.uniform(0, 5.0),
                    last_write_time=base_alloc_time + self.rng.uniform(0, 5.0),
                    write_count=0,
                    cow_fault_count=0,
                    metadata={"ground_truth_cluster": cluster_id, "type": "near_duplicate"}
                )
                pages.append(page)
                page_id_counter += 1

            near_generated += c_size
            cluster_id += 1

        # 2. Generate Exact Duplicates (cloned from existing or new base templates)
        exact_generated = 0
        while exact_generated < num_exact:
            c_size = min(self.rng.randint(2, 8), num_exact - exact_generated)
            base_content = self._create_template_page_content("heap_struct")
            region_id = self.rng.randint(0, self.num_regions - 1)
            alloc_t = sim_start_time + self.rng.uniform(0, 50.0)

            for _ in range(c_size):
                page = Page(
                    page_id=page_id_counter,
                    region_id=region_id,
                    content=bytearray(base_content),
                    alloc_time=alloc_t,
                    last_write_time=alloc_t,
                    write_count=0,
                    cow_fault_count=0,
                    metadata={"ground_truth_cluster": f"exact_{cluster_id}", "type": "exact_duplicate"}
                )
                pages.append(page)
                page_id_counter += 1

            exact_generated += c_size
            cluster_id += 1

        # 3. Generate Unique / High-Entropy Pages
        for _ in range(num_unique):
            region_id = self.rng.randint(0, self.num_regions - 1)
            alloc_t = sim_start_time + self.rng.uniform(0, 50.0)
            page = Page(
                page_id=page_id_counter,
                region_id=region_id,
                content=self._create_unique_page_content(),
                alloc_time=alloc_t,
                last_write_time=alloc_t,
                write_count=0,
                cow_fault_count=0,
                metadata={"ground_truth_cluster": -1, "type": "unique"}
            )
            pages.append(page)
            page_id_counter += 1

        # Deterministically shuffle list using instance RNG
        self.rng.shuffle(pages)
        return pages

    def simulate_write_traffic(
        self,
        pages: List[Page],
        num_ticks: int = 50,
        writes_per_tick: int = 200,
        current_time: float = 200.0,
    ) -> float:
        """
        Simulates Zipfian write traffic across the pages.
        A small fraction of pages receive the majority of writes (Hot), while most are Cold/Warm.
        Returns the updated simulation timestamp.
        """
        n = len(pages)
        if n == 0:
            return current_time

        # Generate Zipf probabilities
        ranks = np.arange(1, n + 1)
        weights = 1.0 / np.power(ranks, self.zipf_param)
        weights /= weights.sum()

        # Map ranks to pages (so specific pages consistently act as hot vs cold)
        page_indices = list(range(n))

        sim_time = current_time
        for tick in range(num_ticks):
            sim_time += 1.0
            # Sample which pages are written this tick according to Zipf weights
            chosen_indices = self.np_rng.choice(page_indices, size=writes_per_tick, p=weights)
            for idx in chosen_indices:
                p = pages[idx]
                # Write a random 8-byte value at random offset
                rand_offset = (self.rng.randint(0, (PAGE_SIZE // 8) - 1)) * 8
                p.write_bytes(rand_offset, struct.pack("<Q", self.rng.randint(0, 0xFFFFFFFF)), sim_time)

        return sim_time
