"""
Module 1 — Region-Aware Sampling & Feedback Engine.

Classifies memory pages into stability tiers (HOT, WARM, COLD) based on allocation age,
write frequencies, idle times, and dynamic COW-break feedback to drastically reduce
CPU scanning overhead.
"""

from dataclasses import dataclass, field
from enum import Enum
import heapq
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.workload.page_generator import Page


class StabilityTier(Enum):
    HOT = "hot"      # Frequently written or high COW churn -> Skip scan
    WARM = "warm"    # Moderate write rate or young -> Low priority scan
    COLD = "cold"    # Long-lived, rarely written -> High priority scan


@dataclass(order=True)
class PrioritizedCandidate:
    """Prioritized wrapper for candidate pages to feed into downstream LSH engine."""
    priority_score: float  # Lower value = higher priority (for min-heap)
    page_id: int = field(compare=False)
    tier: StabilityTier = field(compare=False)
    page: Page = field(compare=False)


@dataclass
class RegionStats:
    """Aggregated statistics for a memory region / heap arena."""
    region_id: int
    total_pages: int = 0
    total_writes: int = 0
    total_cow_breaks: int = 0
    avg_write_rate: float = 0.0
    volatility_score: float = 0.0


class RegionSamplingEngine:
    """
    Region-Aware Sampling & Feedback Engine (Module 1).

    Filters and prioritizes memory pages before similarity clustering (Module 2).
    """

    def __init__(
        self,
        hot_write_rate_threshold: float = 0.08,    # Writes per simulated second to classify as HOT
        cold_write_rate_threshold: float = 0.01,   # Writes per simulated second to classify as COLD
        min_age_for_cold: float = 10.0,             # Minimum lifetime (seconds) to qualify as COLD
        cow_penalty_weight: float = 2.5,           # Multiplier penalty for pages with COW breaks
        region_volatility_threshold: float = 0.25,  # Region write rate above which region is throttled
    ):
        self.hot_write_rate_threshold = hot_write_rate_threshold
        self.cold_write_rate_threshold = cold_write_rate_threshold
        self.min_age_for_cold = min_age_for_cold
        self.cow_penalty_weight = cow_penalty_weight
        self.region_volatility_threshold = region_volatility_threshold

        # Dynamic feedback tracking
        self.page_cow_breaks: Dict[int, int] = {}
        self.page_quarantine: Dict[int, float] = {}  # page_id -> quarantine_until_time

    def classify_page(self, page: Page, current_time: float) -> Tuple[StabilityTier, float]:
        """
        Classifies an individual page into HOT, WARM, or COLD and computes its priority score.

        Returns:
            Tuple[StabilityTier, priority_score]
            (priority_score: lower is scanned earlier in min-heap)
        """
        # Check if page is currently quarantined due to recent COW break
        if page.page_id in self.page_quarantine:
            if current_time < self.page_quarantine[page.page_id]:
                return StabilityTier.HOT, 999999.0
            else:
                del self.page_quarantine[page.page_id]

        age = max(1.0, current_time - page.alloc_time)
        idle_time = max(0.0, current_time - page.last_write_time)
        cow_breaks = self.page_cow_breaks.get(page.page_id, 0) + page.cow_fault_count

        # Effective write rate adjusted by COW break penalty
        effective_write_count = page.write_count + (cow_breaks * self.cow_penalty_weight)
        write_rate = effective_write_count / age

        # Tier decision logic
        if write_rate >= self.hot_write_rate_threshold:
            tier = StabilityTier.HOT
            priority_score = 1000.0 + write_rate
        elif age >= self.min_age_for_cold and write_rate <= self.cold_write_rate_threshold and idle_time > 5.0:
            tier = StabilityTier.COLD
            # Lower score = higher priority. Reward long age & high idle time, penalize write rate.
            priority_score = (write_rate * 10.0) - (idle_time * 0.1) - (age * 0.05)
        else:
            tier = StabilityTier.WARM
            # Warm pages get intermediate score
            priority_score = 100.0 + (write_rate * 20.0) - (idle_time * 0.05)

        return tier, priority_score

    def analyze_regions(self, pages: Iterable[Page], current_time: float) -> Dict[int, RegionStats]:
        """Calculates region-level aggregate metrics across all active heap arenas."""
        regions: Dict[int, RegionStats] = {}
        for p in pages:
            if p.region_id not in regions:
                regions[p.region_id] = RegionStats(region_id=p.region_id)
            r = regions[p.region_id]
            r.total_pages += 1
            r.total_writes += p.write_count
            r.total_cow_breaks += self.page_cow_breaks.get(p.page_id, 0) + p.cow_fault_count

        for r in regions.values():
            if r.total_pages > 0:
                r.avg_write_rate = r.total_writes / float(r.total_pages)
                # Volatility score accounts for both writes and COW faults per page
                r.volatility_score = (r.total_writes + r.total_cow_breaks * self.cow_penalty_weight) / float(r.total_pages)

        return regions

    def filter_and_prioritize(
        self,
        pages: List[Page],
        current_time: float,
        include_warm: bool = True,
        enable_region_filtering: bool = False,
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[PrioritizedCandidate], Dict[str, int]]:
        """
        Scans page set and filters out HOT pages and (optionally) volatile regions.
        Returns a sorted list of candidate pages and scan statistics.
        """
        region_stats = self.analyze_regions(pages, current_time) if enable_region_filtering else {}
        heap: List[PrioritizedCandidate] = []

        stats = {
            "total_pages": len(pages),
            "hot_pages_skipped": 0,
            "warm_pages_queued": 0,
            "cold_pages_queued": 0,
            "volatile_region_pages_skipped": 0,
        }

        for page in pages:
            # Region-level fast filtering (if enabled)
            if enable_region_filtering:
                r_stat = region_stats.get(page.region_id)
                if r_stat and r_stat.volatility_score >= self.region_volatility_threshold:
                    stats["volatile_region_pages_skipped"] += 1
                    continue

            tier, score = self.classify_page(page, current_time)

            if tier == StabilityTier.HOT:
                stats["hot_pages_skipped"] += 1
            elif tier == StabilityTier.COLD:
                stats["cold_pages_queued"] += 1
                candidate = PrioritizedCandidate(priority_score=score, page_id=page.page_id, tier=tier, page=page)
                heapq.heappush(heap, candidate)
            elif tier == StabilityTier.WARM:
                if include_warm:
                    stats["warm_pages_queued"] += 1
                    candidate = PrioritizedCandidate(priority_score=score, page_id=page.page_id, tier=tier, page=page)
                    heapq.heappush(heap, candidate)
                else:
                    stats["hot_pages_skipped"] += 1

        # Extract sorted list of candidates from heap
        candidates: List[PrioritizedCandidate] = []
        while heap:
            cand = heapq.heappop(heap)
            candidates.append(cand)
            if max_candidates and len(candidates) >= max_candidates:
                break

        return candidates, stats

    def record_cow_fault_feedback(self, page_id: int, current_time: float, quarantine_duration: float = 30.0):
        """
        Feedback trigger: when a merged page experiences a Copy-on-Write write fault,
        penalize the page and temporarily quarantine it to prevent thrashing.
        """
        self.page_cow_breaks[page_id] = self.page_cow_breaks.get(page_id, 0) + 1
        self.page_quarantine[page_id] = current_time + quarantine_duration

    def reset_feedback(self):
        """Clear dynamic feedback tracking state."""
        self.page_cow_breaks.clear()
        self.page_quarantine.clear()
