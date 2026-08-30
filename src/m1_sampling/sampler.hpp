#pragma once

#include "../common/page.hpp"
#include "../common/types.hpp"

#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

namespace amdec {

// ---------------------------------------------------------------------------
// SamplerConfig — tunable thresholds for Module 1
// ---------------------------------------------------------------------------
struct SamplerConfig {
    // A page with write_rate > threshold_hot is classified HOT (skipped).
    float threshold_hot  = 0.10f;  // writes per tick

    // A page with write_rate < threshold_cold AND age > min_age is COLD.
    float threshold_cold = 0.01f;  // writes per tick
    uint64_t min_age     = 500;    // minimum ticks since allocation

    // Current simulation time — must be set before calling classify().
    uint64_t current_time = 10'000;
};

// ---------------------------------------------------------------------------
// ScanEntry — element placed on the priority scan queue
// ---------------------------------------------------------------------------
struct ScanEntry {
    PageId   id;
    Tier     tier;

    // Priority ordering: COLD > WARM > HOT
    // Lower numeric value of Tier = lower priority in a max-heap, so we
    // invert: higher Tier enum value → higher scan priority.
    bool operator<(const ScanEntry& o) const {
        return static_cast<uint8_t>(tier) < static_cast<uint8_t>(o.tier);
    }
};

// ---------------------------------------------------------------------------
// Sampler — Module 1: Region-Aware Sampling & Feedback Engine
//
// Classifies each page into HOT / WARM / COLD based on write activity and
// age, then enqueues COLD and WARM pages onto a priority scan queue.
// HOT pages are skipped entirely (no scan overhead incurred).
//
// The classify() method mutates page.tier in-place and returns a priority
// queue of scan candidates.
// ---------------------------------------------------------------------------
class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg = {});

    // Classify a batch of pages in-place and return the scan queue.
    // Pages classified HOT are not enqueued.
    std::priority_queue<ScanEntry> classify(std::vector<Page>& pages);

    // Classify a single page and return its tier.
    // Updates page.tier in-place.
    Tier classifyOne(Page& page) const;

    // Metrics from the last classify() call
    struct Stats {
        uint64_t n_hot;
        uint64_t n_warm;
        uint64_t n_cold;
        uint64_t n_queued;  // n_warm + n_cold
    };
    Stats lastStats() const { return stats_; }

    const SamplerConfig& config() const { return cfg_; }

private:
    SamplerConfig cfg_;
    Stats         stats_ = {};
};

} // namespace amdec
