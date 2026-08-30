#pragma once

#include "../common/page.hpp"
#include "../common/types.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <random>

namespace amdec {

// ---------------------------------------------------------------------------
// WorkloadConfig — parameters controlling the synthetic heap workload
// ---------------------------------------------------------------------------
struct WorkloadConfig {
    // Population
    uint64_t num_pages        = 50'000;   // total pages to generate

    // Similarity ratios (must sum to 1.0)
    double   frac_exact       = 0.07;     // fraction of exact duplicates
    double   frac_near_dup    = 0.80;     // fraction of near-duplicates
    // remaining fraction (1 - exact - near_dup) = unique pages

    // Near-duplicate mutation parameters
    uint32_t min_mutations    = 1;        // min byte positions changed per near-dup
    uint32_t max_mutations    = 64;       // max byte positions changed per near-dup

    // Write activity distribution (Zipf-like exponential)
    //   A small fraction of "hot" pages receives most writes.
    double   zipf_skew        = 1.2;      // higher = more skewed toward hot pages
    uint32_t max_writes       = 500;      // upper bound on write_count for any page
    uint64_t sim_ticks        = 10'000;   // simulated time horizon (ticks)

    // Reproducibility
    uint64_t seed             = 42;
};

// ---------------------------------------------------------------------------
// PageGenerator — builds the synthetic anonymous-heap workload
//
// Produces a vector of Page objects whose content and metadata match
// the distributions claimed in the AMDEC paper:
//   - ~7%  exact duplicates
//   - ~80% near-duplicates (differ in ≤ max_mutations byte positions)
//   - ~13% unique pages
//
// Write counts follow a Zipf-like distribution: a small set of "hot" pages
// accumulates most writes, while the bulk of pages are cold.
// ---------------------------------------------------------------------------
class PageGenerator {
public:
    explicit PageGenerator(const WorkloadConfig& cfg = {});

    // Generate all pages according to cfg.  Returns by value; RVO applies.
    std::vector<Page> generate();

    const WorkloadConfig& config() const { return cfg_; }

private:
    WorkloadConfig      cfg_;
    std::mt19937_64     rng_;

    // Fill page content with a random 4096-byte base template.
    void fillRandom(uint8_t* content);

    // Copy base into dst then flip exactly `n_mutations` random byte positions.
    void applyMutations(uint8_t* dst, const uint8_t* base, uint32_t n_mutations);

    // Sample write_count for page index i using a Zipf-like distribution.
    uint32_t sampleWriteCount(uint64_t page_rank, uint64_t total);

    // Derive last_write_time from write_count (uniform spread over sim_ticks).
    uint64_t sampleLastWriteTime(uint32_t write_count);
};

} // namespace amdec
