#include "page_generator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace amdec {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
PageGenerator::PageGenerator(const WorkloadConfig& cfg)
    : cfg_(cfg), rng_(cfg.seed)
{
    double total = cfg_.frac_exact + cfg_.frac_near_dup;
    if (total > 1.0 + 1e-9) {
        throw std::invalid_argument(
            "frac_exact + frac_near_dup must not exceed 1.0");
    }
}

// ---------------------------------------------------------------------------
// generate() — main entry point
//
// Layout of output vector:
//   [0 .. n_exact)              exact duplicates of a single base template
//   [n_exact .. n_near_dup)     near-duplicates (mutated variants of base)
//   [n_near_dup .. num_pages)   unique pages (independent random content)
//
// Write counts are Zipf-distributed: pages at lower indices get more writes
// so that the hot/warm/cold tier distribution is realistic.
// ---------------------------------------------------------------------------
std::vector<Page> PageGenerator::generate()
{
    const uint64_t N          = cfg_.num_pages;
    const uint64_t n_exact    = static_cast<uint64_t>(N * cfg_.frac_exact);
    const uint64_t n_near_dup = static_cast<uint64_t>(N * cfg_.frac_near_dup);

    std::vector<Page> pages(N);

    // ------------------------------------------------------------------
    // 1. Generate a single base template for exact/near-dup groups
    // ------------------------------------------------------------------
    uint8_t base_template[PAGE_SIZE];
    fillRandom(base_template);

    // ------------------------------------------------------------------
    // 2. Exact duplicates — bitwise copies of base_template
    // ------------------------------------------------------------------
    for (uint64_t i = 0; i < n_exact; ++i) {
        Page& p = pages[i];
        p.id    = i;
        std::memcpy(p.content, base_template, PAGE_SIZE);
    }

    // ------------------------------------------------------------------
    // 3. Near-duplicates — mutate base_template at random byte offsets
    // ------------------------------------------------------------------
    std::uniform_int_distribution<uint32_t> mut_count_dist(
        cfg_.min_mutations, cfg_.max_mutations);

    for (uint64_t i = n_exact; i < n_exact + n_near_dup; ++i) {
        Page& p = pages[i];
        p.id    = i;
        uint32_t n_mut = mut_count_dist(rng_);
        applyMutations(p.content, base_template, n_mut);
    }

    // ------------------------------------------------------------------
    // 4. Unique pages — fully independent random content
    // ------------------------------------------------------------------
    for (uint64_t i = n_exact + n_near_dup; i < N; ++i) {
        Page& p = pages[i];
        p.id    = i;
        fillRandom(p.content);
    }

    // ------------------------------------------------------------------
    // 5. Shuffle so that similar pages are not trivially adjacent
    //    (gives the sampler a realistic ordering challenge)
    // ------------------------------------------------------------------
    std::shuffle(pages.begin(), pages.end(), rng_);

    // Reassign IDs post-shuffle so they are stable indices into the vector
    for (uint64_t i = 0; i < N; ++i) {
        pages[i].id = i;
    }

    // ------------------------------------------------------------------
    // 6. Assign write activity (Zipf-like) and temporal metadata
    //    Lower-index pages in the shuffled order get higher write rates.
    //    This is independent of content — mimicking real heap behaviour.
    // ------------------------------------------------------------------
    for (uint64_t i = 0; i < N; ++i) {
        Page& p          = pages[i];
        p.write_count    = sampleWriteCount(i + 1, N);
        p.alloc_time     = static_cast<uint64_t>(
                               cfg_.sim_ticks * static_cast<double>(i) / N);
        p.last_write_time = sampleLastWriteTime(p.write_count);

        // write_rate = writes per elapsed tick since allocation
        uint64_t age     = (p.last_write_time > p.alloc_time)
                           ? (p.last_write_time - p.alloc_time)
                           : 1;
        p.write_rate     = static_cast<float>(p.write_count) / age;

        // Tier is left unset (Tier::HOT default); Module 1 will classify.
        p.tier           = Tier::HOT;
        p.cluster_id     = -1;
    }

    return pages;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void PageGenerator::fillRandom(uint8_t* content)
{
    // Fill 4096 bytes using 64-bit Mersenne Twister output (fast on arm64)
    uint64_t* words = reinterpret_cast<uint64_t*>(content);
    static_assert(PAGE_SIZE % sizeof(uint64_t) == 0,
                  "PAGE_SIZE must be a multiple of 8");
    for (size_t w = 0; w < PAGE_SIZE / sizeof(uint64_t); ++w) {
        words[w] = rng_();
    }
}

void PageGenerator::applyMutations(uint8_t*       dst,
                                   const uint8_t* base,
                                   uint32_t       n_mutations)
{
    std::memcpy(dst, base, PAGE_SIZE);

    std::uniform_int_distribution<uint32_t> pos_dist(0, PAGE_SIZE - 1);
    std::uniform_int_distribution<uint8_t>  val_dist(0, 255);

    for (uint32_t m = 0; m < n_mutations; ++m) {
        uint32_t pos  = pos_dist(rng_);
        uint8_t  val  = val_dist(rng_);
        dst[pos] = val;
    }
}

uint32_t PageGenerator::sampleWriteCount(uint64_t page_rank, uint64_t total)
{
    // Zipf: expected writes ∝ 1 / rank^skew.
    // Normalised to [0, max_writes].
    double harmonic_N = 0.0;
    // Approximate harmonic number for first `total` terms — use a cap for speed
    uint64_t cap = std::min(total, uint64_t{10000});
    for (uint64_t k = 1; k <= cap; ++k) {
        harmonic_N += 1.0 / std::pow(static_cast<double>(k), cfg_.zipf_skew);
    }

    double weight = (1.0 / std::pow(static_cast<double>(page_rank), cfg_.zipf_skew))
                    / harmonic_N;

    uint32_t count = static_cast<uint32_t>(weight * cfg_.max_writes);
    return std::max(count, uint32_t{1});  // every page has at least 1 write
}

uint64_t PageGenerator::sampleLastWriteTime(uint32_t write_count)
{
    // Spread last write across the simulated tick range proportionally
    double fraction = static_cast<double>(write_count) / cfg_.max_writes;
    // Hot pages were written near the end (recently); cold ones near the start
    uint64_t t = static_cast<uint64_t>(cfg_.sim_ticks * (1.0 - fraction * 0.5));
    std::uniform_int_distribution<uint64_t> jitter(0, cfg_.sim_ticks / 20);
    return std::min(t + jitter(rng_), cfg_.sim_ticks - 1);
}

} // namespace amdec
