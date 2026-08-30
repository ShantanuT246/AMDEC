#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/workload/page_generator.hpp"
#include "../src/m1_sampling/sampler.hpp"

using namespace amdec;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a small workload and run the sampler, returning the pages + stats.
static std::pair<std::vector<Page>, Sampler::Stats>
runSampler(uint64_t n_pages,
           double   frac_exact    = 0.07,
           double   frac_near_dup = 0.80,
           float    threshold_hot = 0.10f,
           float    threshold_cold = 0.01f,
           uint64_t min_age        = 500,
           uint64_t seed           = 42)
{
    WorkloadConfig wc;
    wc.num_pages    = n_pages;
    wc.frac_exact   = frac_exact;
    wc.frac_near_dup = frac_near_dup;
    wc.seed         = seed;
    wc.sim_ticks    = 10'000;
    PageGenerator gen(wc);
    auto pages = gen.generate();

    SamplerConfig sc;
    sc.threshold_hot  = threshold_hot;
    sc.threshold_cold = threshold_cold;
    sc.min_age        = min_age;
    sc.current_time   = wc.sim_ticks;
    Sampler sampler(sc);
    sampler.classify(pages);

    return {std::move(pages), sampler.lastStats()};
}

// ---------------------------------------------------------------------------
// TEST: Hot pages are NEVER enqueued
// ---------------------------------------------------------------------------
TEST_CASE("Hot pages are classified HOT and never queued", "[sampler][m1]")
{
    // Use an extremely low hot threshold so almost all pages are HOT
    WorkloadConfig wc;
    wc.num_pages     = 1000;
    wc.frac_exact    = 0.07;
    wc.frac_near_dup = 0.80;
    wc.seed          = 1;
    wc.sim_ticks     = 10'000;
    PageGenerator gen(wc);
    auto pages = gen.generate();

    SamplerConfig sc;
    sc.threshold_hot  = 0.0f;  // everything with write_rate > 0 is HOT
    sc.threshold_cold = -1.0f; // nothing can be COLD
    sc.min_age        = 0;
    sc.current_time   = wc.sim_ticks;
    Sampler sampler(sc);
    auto queue = sampler.classify(pages);

    // All pages have write_rate > 0, so all should be HOT
    auto stats = sampler.lastStats();
    REQUIRE(stats.n_hot > 0);
    REQUIRE(stats.n_queued == 0);
    REQUIRE(queue.empty());

    // Verify page.tier is correctly set
    for (const auto& p : pages) {
        REQUIRE(p.tier == Tier::HOT);
    }
}

// ---------------------------------------------------------------------------
// TEST: Cold pages rank above warm pages in the priority queue
// ---------------------------------------------------------------------------
TEST_CASE("Cold pages have higher priority than warm pages", "[sampler][m1]")
{
    // Craft two pages manually: one COLD, one WARM
    Page cold_page{};
    cold_page.id          = 0;
    cold_page.write_count = 1;       // very low write rate → COLD
    cold_page.alloc_time  = 0;

    Page warm_page{};
    warm_page.id          = 1;
    warm_page.write_count = 50;      // moderate write rate → WARM
    warm_page.alloc_time  = 9000;

    SamplerConfig sc;
    sc.threshold_hot  = 0.10f;
    sc.threshold_cold = 0.005f;
    sc.min_age        = 500;
    sc.current_time   = 10'000;
    Sampler sampler(sc);

    Tier t_cold = sampler.classifyOne(cold_page);
    Tier t_warm = sampler.classifyOne(warm_page);

    // ScanEntry comparison: COLD (uint8 = 2) > WARM (uint8 = 1)
    ScanEntry cold_entry{cold_page.id, Tier::COLD};
    ScanEntry warm_entry{warm_page.id, Tier::WARM};

    // In max-heap ordering: cold_entry should NOT be less than warm_entry
    REQUIRE_FALSE(cold_entry < warm_entry);
    REQUIRE(warm_entry < cold_entry);

    // Also verify tier assignment
    REQUIRE(t_cold == Tier::COLD);
    REQUIRE(t_warm == Tier::WARM);
}

// ---------------------------------------------------------------------------
// TEST: Population split is approximately correct
// ---------------------------------------------------------------------------
TEST_CASE("Tier distribution roughly matches expected fractions", "[sampler][m1]")
{
    // With default thresholds, most pages should NOT be HOT
    // (the generator creates pages with varied write rates)
    auto [pages, stats] = runSampler(5000);
    uint64_t total = pages.size();

    INFO("HOT: "    << stats.n_hot
      << "  WARM: " << stats.n_warm
      << "  COLD: " << stats.n_cold);

    // Sanity: counts must sum to total
    REQUIRE(stats.n_hot + stats.n_warm + stats.n_cold == total);

    // At least some pages should be COLD and some WARM
    // (exact fractions depend on Zipf distribution but should be non-trivial)
    REQUIRE(stats.n_cold > 0);
    REQUIRE(stats.n_warm > 0);
    // HOT fraction should be a minority (top of Zipf distribution)
    REQUIRE(stats.n_hot < total / 2);
}

// ---------------------------------------------------------------------------
// TEST: All pages have tier set after classify()
// ---------------------------------------------------------------------------
TEST_CASE("Every page has tier set to a valid value after classify()", "[sampler][m1]")
{
    auto [pages, stats] = runSampler(2000);
    for (const auto& p : pages) {
        bool valid = (p.tier == Tier::HOT  ||
                      p.tier == Tier::WARM ||
                      p.tier == Tier::COLD);
        REQUIRE(valid);
    }
}

// ---------------------------------------------------------------------------
// TEST: Stats consistency
// ---------------------------------------------------------------------------
TEST_CASE("Stats n_queued equals n_warm + n_cold", "[sampler][m1]")
{
    auto [pages, stats] = runSampler(3000);
    REQUIRE(stats.n_queued == stats.n_warm + stats.n_cold);
}

// ---------------------------------------------------------------------------
// TEST: write_rate is recomputed and consistent with metadata
// ---------------------------------------------------------------------------
TEST_CASE("classifyOne recomputes write_rate from raw metadata", "[sampler][m1]")
{
    Page p{};
    p.id           = 99;
    p.write_count  = 100;
    p.alloc_time   = 0;
    p.write_rate   = 999.0f;  // deliberately wrong — sampler should overwrite

    SamplerConfig sc;
    sc.current_time = 10'000;
    Sampler sampler(sc);
    sampler.classifyOne(p);

    // write_rate must be recomputed: 100 / 10000 = 0.01
    REQUIRE(p.write_rate == Approx(0.01f).epsilon(1e-4f));
}
