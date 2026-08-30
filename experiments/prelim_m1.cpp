// prelim_m1.cpp — Week 2 smoke-test / preliminary metrics binary
//
// Exercises the full generator → M1 sampler pipeline and prints:
//   - Workload composition (exact / near-dup / unique page counts)
//   - Tier distribution (HOT / WARM / COLD)
//   - Candidate reduction ratio  = queued / total
//
// Output is also written to results/raw/prelim_m1_<timestamp>.csv
// for later consumption by analysis/plot_results.py.

#include "src/workload/page_generator.hpp"
#include "src/m1_sampling/sampler.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace amdec;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string timestamp()
{
    auto now  = std::chrono::system_clock::now();
    auto t    = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
    return oss.str();
}

static void writeCsv(const std::string& path,
                     const WorkloadConfig& wc,
                     const SamplerConfig&  sc,
                     const Sampler::Stats& stats,
                     double gen_ms,
                     double classify_ms)
{
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[warn] Could not open " << path << " for writing\n";
        return;
    }

    f << "metric,value\n"
      << "num_pages,"          << wc.num_pages        << "\n"
      << "frac_exact,"         << wc.frac_exact        << "\n"
      << "frac_near_dup,"      << wc.frac_near_dup     << "\n"
      << "threshold_hot,"      << sc.threshold_hot     << "\n"
      << "threshold_cold,"     << sc.threshold_cold    << "\n"
      << "min_age,"            << sc.min_age           << "\n"
      << "n_hot,"              << stats.n_hot          << "\n"
      << "n_warm,"             << stats.n_warm         << "\n"
      << "n_cold,"             << stats.n_cold         << "\n"
      << "n_queued,"           << stats.n_queued       << "\n"
      << "candidate_reduction_ratio,"
          << static_cast<double>(stats.n_queued) / wc.num_pages << "\n"
      << "gen_time_ms,"        << gen_ms               << "\n"
      << "classify_time_ms,"   << classify_ms          << "\n";

    std::cout << "[info] Results written to " << path << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // ------------------------------------------------------------------
    // Configuration — matches roadmap defaults
    // ------------------------------------------------------------------
    WorkloadConfig wc;
    wc.num_pages     = 50'000;
    wc.frac_exact    = 0.07;
    wc.frac_near_dup = 0.80;
    wc.seed          = 42;
    wc.sim_ticks     = 10'000;

    SamplerConfig sc;
    sc.threshold_hot  = 0.10f;
    sc.threshold_cold = 0.01f;
    sc.min_age        = 500;
    sc.current_time   = wc.sim_ticks;

    // ------------------------------------------------------------------
    // Generate workload
    // ------------------------------------------------------------------
    std::cout << "=== AMDEC Preliminary M1 Evaluation ===\n\n";
    std::cout << "Generating " << wc.num_pages << " pages...\n";

    auto t0 = std::chrono::steady_clock::now();
    PageGenerator gen(wc);
    auto pages = gen.generate();
    auto t1 = std::chrono::steady_clock::now();

    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Generation time: " << std::fixed << std::setprecision(1)
              << gen_ms << " ms\n";

    // ------------------------------------------------------------------
    // Run sampler
    // ------------------------------------------------------------------
    std::cout << "\nRunning M1 sampler (classify)...\n";

    Sampler sampler(sc);
    auto t2 = std::chrono::steady_clock::now();
    auto queue = sampler.classify(pages);
    auto t3 = std::chrono::steady_clock::now();

    double classify_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    auto stats = sampler.lastStats();

    // ------------------------------------------------------------------
    // Print results
    // ------------------------------------------------------------------
    double total = static_cast<double>(wc.num_pages);
    std::cout << "\n--- Tier Distribution ---\n"
              << "  HOT  : " << stats.n_hot
              << "  (" << std::fixed << std::setprecision(1)
              << 100.0 * stats.n_hot  / total << "%)\n"
              << "  WARM : " << stats.n_warm
              << "  (" << 100.0 * stats.n_warm / total << "%)\n"
              << "  COLD : " << stats.n_cold
              << "  (" << 100.0 * stats.n_cold / total << "%)\n";

    double reduction = static_cast<double>(stats.n_queued) / total;
    std::cout << "\n--- Candidate Reduction ---\n"
              << "  Queued (WARM+COLD): " << stats.n_queued
              << " / " << wc.num_pages
              << "  (reduction ratio: " << std::setprecision(3)
              << reduction << ")\n";

    std::cout << "\n--- Timing ---\n"
              << "  Workload generation : " << std::setprecision(1) << gen_ms      << " ms\n"
              << "  M1 classification   : " << classify_ms << " ms\n"
              << "  Classify throughput : "
              << static_cast<int>(total / classify_ms * 1e3) << " pages/sec\n";

    // ------------------------------------------------------------------
    // Write CSV
    // ------------------------------------------------------------------
    std::string csv_path = "results/raw/prelim_m1_" + timestamp() + ".csv";
    writeCsv(csv_path, wc, sc, stats, gen_ms, classify_ms);

    return 0;
}
