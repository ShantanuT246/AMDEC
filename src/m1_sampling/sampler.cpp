#include "sampler.hpp"

namespace amdec {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
Sampler::Sampler(const SamplerConfig& cfg) : cfg_(cfg) {}

// ---------------------------------------------------------------------------
// classifyOne — classify a single page and set page.tier in-place
//
// Decision logic (mirrors pseudocode in Methodology_Architecture_Roadmap):
//
//   age       = current_time - alloc_time
//   write_rate is pre-computed by the generator (or re-derived here)
//
//   if write_rate > threshold_hot         → HOT   (skip)
//   elif age > min_age AND
//        write_rate < threshold_cold      → COLD  (high priority)
//   else                                  → WARM  (low priority)
// ---------------------------------------------------------------------------
Tier Sampler::classifyOne(Page& page) const
{
    uint64_t age = (cfg_.current_time > page.alloc_time)
                   ? (cfg_.current_time - page.alloc_time)
                   : 1;

    // Recompute write_rate from ground-truth metadata so classification
    // is not dependent on the generator's pre-computed field.
    float rate = static_cast<float>(page.write_count) / static_cast<float>(age);
    page.write_rate = rate;

    Tier t;
    if (rate > cfg_.threshold_hot) {
        t = Tier::HOT;
    } else if (age > cfg_.min_age && rate < cfg_.threshold_cold) {
        t = Tier::COLD;
    } else {
        t = Tier::WARM;
    }

    page.tier = t;
    return t;
}

// ---------------------------------------------------------------------------
// classify — classify all pages and return the priority scan queue
//
// HOT pages are skipped (not enqueued).
// COLD pages get higher priority than WARM pages in the max-heap
// (ScanEntry::operator< is defined so COLD beats WARM).
// ---------------------------------------------------------------------------
std::priority_queue<ScanEntry> Sampler::classify(std::vector<Page>& pages)
{
    stats_ = {};  // reset metrics

    std::priority_queue<ScanEntry> queue;

    for (Page& p : pages) {
        Tier t = classifyOne(p);

        switch (t) {
            case Tier::HOT:
                ++stats_.n_hot;
                break;  // explicitly skipped — no enqueue
            case Tier::WARM:
                ++stats_.n_warm;
                queue.push(ScanEntry{p.id, Tier::WARM});
                break;
            case Tier::COLD:
                ++stats_.n_cold;
                queue.push(ScanEntry{p.id, Tier::COLD});
                break;
        }
    }

    stats_.n_queued = stats_.n_warm + stats_.n_cold;
    return queue;
}

} // namespace amdec
