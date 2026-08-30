#pragma once

#include "types.hpp"
#include <cstdint>
#include <cstddef>

namespace amdec {

// ---------------------------------------------------------------------------
// Page — the fundamental unit of simulation
//
// Represents one 4096-byte anonymous memory page.  All fields except
// `content` are metadata used by the sampling and tiering logic (Module 1).
//
// Memory layout note: content is first so that &page == &page.content,
// which keeps byte-diff hot paths cache-friendly.
// ---------------------------------------------------------------------------
struct Page {
    // --- Content (4096 B) ------------------------------------------------
    uint8_t  content[PAGE_SIZE];   // raw byte content of the page

    // --- Identity --------------------------------------------------------
    PageId   id;                   // unique page identifier

    // --- Temporal metadata -----------------------------------------------
    uint64_t alloc_time;           // simulated tick at allocation
    uint64_t last_write_time;      // simulated tick of most recent write
    uint32_t write_count;          // total number of writes since alloc

    // --- Derived (computed by M1 sampler) --------------------------------
    float    write_rate;           // write_count / age  (writes per tick)
    Tier     tier;                 // HOT / WARM / COLD classification

    // --- Similarity group (set by M2 clustering) -------------------------
    int64_t  cluster_id;           // -1 = unassigned
};

static_assert(offsetof(Page, content) == 0,
    "content must be the first member for hot-path pointer aliasing");

} // namespace amdec