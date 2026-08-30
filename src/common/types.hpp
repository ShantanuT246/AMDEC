#pragma once

#include <cstdint>
#include <cstddef>

namespace amdec {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr size_t PAGE_SIZE = 4096;  // bytes per simulated memory page

// ---------------------------------------------------------------------------
// PageId — unique identifier for a simulated page
// ---------------------------------------------------------------------------
using PageId = uint64_t;

// ---------------------------------------------------------------------------
// Tier — stability classification assigned by Module 1
//   HOT   → written frequently; skip scanning entirely
//   WARM  → moderate activity; low-priority scan candidate
//   COLD  → long-lived, rarely written; high-priority scan candidate
// ---------------------------------------------------------------------------
enum class Tier : uint8_t {
    HOT  = 0,
    WARM = 1,
    COLD = 2,
};

} // namespace amdec
